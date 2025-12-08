from typing import Dict
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from .losses import pre_comp_nc_vec_con, calc_parts_and_weights, calc_parts_and_weights_brdy
from .create_loss_funcs import calc_varifold_compression, calc_nc_compression

from .train import (
    NCCompressedInfo,
    TrainingData,
    setup_triangle_tracker,
    extract_local_vertices
)

def pad_to_size(array, target_size, fill_value=jnp.nan):
    """Pad array to target_size rows."""
    current_size = array.shape[0]
    if current_size >= target_size:
        return array  # Already large enough

    padding_needed = target_size - current_size
    return jnp.pad(
        array,
        ((0, padding_needed), (0, 0)),
        mode='constant',
        constant_values=fill_value
    )

def init_nc_compressed_info(problem_data: Dict) -> NCCompressedInfo:
    has_boundary = bool(problem_data["has_boundary"])
    f_src = np.asarray(problem_data["f_source"])
    f_tgt = np.asarray(problem_data["f_target"])


    template_nc_struct = pre_comp_nc_vec_con(f_src, has_boundary)
    target_nc_struct = pre_comp_nc_vec_con(f_tgt, has_boundary)

    if has_boundary:
        centres, _ = calc_parts_and_weights_brdy(
        problem_data["v_source"], problem_data["f_source"], template_nc_struct
        )
    else:
        centres, _ = calc_parts_and_weights(
        problem_data["v_source"], problem_data["f_source"], template_nc_struct
        )


    return NCCompressedInfo(
        has_boundary=has_boundary,
        template_nc_struct=template_nc_struct,
        target_nc_struct=target_nc_struct,
        nc_source_centres=jnp.asarray(centres),
    )

def prepare_initial_training_data(problem_data: Dict, config, var_dtype) -> TrainingData:
    src_kp_ids = problem_data.get("source_keypoints_ids")
    tgt_kp_ids = problem_data.get("target_keypoints_ids")

    if src_kp_ids is not None and tgt_kp_ids is not None:
        src_kps = problem_data["v_source"][src_kp_ids]
        tgt_kps = problem_data["v_target"][tgt_kp_ids]
    else:
        src_kps = None
        tgt_kps = None


    if not config.compression.compress_source:
        return TrainingData(
            points=problem_data["v_source"][None, :],
            faces=problem_data["f_source"][None, :],
            weights=None,
            source_keypoints=src_kps,
            target_keypoints=tgt_kps,
        )

    v_src = np.asarray(problem_data["v_source"])
    f_src = np.asarray(problem_data["f_source"])
    sigma = config.varifold.sigmas[-1]
    sigma_sph = config.varifold.sigma_sphs[-1]
    K = int(config.compression.compressed_var_source_size)

    props = calc_varifold_compression(v_src, f_src, sigma, sigma_sph, K)
    unique_ids, local_faces = setup_triangle_tracker(f_src, props["idxs"])
    local_verts = extract_local_vertices(v_src, unique_ids)

    local_verts_jax = jnp.asarray(local_verts)
    pad_size = int(local_verts_jax.shape[0] * 1.05)
    local_verts_jax = pad_to_size(jnp.asarray(local_verts), pad_size, 0.0)

    return TrainingData(
        points=local_verts_jax[None, :],
        faces=jnp.asarray(local_faces)[None, :],
        weights=jnp.asarray(props["w_weights"])[None, :],
        source_keypoints=src_kps,
        target_keypoints=tgt_kps,
    )


def update_varifold_compression(
    deformed_verts: jnp.ndarray,
    v_source_np: np.ndarray,
    f_source_np: np.ndarray,
    max_buffer: int,
    config,
):
    deformed_np = np.asarray(deformed_verts)
    sigma = config.varifold.sigmas[-1]
    sigma_sph = config.varifold.sigma_sphs[-1]
    K = int(config.compression.compressed_var_source_size)

    props = calc_varifold_compression(deformed_np, f_source_np, sigma, sigma_sph, K)
    unique_ids, local_faces = setup_triangle_tracker(f_source_np, props["idxs"])
    local_verts = extract_local_vertices(v_source_np, unique_ids)

    local_verts_jax = pad_to_size(jnp.asarray(local_verts), max_buffer, 0.0)

    return (
        local_verts_jax[None, :],
        jnp.asarray(local_faces)[None, :],
        jnp.asarray(props["w_weights"])[None, :],
    )

def update_nc_compression(
    deformed_verts: jnp.ndarray,
    f_source: jnp.ndarray,
    comp_info: NCCompressedInfo,
    config,
):

    has_boundary = comp_info.has_boundary


    if has_boundary:
        centres, weights = calc_parts_and_weights_brdy(
        deformed_verts, f_source, comp_info.template_nc_struct
        )
    else:
        centres, weights = calc_parts_and_weights(
        deformed_verts, f_source, comp_info.template_nc_struct
        )

    centres_np = np.array(centres, copy=True)
    weights_np = np.array(weights, copy=True)

    sigma = config.normal_cycles.sigmas[-1]
    K = int(config.compression.compressed_nc_source_size)

    props = calc_nc_compression(centres_np, weights_np, sigma, K)
    compressed_centres = comp_info.nc_source_centres[props["idxs"]]

    return compressed_centres[None, :], jnp.asarray(props["w_weights"])[None, :]