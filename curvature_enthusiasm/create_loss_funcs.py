"""Loss functions setup and computation.

This module provides the framework for setting up and computing various loss terms
used in anatomical shape registration and deformation. The losses combine geometric
matching, regularization, and physical plausibility constraints.

Main components:
- Varifold loss: Measures shape similarity using oriented surface elements
- Normal cycles loss: Alternative shape matching using normal cycle theory
- ACAP (As-Conformal-As-Possible): Regularizes deformation to be locally conformal
- MDMM ():
- Keypoint matching: Aligns specific anatomical landmarks
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

from .losses import (
    Keops_Varifold_Loss,
    Keops_Normal_Cycles_Loss,
    ACAP_Surface_Energy,
    ACAP_Sample_Energy,
    pre_comp_nc_vec_con,
    calc_parts_and_weights,
    calc_parts_and_weights_brdy,
    extract_varifold_properties
)


from .losses.geometric_measure_losses import calc_varifold_compression_torch, calc_normal_cycle_compression_torch


# IDX / CENTRE / WEIGHTS OF FACES
def calc_varifold_compression(v_np, f_np, sigma, sigma_sph, compressed_size):
    print('Calculating varifold compression...')
    idxs, w_centres, w_normals, w_weights = calc_varifold_compression_torch(v_np, f_np, sigma, sigma_sph, compressed_size, check_quality=True, return_numpy=True)
    return {
        'idxs': jnp.asarray(idxs, dtype=jnp.int32),
        'w_centres': jnp.asarray(w_centres),
        'w_normals': jnp.asarray(w_normals),
        'w_weights': jnp.asarray(w_weights).reshape(-1, 1) # CRUCIAL IT IS THIS SHAPE !!! CAUSING BIG DIFFERENCE IN VARFOLD LOSS OTHERWISE
    }

# IDX / CENTRE / WEIGHTS OF EDGES
def calc_nc_compression(gen_centres, gen_weights, sigma, compressed_size):
    idxs, w_centres, w_weights = calc_normal_cycle_compression_torch(gen_centres, gen_weights, sigma, compressed_size, return_numpy=True)
    return {
        'idxs': jnp.asarray(idxs, dtype=jnp.int32),
        'w_centres': jnp.asarray(w_centres),
        'w_weights': jnp.asarray(w_weights)
    }


def create_varifold_loss(v_target, f_target, varifold_sigma, varifold_sigma_sph, compress_target,
                         compressed_var_target_size):
    """
    Create a varifold loss function for the given target mesh.

    Args:
        v_target: Target vertices
        f_target: Target faces
        varifold_sigma: Spatial kernel scale
        varifold_sigma_sph: Spherical kernel scale
        compress_target: Whether to compress the target
        compressed_var_target_size: Number of centers if compressing

    Returns:
        Keops_Varifold_Loss instance
    """
    v_target_np = np.asarray(v_target)
    f_target_np = np.asarray(f_target)

    if compress_target:
        target_varifold_props = calc_varifold_compression(
            v_target_np, f_target_np, varifold_sigma, varifold_sigma_sph, compressed_var_target_size
        )
    else:
        target_varifold_props = extract_varifold_properties(v_target, f_target)

    return Keops_Varifold_Loss(target_varifold_props)


def create_normal_cycle_loss(f_source, v_target, f_target, has_boundary, compress_target, smallest_nc_sigma, compressed_nc_target_size):

    f_source_np = np.asarray(f_source)
    f_target_np = np.asarray(f_target)

    template_nc_struct = pre_comp_nc_vec_con(f_source_np, has_boundary)
    target_nc_struct = pre_comp_nc_vec_con(f_target_np, has_boundary)

    if has_boundary:
        target_centres, target_weights = calc_parts_and_weights_brdy(
            v_target, f_target, target_nc_struct
        )
    else:
        target_centres, target_weights = calc_parts_and_weights(
            v_target, f_target, target_nc_struct
        )

    if compress_target:
        target_centres_np = np.array(target_centres)
        target_weights_np = np.array(target_weights)
        target_nc_compressed_props = calc_nc_compression(target_centres_np, target_weights_np, smallest_nc_sigma, compressed_nc_target_size)
        target_centres = target_nc_compressed_props['w_centres']
        target_weights = target_nc_compressed_props['w_weights']

    normal_cycle_loss = Keops_Normal_Cycles_Loss(
        template_faces=f_source,
        template_struct=template_nc_struct,
        target_centres=target_centres,
        target_weights=target_weights,
        has_boundary=has_boundary
    )

    return normal_cycle_loss


def setup_losses(mesh_data, config):
    """
    Setup all loss functions required for optimization.

    Loss functions prepared:
    - Varifold Loss:
        Measures geometric discrepancy between prediction and target surfaces
        using varifold metrics. Sensitive to geometry but invariant to mesh
        parametrization.
    - Normal Cycles Loss:
        Matches curvature features (via normal cycles) between source and target.
        Uses a boundary-aware variant if the mesh has boundaries.
    - ACAP Surface Energy:
        As-Conformal-As-Possible penalty on mesh surfaces to discourage
        distortion during deformation.
    - ACAP Sample Energy:
        As-Conformal-As-Possible penalty on volumetric samples (tissue).

    Args:
        mesh_data: Dictionary with source/target vertices, faces, edges,
            and boundary information.
        varifold_config: Holds kernel scales (sigma, sigma_sph) for varifold.
        compression_config: Controls whether varifold compression is applied
            and sample size.

    Returns:
        dict containing:
            'varifold_loss'
            'normal_cycle_loss'
            'acap_surface_energy'
            'acap_sample_energy'
    """

    compress_target = config.compression.compress_target
    compressed_var_target_size = config.compression.compressed_var_target_size
    compressed_nc_target_size = config.compression.compressed_nc_target_size
    varifold_sigma = config.varifold.sigmas[-1]
    varifold_sigma_sph = config.varifold.sigma_sphs[-1]

    smallest_nc_sigma = config.normal_cycles.sigmas[-1]

    # v_source_np = np.asarray(mesh_data['v_source'])
    # f_source_np = np.asarray(mesh_data['f_source'])
    # v_target_np = np.asarray(mesh_data['v_target'])
    # f_target_np = np.asarray(mesh_data['f_target'])

    varifold_loss = create_varifold_loss(
        mesh_data['v_target'],
        mesh_data['f_target'],
        varifold_sigma,
        varifold_sigma_sph,
        compress_target,
        compressed_var_target_size
    )

    normal_cycle_loss = create_normal_cycle_loss(
        mesh_data['f_source'],
        mesh_data['v_target'],
        mesh_data['f_target'],
        mesh_data['has_boundary'],
        compress_target,
        smallest_nc_sigma,
        compressed_nc_target_size
    )


    # Normal cycles loss
    # NEED TO USE NUMPY TO CALC PRE_COMP_NC_VEC_CON
    # has_boundary = mesh_data['has_boundary']
    #
    # template_nc_struct = pre_comp_nc_vec_con(f_source_np, has_boundary)
    # target_nc_struct = pre_comp_nc_vec_con(f_target_np, has_boundary)
    #
    # if has_boundary:
    #     target_centres, target_weights = calc_parts_and_weights_brdy(
    #         mesh_data['v_target'], mesh_data['f_target'], target_nc_struct
    #     )
    # else:
    #     target_centres, target_weights = calc_parts_and_weights(
    #         mesh_data['v_target'], mesh_data['f_target'], target_nc_struct
    #     )
    #
    # if compress_target:
    #     target_centres_np = np.array(target_centres)
    #     target_weights_np = np.array(target_weights)
    #     target_nc_compressed_props = calc_nc_compression(target_centres_np, target_weights_np, smallest_nc_sigma, compressed_nc_target_size)
    #     target_centres = target_nc_compressed_props['w_centres']
    #     target_weights = target_nc_compressed_props['w_weights']
    #
    # normal_cycle_loss = Keops_Normal_Cycles_Loss(
    #     template_faces=mesh_data['f_source'],
    #     template_struct=template_nc_struct,
    #     target_centres=target_centres,
    #     target_weights=target_weights,
    #     has_boundary=has_boundary
    # )

    # ACAP losses
    acap_surface_energy = ACAP_Surface_Energy(mesh_data['e_source'])
    acap_sample_energy = ACAP_Sample_Energy()

    return {
        'varifold_loss': varifold_loss,
        'normal_cycle_loss': normal_cycle_loss,
        'acap_surface_energy': acap_surface_energy,
        'acap_sample_energy': acap_sample_energy,
    }

def _extract_keypoints(keypoint_data):
    """Extract and prepare keypoint data, handling None case."""
    if keypoint_data is not None:
        source_keypoints, target_keypoints = keypoint_data
        return jnp.squeeze(source_keypoints), jnp.squeeze(target_keypoints)
    else:
        return None, None

# Assume TrainingDataState is defined as:
# class TrainingDataState(eqx.Module):
#     vertices: jax.Array  # (N, 3)
#     faces: jax.Array     # (F, 3)
#     weights: jax.Array   # (N, 1) or None
#     source_keypoints: eqx.Pytree # (K, 3) or None
#     target_keypoints: eqx.Pytree # (K, 3) or None
def compute_common_cost_terms(model, data, ode_output_scale, random_key,
                              tetra_centres, bone_ids, n_joints, acap_surface_energy, acap_sample_energy):
    """
    Compute deformation costs that are shared across all loss functions.

    Terms computed:
    - y_pred:
        Final predicted point cloud from ODE integration.
    - Bone trajectory costs:
        * end_samples_mdmm_loss: MDMM penalty aligning predicted bone endpoints
          with rigidly rotated bones at the final timestep.
        * traj_samples_mdmm_loss: MDMM penalty aligning bone trajectories
          with rigid rotations across intermediate timesteps.
    - ACAP energies:
        * point_cloud_acap: As-Conformal-As-Possible penalty on surface point
          cloud deformation.
        * tissue_acap: As-Conformal-As-Possible penalty on volumetric tissue
          trajectories.
    - Keypoint matching:
        Euclidean penalty on matching annotated source/target landmarks
        (zero if keypoints are not provided).
    - Conformal Lipschitz loss:
        Regularization term from the conformal function network
        to prevent unstable deformation flows.

    Args:
        model: Neural ODE model predicting deformation.
        data: Tuple of (source_points, target_points).
        ode_output_scale: Scaling factor for ODE outputs.
        random_key: JAX PRNG key.
        tetra_centres: Tetrahedral sample centres used for tissue tracking.
        source_keypoints: Indices of source landmark points (optional).
        bone_ids: Array of bone indices.
        n_joints: Number of joints in the skeleton.
        acap_surface_energy: ACAP surface energy function.
        acap_sample_energy: ACAP sample energy function.

    Returns:
        dict with:
            'y_pred_source_points'
            'end_samples_mdmm_loss'
            'traj_samples_mdmm_loss'
            'point_cloud_acap'
            'tissue_acap'
            'keypoint_match'
            'conformal_lipschitz_loss'
    """

    # source_points, source_faces, source_weights, source_keypoints, target_keypoints = data

    source_points = jnp.squeeze(data.points)
    source_faces = jnp.squeeze(data.faces)
    source_weights = jnp.squeeze(data.weights).reshape(-1, data.weights.shape[-1]) if data.weights is not None else None
    source_keypoints = jnp.squeeze(data.source_keypoints) if data.source_keypoints is not None else None
    target_keypoints = jnp.squeeze(data.target_keypoints) if data.target_keypoints is not None else None

    # source_keypoints, target_keypoints = _extract_keypoints(keypoint_data)
    n_source = source_points.shape[0]

    # Forward pass
    sol, Q_point_cloud, Q_tissue = model(
        source_points=source_points,
        source_keypoints=source_keypoints,
        tetra_centres=tetra_centres,
        ode_output_scale=ode_output_scale,
        random_key=random_key
    )

    # Extract trajectories (source_points come first in concatenation)
    y_ode_source_points_traj = sol.y_ode_traj[:, :n_source]
    y_pred_source_points = y_ode_source_points_traj[-1]

    # Extract keypoint predictions if present
    if source_keypoints is not None:
        y_ode_keypoints_traj = sol.y_ode_traj[:, n_source:]
        y_pred_keypoints = y_ode_keypoints_traj[-1]
    else:
        y_pred_keypoints = None

    bone_sample_predict = sol.bone_ode_traj[-1]

    # Bone trajectory costs
    bone_traj_cost = jnp.mean(
        jnp.square(sol.bone_ode_traj[1:-1] - sol.rigid_rotated_bone_samples_traj[1:-1]),
        axis=-1
    )

    end_samples_diff_cost = jnp.mean(
        jnp.square(bone_sample_predict - sol.rigid_rotated_bone_samples_traj[-1]),
        axis=1
    )

    # ACAP energies
    point_cloud_acap, _ = acap_surface_energy(
        source_points, y_ode_source_points_traj[1:], Q_point_cloud
    )
    tissue_acap, _, _ = acap_sample_energy(sol.tissue_ode_traj[1:], Q_tissue)
    tissue_acap = jnp.mean(tissue_acap)

    # MDMM losses
    end_samples_diff_cost = jnp.mean(
        end_samples_diff_cost.reshape(n_joints, -1), axis=1
    )
    end_samples_mdmm_scalar = jnp.mean(
        model.end_traj_bone_mdmm(bone_ids, end_samples_diff_cost)
    )

    rigid_sample_diff = jnp.mean(
        bone_traj_cost.reshape(bone_traj_cost.shape[0], model.ik_system.n_bones, -1),
        axis=(0, 2)
    )
    traj_samples_mdmm_scalar = jnp.mean(
        model.traj_bone_mdmm(bone_ids, rigid_sample_diff)
    )

    # Other losses
    conformal_lipschitz_loss = model.conformal_func.mlp.get_lipschitz_loss()

    if source_keypoints is not None:
        keypoint_match = jnp.mean(
            jnp.square(y_pred_keypoints - target_keypoints)
        )
    else:
        keypoint_match = 0.0

    return {
        'y_pred_source_points': y_pred_source_points,
        'source_faces': source_faces,
        'source_weights': source_weights,
        'end_samples_mdmm_loss': end_samples_mdmm_scalar,
        'traj_samples_mdmm_loss': traj_samples_mdmm_scalar,
        'point_cloud_acap': point_cloud_acap,
        'tissue_acap': tissue_acap,
        'keypoint_match': keypoint_match,
        'conformal_lipschitz_loss': conformal_lipschitz_loss,
    }


def create_cost_functions(loss_functions, tetra_centres, ik_template_skeleton):
    """
    Build full cost functions combining geometry and regularization terms.

    Two cost functions are created:

    1. varifold_cost_fn:
        Data fidelity via Varifold loss (geometry distribution).
        + End-sample MDMM penalty
        + ACAP surface/tissue regularization
        + Optional keypoint matching
        + Lipschitz regularization

    2. nc_cost_fn:
        Data fidelity via Normal Cycles loss (curvature distribution).
        + End-sample and trajectory MDMM penalties
        + ACAP surface/tissue regularization
        + Optional keypoint matching
        + Lipschitz regularization

    Args:
        loss_functions: Dictionary with varifold, normal cycles,
            and ACAP energy functions.
        tetra_centres: Tetrahedral sample centres for tissue tracking.
        ik_template_skeleton: Dictionary with template skeleton info
            (joints_positions, bone_edges).

    Returns:
        varifold_cost_fn, nc_cost_fn
        Two callable functions(model, data, ode_output_scale, kernel_params,
                               lambda_weights, random_key, **kwargs) -> scalar loss
    """

    n_joints = int(ik_template_skeleton['joints_positions'].shape[0] - 1)
    bone_ids = jnp.arange(ik_template_skeleton['bone_edges'].shape[0])

    def varifold_cost_fn(model, data, ode_output_scale, kernel_params,
                         lambda_weights, random_key, **kwargs):
        """Varifold-based cost function."""
        common = compute_common_cost_terms(
            model, data, ode_output_scale, random_key,
            tetra_centres,
            bone_ids, n_joints,
            loss_functions['acap_surface_energy'],
            loss_functions['acap_sample_energy']
        )

        var_data_cost = jnp.sum(loss_functions['varifold_loss'](common['y_pred_source_points'], common['source_faces'], kernel_params, template_weights=common['source_weights']))

        return (
                var_data_cost +
                lambda_weights['sample_weights'] * common['end_samples_mdmm_loss'] +
                lambda_weights['pc_acap_coeff'] * common['point_cloud_acap'] +
                lambda_weights['tissue_acap_coeff'] * common['tissue_acap'] +
                lambda_weights['key_points_ep'] * common['keypoint_match'] +
                1.0e-10 * common['conformal_lipschitz_loss']
        )

    def nc_cost_fn(model, data, ode_output_scale, kernel_params,
                   lambda_weights, random_key, **kwargs):
        """Normal cycle based cost function."""
        common = compute_common_cost_terms(
            model, data, ode_output_scale, random_key,
            tetra_centres,
            bone_ids, n_joints,
            loss_functions['acap_surface_energy'],
            loss_functions['acap_sample_energy']
        )

        nc_cost = jnp.sum(loss_functions['normal_cycle_loss'](common['y_pred_source_points'], kernel_params[0], template_weights=common['source_weights']))
        nc_data_cost = jnp.mean(nc_cost)

        return (
                nc_data_cost +
                lambda_weights['sample_weights'] * common['end_samples_mdmm_loss'] +
                lambda_weights['pc_acap_coeff'] * common['point_cloud_acap'] +
                lambda_weights['tissue_acap_coeff'] * common['tissue_acap'] +
                lambda_weights['sample_traj_weight'] * common['traj_samples_mdmm_loss'] +
                lambda_weights['key_points_ep'] * common['keypoint_match'] +
                1.0e-10 * common['conformal_lipschitz_loss']
        )

    return varifold_cost_fn, nc_cost_fn