"""Data loading functions."""
import os
import jax.numpy as jnp
import igl
import trimesh
import numpy as np

from .skeleton_loader import _load_skeleton
from .utils import create_tet_template, load_mesh_pair, preprocess_mesh_pair, subdivide, has_bdry


def load_problem_data(training_data_config, preprocess_data=True, desired_volume=None, resample_tets=True):
    """Load all mesh data and return as dictionary."""
    if desired_volume is None:
        desired_volume = (4.0 * np.pi) / 3.0

    config = training_data_config

    # Load mesh pair
    v_source_np, f_source_np, v_target_np, f_target_np = load_mesh_pair(
        config['dataset_name'],
        config['source_pose'],
        config['target_pose']
    )

    norm = {
        'scale': None,
        'centroid': None
    }

    # Apply preprocessing
    if preprocess_data:
        v_source_np, f_source_np, v_target_np, f_target_np, norm = preprocess_mesh_pair(
            config['dataset_name'], v_source_np, f_source_np, v_target_np, f_target_np, desired_volume
        )

    # Load skeleton (before subdivision)
    ik_template_skeleton = _load_skeleton(
        config['dataset_name'],
        v_source_np,
        norm,
        config['source_pose'],
        config['skeleton_type'],
        config['skeleton_file']
    )

    # Dataset-specific mesh modifications
    if config['dataset_name'] == 'MANO':
        subdivide_levels = config.get('subdivide_levels', 1)
        v_source_np, f_source_np = subdivide(v_source_np, f_source_np, levels=subdivide_levels)
        v_target_np, f_target_np = subdivide(v_target_np, f_target_np, levels=subdivide_levels)

    # Save meshes
    igl.writeOBJ(f'{config["output_dir"]}source.obj', v_source_np, f_source_np)
    igl.writeOBJ(f'{config["output_dir"]}target.obj', v_target_np, f_target_np)

    # Load or create tetrahedra
    tetra_centres = _load_or_create_tetrahedra(
        config['dataset_name'], config['source_pose'], v_source_np, f_source_np,
        config['output_dir'], resample_tets
    )

    # Load keypoints
    src_kp, tgt_kp = _setup_keypoints(config.get('key_points_file'))

    # Load correspondences
    corr_x, corr_y = _setup_correspondences(
        config['corres_mode'],
        config.get('gt_corres_files'),
        v_source_np,
        v_target_np
    )

    # Check if mesh has a boundary
    has_boundary = has_bdry(f_source_np)

    print(f"Mesh has boundary: {has_boundary}")

    # Convert to JAX arrays
    return {
        'v_source': jnp.array(v_source_np),
        'f_source': jnp.array(f_source_np, dtype=config['int_var_dtype']),
        'v_target': jnp.array(v_target_np, dtype=config['var_dtype']),
        'f_target': jnp.array(f_target_np, dtype=config['int_var_dtype']),
        'e_source': jnp.array(igl.edges(f_source_np), dtype=config['int_var_dtype']),
        'tetra_centres': tetra_centres,
        'ik_template_skeleton': ik_template_skeleton,
        'source_keypoints_ids': src_kp,
        'target_keypoints_ids': tgt_kp,
        'corr_x': corr_x,
        'corr_y': corr_y,
        'has_boundary': has_boundary
    }


def _setup_correspondences(corres_mode, gt_corres_files, v_source, v_target):
    """Load correspondences based on mode."""
    if corres_mode == 'files':
        return _load_correspondences_from_files(gt_corres_files)
    elif corres_mode == 'identity':
        assert v_source.shape[0] == v_target.shape[0]
        return jnp.array([]), jnp.array([])
    else:
        return None, None

def _setup_keypoints(key_points_file):
    """Load and validate keypoints, returning None if not provided."""
    if not key_points_file:
        return None, None

    src_kp, tgt_kp = _load_keypoints_from_file(key_points_file)
    if len(src_kp) != len(tgt_kp):
        raise ValueError("Keypoint index arrays must be the same length.")

    return np.asarray(src_kp, dtype=np.int64), np.asarray(tgt_kp, dtype=np.int64)

def _load_or_create_tetrahedra(dataset_name, source_pose, v_source_np, f_source_np, output_dir, resample_tets):
    """Load existing tetrahedra or create new ones."""
    tet_fn = f'data/{dataset_name}/tets/{source_pose}_tets.npz'

    if not os.path.exists(tet_fn) or resample_tets:
        TV, TT, tetra_centres = create_tet_template(v_source_np, f_source_np)
        igl.writeOBJ(f'{output_dir}source_tet_shape.obj', TV, TT)
        return jnp.array(tetra_centres)
    else:
        data = np.load(tet_fn, allow_pickle=True)
        points = data['tetra_centres']

        # Validate tetra centres are within mesh
        tet_mesh_check = trimesh.load_mesh('tmp/source_shape.obj')
        bounds_min, bounds_max = tet_mesh_check.bounds
        inside_bbox = np.all((points >= bounds_min) & (points <= bounds_max), axis=1)
        assert np.all(inside_bbox), "Tetra centres are not within the mesh"

        return jnp.array(points)

def _load_correspondences_from_files(gt_corr_files):
    """Load ground truth correspondences if available."""
    # if dataset_name not in ['FAUST_r', 'SCAPE_r']:
    #     return None, None

    # source_keypoints = source_pose.rsplit("_", 1)[0] + '.vts'
    # target_keypoints = target_pose.rsplit("_", 1)[0] + '.vts'

    # source_corres_fn = f'data/{dataset_name}/corres/{source_keypoints}'
    # target_corres_fn = f'data/{dataset_name}/corres/{target_keypoints}'

    source_corres_fn = gt_corr_files[0]
    target_corres_fn = gt_corr_files[1]

    # check the files exist
    assert os.path.exists(source_corres_fn), f"Ground-truth correspondence file {source_corres_fn} does not exist."
    assert os.path.exists(target_corres_fn), f"Ground-truth correspondence file {target_corres_fn} does not exist."

    corr_x = np.loadtxt(source_corres_fn, dtype=np.int32) - 1
    corr_y = np.loadtxt(target_corres_fn, dtype=np.int32) - 1
    return corr_x, corr_y

# def _load_keypoints_from_file(file_path):
#     """
#     Load sparse keypoint correspondences from a text file.
#
#     File format:
#         # source_id target_id
#         12 34
#         58 77
#         102 94
#
#     Args:
#         file_path (str): Path to keypoints file.
#
#     Returns:
#         source_ids (np.ndarray): Array of source vertex indices.
#         target_ids (np.ndarray): Array of target vertex indices.
#     """
#     src, tgt = [], []
#     with open(file_path, 'r') as f:
#         for line in f:
#             line = line.strip()
#             if not line or line.startswith("#"):
#                 continue
#             s, t = line.split()
#             src.append(int(s))
#             tgt.append(int(t))
#     return np.array(src, dtype=int), np.array(tgt, dtype=int)

import numpy as np

import numpy as np


def _load_keypoints_from_file(file_path):
    """
    Load sparse keypoint correspondences from a text file with a row-based format.

    File format (CSV style):
        Row 1: Metadata/Info (Ignored)
        Row 2: Source Vertex IDs (comma separated)
        Row 3: Target Vertex IDs (comma separated)

    Args:
        file_path (str): Path to keypoints file.

    Returns:
        source_ids (np.ndarray): Array of source vertex indices.
        target_ids (np.ndarray): Array of target vertex indices.

    Raises:
        ValueError: If the file has insufficient rows or if the number of
                    source and target keypoints do not match.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 1. Validate file structure
    if len(lines) < 3:
        raise ValueError(f"Keypoint file '{file_path}' is missing required rows.")

    # Helper to parse a comma-separated line into a list of ints
    def parse_line(line):
        return [int(x.strip()) for x in line.strip().split(',') if x.strip()]

    # 2. Parse rows
    src_list = parse_line(lines[1])
    tgt_list = parse_line(lines[2])

    # 3. Validate data integrity (New Step)
    if len(src_list) != len(tgt_list):
        raise ValueError(
            f"Mismatch in keypoints: Found {len(src_list)} source points "
            f"but {len(tgt_list)} target points in '{file_path}'."
        )

    return np.array(src_list, dtype=int), np.array(tgt_list, dtype=int)