"""Skeleton loading utilities with flexible type selection."""
import os

from curvature_enthusiasm.utils.tgf_skeleton_funcs import load_skeleton_tgf, setup_tgf_skeleton
from curvature_enthusiasm.utils.dataset_setup_funcs import (
    setup_smpl_skeleton, load_smpl_model, determine_gender_dfaust,
    setup_mano_skeleton,
    setup_skel_skeleton, load_skel_model,
    setup_smal_skeleton
)

def _load_skeleton(dataset_name, v_source_np, norm,source_pose, effective_type, skeleton_file=None):
    """
    Load skeleton configuration based on type specification.

    Args:
        v_source_np: Source mesh vertices
        dataset_name: Name of dataset (used for defaults)
        source_pose: Source pose identifier
        skeleton_type: One of ['tgf', 'smpl', 'mano', 'skel', None]
            - 'tgf': Load from .tgf file (requires skeleton_file)
            - 'smpl': Generate SMPL skeleton
            - 'mano': Generate MANO skeleton
            - 'skel': Generate SKEL skeleton
            - 'smal': Generate SMAL skeleton
            - None: Use dataset default
        skeleton_file: Path to .tgf skeleton file (required if skeleton_type='tgf')

    Returns:
        Dictionary containing skeleton configuration
    """
    # Determine effective skeleton type
    # effective_type = _determine_skeleton_type(dataset_name, skeleton_type)

    print(f"  Using skeleton type: {effective_type}")

    # Load skeleton based on type
    if effective_type == 'tgf':
        return _load_tgf_skeleton(skeleton_file, norm,source_pose, dataset_name)

    elif effective_type == 'smpl':
        return _load_smpl_skeleton(v_source_np, dataset_name, source_pose)

    elif effective_type == 'mano':
        return _load_mano_skeleton(v_source_np)

    elif effective_type == 'skel':
        return _load_skel_skeleton(v_source_np, dataset_name, source_pose)

    elif effective_type == 'smal':
        return _load_smal_skeleton(v_source_np)

    else:
        raise ValueError(f"Unknown skeleton type: {effective_type}")





def _load_tgf_skeleton(skeleton_file, normalization,source_pose, dataset_name):
    """Load skeleton from .tgf file."""

    # If skeleton_file not provided, try to find default location
    if skeleton_file is None:
        # Try common locations
        possible_paths = [
            f'data/{dataset_name}/skeletons/{source_pose}_skel.tgf',
            f'data/{dataset_name}/{source_pose}_skel.tgf',
            f'data/{dataset_name}/preprocessed/{source_pose}_skel.tgf',
        ]

        skeleton_file = None
        for path in possible_paths:
            if os.path.exists(path):
                skeleton_file = path
                print(f"  Found skeleton file: {skeleton_file}")
                break

        if skeleton_file is None:
            raise FileNotFoundError(
                f"Could not find skeleton file for {source_pose}. Tried:\n" +
                "\n".join(f"  - {p}" for p in possible_paths) +
                "\nPlease specify --skeleton_file explicitly."
            )

    if not os.path.exists(skeleton_file):
        raise FileNotFoundError(f"Skeleton file not found: {skeleton_file}")

    print(f"  Loading TGF skeleton from: {skeleton_file}")
    joint_positions, edges = load_skeleton_tgf(skeleton_file)

    s = normalization.get("scale", None)
    c = normalization.get("centroid", None)
    if s is not None and c is not None:
        joint_positions = (joint_positions - c) * s

    ik_skeleton = setup_tgf_skeleton(joint_positions, edges)

    return ik_skeleton


def _load_mano_skeleton(v_source_np):
    """Load MANO skeleton (for hand datasets)."""
    print(f"  Loading MANO skeleton")
    ik_skeleton = setup_mano_skeleton(v_source_np)
    return ik_skeleton


def _load_smpl_skeleton(v_source_np, dataset_name, source_pose):
    """Load SMPL skeleton (for human body datasets)."""

    # Determine gender if DFAUST dataset
    gender = 'male'  # default
    if dataset_name == 'DFAUST':
        try:
            subject_id = int(source_pose.split('_')[0])
            gender = determine_gender_dfaust(subject_id)
            print(f"  Detected DFAUST subject {subject_id}, gender: {gender}")
        except (ValueError, IndexError):
            print(f"  Could not determine gender from pose name, using default: {gender}")

    print(f"  Loading SMPL skeleton (gender: {gender})")
    joints_positions = load_smpl_model(v_source_np, gender=gender)
    ik_skeleton = setup_smpl_skeleton(v_source_np, joints_positions, gender=gender)

    return ik_skeleton



def _load_skel_skeleton(v_source_np, dataset_name, source_pose):
    """Load SKEL skeleton (alternative SMPL configuration)."""
    print(f"  Loading SKEL skeleton")

    # Determine gender if needed
    gender = 'male'  # default
    if dataset_name == 'DFAUST':
        try:
            subject_id = int(source_pose.split('_')[0])
            gender = determine_gender_dfaust(subject_id)
            print(f"  Detected DFAUST subject {subject_id}, gender: {gender}")
        except (ValueError, IndexError):
            print(f"  Could not determine gender from pose name, using default: {gender}")

    kintree_table, joints_positions, smpl_skel = load_skel_model(v_source_np, gender=gender)
    ik_skeleton = setup_skel_skeleton(v_source_np, kintree_table, joints_positions, smpl_skel)

    return ik_skeleton


def _load_smal_skeleton(v_source_np):
    """Load SMAL skeleton (for animal datasets)."""

    print(f"  Loading SMAL skeleton")
    ik_skeleton = setup_smal_skeleton(v_source_np, ADD_EXTRA_BONES=True)

    return ik_skeleton