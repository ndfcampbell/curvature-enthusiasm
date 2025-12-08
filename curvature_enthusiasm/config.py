"""Configuration loading and setup."""
import os
from pathlib import Path
import jax.numpy as jnp
import jax.random as jr
import pprint

from .config_manager import load_config, save_resolved_config  # <- from the new code we wrote

# def determine_skeleton_type(config_fn):
#     """Determine skeleton type from config filename."""
#     if 'smpl' in config_fn:
#         return 'SMPL'
#     elif 'skel' in config_fn:
#         return 'SKEL'
#     return 'standard'

def determine_skeleton_type(dataset_name, skeleton_type):
    """Determine which skeleton type to use based on dataset and user input.

    If skeleton_type is "default" or None, use the dataset-specific default.
    """

    dataset_defaults = {
        'MANO': 'mano',
        'DFAUST': 'smpl',
        'FAUST_r': 'smpl',
        'SCAPE_r': 'smpl',
        'SMAL': 'smal',
        '3DSS': 'tgf',
        '3DSS_Hand': 'tgf',
        'TOSCA': 'tgf',
        'TOSCA_h': 'tgf',
    }

    default_type = dataset_defaults.get(dataset_name, 'tgf')

    # If user specifies None or "default", fall back to dataset default
    if skeleton_type is None or skeleton_type == "default":
        print(f"  No skeleton type specified, using default for {dataset_name}: {default_type}")
        return default_type

    # Otherwise, use what was explicitly passed
    return skeleton_type


def create_output_paths(dataset_name, config_fn, source_pose, target_pose):
    """Create output directory path and model filename."""
    suffix = ''
    if 'skel' in config_fn:
        suffix = '_skel'
    elif 'smpl' in config_fn:
        suffix = '_smpl'

    output_dir = f'results/{dataset_name}{suffix}/{dataset_name}_{source_pose}-{target_pose}/'
    model_fn = f'{output_dir}{source_pose}-{target_pose}_nc.eqx'

    os.makedirs(output_dir, exist_ok=True)

    return output_dir, model_fn




def create_traininig_config(
    args,
    random_key,
    *,
    configs_dir: str = "configs",
    save_resolved: bool = False,
):
    """Load complete training configuration."""

    # Split random keys
    ode_key, quat_key, iter_key, predict_key, raw_pred_key = jr.split(random_key, 5)

    # Resolve path and load validated config
    cfg_path = Path(configs_dir) / args.config
    config = load_config(cfg_path)

    print("\n--- Loaded Configuration ---")
    pprint.pprint(config.to_resolved_dict(), sort_dicts=False, width=100)
    print("--- End of Configuration ---\n")

    dataset_name = config.dataset.dataset_name
    skeleton_type = determine_skeleton_type(dataset_name, args.skeleton_type)

    output_dir, model_fn = create_output_paths(dataset_name, args.config, args.start_pose, args.end_pose)

    # Save resolved config if needed
    if save_resolved:
        save_resolved_config(config, Path(output_dir) / "resolved_config.yaml")

    # --- Keypoints lookup ---
    key_points_file = args.key_points_file
    if key_points_file is None:
        default_path = os.path.join("data", dataset_name, "keypoints", f"{args.start_pose}-{args.end_pose}.txt")
        if os.path.exists(default_path):
            key_points_file = default_path
            print(f"No key points file provided, using default: {key_points_file}")
        else:
            print("No key points file provided and none found at default path.")

    # --- Ground-truth correspondence lookup ---
    gt_corres_files = args.gt_corres_files
    if args.corres_mode == 'none' and gt_corres_files is None:
        src_corr = os.path.join("data", dataset_name, "corres", f"{args.start_pose}.vts")
        tgt_corr = os.path.join("data", dataset_name, "corres", f"{args.end_pose}.vts")
        if os.path.exists(src_corr) and os.path.exists(tgt_corr):
            gt_corres_files = [src_corr, tgt_corr]
            print(f"No gt correspondence files provided, using defaults: {src_corr}, {tgt_corr}")
        else:
            print("No gt correspondence files provided and no valid pair found at default path.")

    print(f"Output directory: {output_dir}")
    print(f"Model filename: {model_fn}")

    return {
        "config": config,
        "dataset_name": dataset_name,
        "skeleton_type": skeleton_type,
        "skeleton_file": args.skeleton_file,
        "key_points_file": key_points_file,
        "corres_mode": args.corres_mode,
        "gt_corres_files": gt_corres_files,
        "source_pose": args.start_pose,
        "target_pose": args.end_pose,
        "output_dir": output_dir,
        "model_fn": model_fn,
        "ode_random_key": ode_key,
        "quat_random_key": quat_key,
        "raw_pred_quat_key": raw_pred_key,
        "iter_key": iter_key,
        "predict_key": predict_key,
        "nn_dtype": jnp.float32,
        "var_dtype": jnp.float32,
        "int_var_dtype": jnp.int32,
    }
