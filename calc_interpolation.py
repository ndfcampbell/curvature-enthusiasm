"""Main training script for shape deformation model."""
import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = '.50'


import argparse
import jax
import jax.random as jr
from curvature_enthusiasm import create_traininig_config, run_training, generate_results, load_model_from_file
from curvature_enthusiasm.load_test_data import load_problem_data

import numpy as np
import polyscope as ps

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Training script with pose and configs arguments")

    parser.add_argument("--config", type=str, default="mano_config.yaml",
                        help="Configuration file for the training")
    parser.add_argument("--start_pose", type=str, default='01_01r',
                        help="Starting pose mesh file")
    parser.add_argument("--end_pose", type=str, default='01_02r',
                        help="Ending pose mesh file")
    parser.add_argument("--skeleton_type", type=str, default='default',
                        choices=['default', 'smpl', 'skel', 'tgf'],
                        help="Type of skeleton to use. 'tgf' loads from .tgf file, "
                             "'default'/'smpl'/'skel'/ use procedural generation, "
                             "'default' uses dataset default")
    parser.add_argument("--skeleton_file", type=str, default=None,
                        help="Path to .tgf skeleton file (required if skeleton_type='tgf')")
    parser.add_argument("--key_points_file", type=str, default=None,
                        help="Path to key points file (optional, e.g. sparse correspondences)")
    # New: explicit correspondence mode
    parser.add_argument("--corres_mode", type=str, default="identity",
                        choices=["none", "identity", "files"],
                        help="Dense correspondence mode: 'none' (default), 'identity' (same topology), or 'files' (use .vts pair).")
    # New: ground-truth correspondence files (used when --correspondence=files)
    parser.add_argument("--gt_corres_files", type=str, nargs="*", default=None,
                        help="Pair of .vts files with ground-truth correspondences (source and target). Required if --correspondence=files unless defaults are found.")

    return parser.parse_args()


def main():
    """Main entry point for training."""
    args = parse_arguments()

    # Validate arguments
    if args.skeleton_type == 'tgf' and args.skeleton_file is None:
        raise ValueError("--skeleton_file must be provided when --skeleton_type='tgf'")

    # Validate correspondence mode
    if args.corres_mode == "files" and args.gt_corres_files is not None:
        if len(args.gt_corres_files) != 2:
            raise ValueError("--gt_corres_files must specify exactly two .vts files (source and target).")
        print(f"  Ground-truth correspondence files: {args.gt_corres_files[0]}, {args.gt_corres_files[1]}")

    print(f"Starting training with:")
    print(f"  Start pose: {args.start_pose}")
    print(f"  End pose: {args.end_pose}")
    print(f"  Config file: {args.config}")
    print(f"  Skeleton type: {args.skeleton_type or 'default'}")
    if args.skeleton_file:
        print(f"  Skeleton file: {args.skeleton_file}")

    # Load configuration
    init_random_key = jr.key(42)
    training_data_config = create_traininig_config(args, random_key=init_random_key)

    # Load data (will use preprocessed if available, or fail with helpful message)
    print("Loading data...")
    problem_data = load_problem_data(training_data_config, preprocess_data=True)

    ik_template_skeleton = problem_data['ik_template_skeleton']

    # 'v_source': jnp.array(v_source_np),
    # 'f_source': jnp.array(f_source_np, dtype=config['int_var_dtype']),
    # 'v_target': jnp.array(v_target_np, dtype=config['var_dtype']),
    # 'f_target': jnp.array(f_target_np, dtype=config['int_var_dtype']),
    # 'e_source': jnp.array(igl.edges(f_source_np), dtype=config['int_var_dtype']),
    # 'tetra_centres': tetra_centres,
    # 'ik_template_skeleton': ik_template_skeleton,
    # 'source_keypoints_ids': src_kp,
    # 'target_keypoints_ids': tgt_kp,

    # display the source and target meshes using polyscope
    ps.init()
    ps.set_ground_plane_mode("shadow_only")  # Set ground plane to shadow only
    ps.register_surface_mesh("source", np.array(problem_data['v_source']), np.array(problem_data['f_source']))
    ps.register_surface_mesh("target", np.array(problem_data['v_target']), np.array(problem_data['f_target']))

    # Generate colors for keypoints (one color per keypoint pair)
    n_keypoints = len(problem_data['source_keypoints_ids'])
    # colors = np.random.rand(n_keypoints, 3)  # RGB colors for each keypoint
    # Generate distinct colors using a colormap
    import colorsys
    golden_ratio = 0.618033988749895
    colors = []
    for i in range(n_keypoints):
        hue = (i * golden_ratio) % 1.0
        saturation = 0.9 if i % 2 == 0 else 0.6
        value = 0.85 if i % 3 == 0 else 0.95
        colors.append(colorsys.hsv_to_rgb(hue, saturation, value))
    colors = np.array(colors)

    # display source and target keypoints with matching colors
    source_kpts = ps.register_point_cloud(
        "source_keypoints",
        np.array(problem_data['v_source'][problem_data['source_keypoints_ids']])
    )
    source_kpts.add_color_quantity("colors", colors, enabled=True)

    target_kpts = ps.register_point_cloud(
        "target_keypoints",
        np.array(problem_data['v_target'][problem_data['target_keypoints_ids']])
    )
    target_kpts.add_color_quantity("colors", colors, enabled=True)

    # Joint positions (N, 3)
    joints_positions = np.array(ik_template_skeleton['joints_positions'])

    # Bone edges (M, 2) – assumed 0-based; if 1-based, uncomment next line
    bone_edges = np.array(ik_template_skeleton['bone_edges'], dtype=int)
    # bone_edges = bone_edges - 1  # use this if your indices are 1-based

    print(bone_edges.min(), bone_edges.max())

    # Draw joints as a point cloud
    joint_cloud = ps.register_point_cloud("joints", joints_positions)
    joint_cloud.set_radius(0.005, relative=True)
    joint_cloud.set_color((1.0, 0.2, 0.2))  # optional: reddish joints

    # Draw skeleton as a curve network
    skeleton_net = ps.register_curve_network("skeleton", joints_positions, bone_edges)
    skeleton_net.set_radius(0.003, relative=True)
    skeleton_net.set_color((0.2, 0.6, 1.0))  # optional: bluish bones

    ps.show()


    # Check if we should load from file
    if training_data_config['config'].results.load_from_file:
        print("Loading model from file, skipping training.")
        model = load_model_from_file(training_data_config, problem_data)
    else:
        # Run training
        model = run_training(problem_data, training_data_config)
        print("Training completed successfully!")

    # Generate results
    generate_results(model, training_data_config, problem_data)

    jax.clear_caches()


if __name__ == "__main__":
    main()
