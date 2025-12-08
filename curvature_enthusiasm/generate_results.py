"""Generate final results and animations."""
import os

import igl
import jax.random as jr
import numpy as np

from .load_test_data import _setup_correspondences
from .metrics import calculate_dense_correspondece_matches, compute_match_score


def generate_results(model, training_data, mesh_data):
    """Generate final results and animations."""
    print("\nGenerating results...")

    # Create output directories
    analytic_dir = f'{training_data["output_dir"]}analytic/'
    animation_dir = f'{training_data["output_dir"]}animation/'
    os.makedirs(analytic_dir, exist_ok=True)
    os.makedirs(animation_dir, exist_ok=True)

    # Generate trajectories
    predict_key_1, predict_key_2 = jr.split(training_data['predict_key'])

    # Analytic trajectory (ODE steps)
    print("Generating analytic trajectory...")
    ode_sol, _, _ = model(
        source_points=mesh_data['v_source'],
        source_keypoints=None,
        tetra_centres=mesh_data['tetra_centres'],
        ode_output_scale=0.01,
        random_key=predict_key_1
    )

    f_source_np = np.array(mesh_data['f_source'])

    save_trajectory(ode_sol, analytic_dir, f_source_np)

    # Animation trajectory (smooth)
    print("Generating animation...")
    n_animation_steps = 100
    ode_sol_animation = model.calc_ode_trajectory(
        mesh_data['v_source'],
        n_animation_steps,
        mesh_data['tetra_centres'],
        0.001,
        predict_key_2
    )
    save_trajectory(ode_sol_animation, animation_dir, f_source_np)

    # Save final metrics
    save_final_metrics(ode_sol, mesh_data, training_data)

    print("Results generation complete!")



def save_trajectory(ode_sol, output_dir, f_source_np):
    """Save ODE trajectory to mesh files."""
    for i in range(ode_sol.y_ode_traj.shape[0]):
        output_fn = f'{output_dir}frame_{i}.obj'
        igl.write_triangle_mesh(
            output_fn,
            np.array(ode_sol.y_ode_traj[i]),
            f_source_np
        )
    print(f"Saved {ode_sol.y_ode_traj.shape[0]} frames to {output_dir}")


def save_final_metrics(ode_sol, mesh_data, training_data):
    """Compute and save final metrics."""
    v_pred_target = np.array(ode_sol.y_ode_traj[-1])
    v_source = mesh_data['v_source']
    v_target = mesh_data['v_target']
    dataset_name = training_data['dataset_name']
    gt_corr_files = training_data['gt_corres_files']

    corr_x, corr_y = _setup_correspondences(
        training_data['corres_mode'],
        training_data['gt_corres_files'],
        v_source,
        v_target
    )

    metrics_file = f'{training_data["output_dir"]}results.txt'

    with open(metrics_file, 'w') as f:
        # if dataset_name in ['FAUST_r', 'SCAPE_r']:
        #     # Load correspondences
        #     # corr_x, corr_y = load_correspondences(
        #     #     dataset_name,
        #     #     training_data['source_pose'],
        #     #     training_data['target_pose']
        #     # )
        #     if gt_corr_files:
        #         corr_x, corr_y = load_correspondences(gt_corr_files)
        #     else:
        #         corr_x = None
        #         corr_y = None

        #     # Compute geodesic distances
        #     from curvature_enthusiasm.metrics import compute_geodesic_distmat
        #     dist_x = compute_geodesic_distmat(
        #         mesh_data['v_source'], mesh_data['f_source']
        #     )
        #
        #     v_target_np = np.array(mesh_data['v_target'])
        #     f_target_np = np.array(mesh_data['f_target'])
        #
        #     # Calculate surface area for scaling
        #     double_area = igl.doublearea(v_target_np, f_target_np)
        #     surface_area = np.sum(double_area) / 2.0
        #     scaling_factor = np.sqrt(surface_area)
        #
        #     score, geo_err = calculate_dense_correspondece_matches(
        #         v_pred_target, v_target, corr_x, corr_y, dist_x
        #     )
        #     geo_err /= scaling_factor
        #
        #     print(f"\nFinal Results:")
        #     print(f"  Dense correspondence: {score * 100:.2f}%")
        #     print(f"  Geodesic error: {np.mean(geo_err):.2e}")
        #
        #     f.write(f"Dense Correspondence %: {score * 100:.2f}%\n")
        #     f.write(f"Av Geodesic error: {np.mean(geo_err):.2e}\n")
        # else:
            # score = compute_match_score(v_pred_target, v_target)

        # Correspondence metrics
        if corr_x is None:
            score = 0.0
            geo_err = None
        elif corr_x.size == 0:
            score = compute_match_score(v_pred_target, v_target)
            geo_err = None
        else:
            # score, geo_err = compute_dense_correspondence(
            #     y_pred, target_points, corr_x, corr_y, dist_x, f_target
            # )
            score = 0.0
            geo_err = None


        print(f"\nFinal Results:")
        print(f"  Match score: {score * 100:.2f}%")
        f.write(f"Match Score: {score * 100:.2f}%\n")

    print(f"Results saved to: {metrics_file}")