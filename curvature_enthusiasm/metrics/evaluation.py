"""Model evaluation functions."""
import numpy as np
import jax.numpy as jnp
import igl
import equinox as eqx

from .misc import calculate_dense_correspondece_matches
from .point_matching import compute_match_score
from .chamfer_distance import keops_chamfer_distance

@eqx.filter_jit
def evaluate_model(model, data, tetra_centres, corr_x, corr_y, dist_x, ode_output_scale, random_key):
    """Run full evaluation on the model."""
    source_points, source_faces, target_points = data
    source_points = jnp.squeeze(source_points)
    source_faces = jnp.squeeze(source_faces)
    target_points = jnp.squeeze(target_points)

    # Forward pass
    sol, _, _ = model(
        source_points=source_points,
        source_keypoints=None,
        tetra_centres=tetra_centres,
        ode_output_scale=ode_output_scale,
        random_key=random_key
    )

    y_pred = sol.y_ode_traj[-1]
    bone_sample_predict = sol.bone_ode_traj[-1]

    # Compute bone trajectory cost
    bone_traj_cost = jnp.mean(
        jnp.square(sol.bone_ode_traj[1:-1] - sol.rigid_rotated_bone_samples_traj[1:-1])
    )

    end_samples_diff_cost = jnp.mean(
        jnp.square(bone_sample_predict - sol.rigid_rotated_bone_samples_traj[-1]),
    )

    # MSE loss (if same number of vertices)
    if y_pred.shape[0] == target_points.shape[0]:
        mse_loss = jnp.mean(jnp.square(y_pred - target_points))
    else:
        mse_loss = 0.0

    # Chamfer distance
    chamfer_dist = jnp.mean(keops_chamfer_distance(
        y_pred[None, :], target_points[None, :]
    ))

    # Correspondence metrics
    if corr_x is None:
        score = 0.0
        geo_err = None
    elif corr_x.size == 0:
        score = compute_match_score(y_pred, target_points)
        geo_err = None
    else:
        # score, geo_err = compute_dense_correspondence(
        #     y_pred, target_points, corr_x, corr_y, dist_x, f_target
        # )
        score = 0.0
        geo_err = None

    return {
        'y_pred': y_pred,
        'mse': mse_loss,
        'chamfer': chamfer_dist,
        'score': score * 100,
        'geo_err': jnp.mean(geo_err) if geo_err is not None else None,
        'bone_traj_cost': bone_traj_cost,
        'end_samples_diff_cost': end_samples_diff_cost
    }


def compute_dense_correspondence(y_pred, target_points, corr_x, corr_y, dist_x, f_target_np):
    """Compute dense correspondence metrics with geodesic scaling."""
    # Calculate surface area for normalization
    double_area = igl.doublearea(np.array(target_points), f_target_np)
    surface_area = np.sum(double_area) / 2.0
    scaling_factor = np.sqrt(surface_area)

    # Compute correspondence
    score, geo_err = calculate_dense_correspondece_matches(
        y_pred, target_points, corr_x, corr_y, dist_x
    )

    # Normalize geodesic error
    geo_err = geo_err / scaling_factor

    return score, geo_err