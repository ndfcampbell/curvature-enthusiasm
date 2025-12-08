import jax
import jax.numpy as jnp


def nn_query(feat_x, feat_y, dim=-2):
    """
    Find correspondences via nearest neighbor query using JAX.

    Args:
        feat_x: Feature vectors of shape x. [V1, C].
        feat_y: Feature vectors of shape y. [V2, C].
        dim: Number of dimension along which to find the nearest neighbor (default: -2).

    Returns:
        p2p: Point-to-point map (shape y -> shape x). [V2].
    """
    dist = jnp.linalg.norm(feat_x[:, None, :] - feat_y[None, :, :], axis=-1)  # [V1, V2]
    p2p = jnp.argmin(dist, axis=dim)  # Nearest neighbor index
    return p2p


def calculate_geodesic_error(dist_x, corr_x, corr_y, p2p, return_mean=True):
    """
    Calculate the geodesic error between predicted correspondence and gt correspondence

    Args:
        dist_x (np.ndarray): Geodesic distance matrix of shape x. shape [Vx, Vx]
        corr_x (np.ndarray): Ground truth correspondences of shape x. shape [V]
        corr_y (np.ndarray): Ground truth correspondences of shape y. shape [V]
        p2p (np.ndarray): Point-to-point map (shape y -> shape x). shape [Vy]
        return_mean (bool, optional): Average the geodesic error. Default True.
    Returns:
        avg_geodesic_error (np.ndarray): Average geodesic error.
    """
    ind21 = jnp.stack([corr_x, p2p[corr_y]], axis=-1)
    ind21 = jnp.ravel_multi_index(ind21.T, dims=[dist_x.shape[0], dist_x.shape[0]])
    geo_err = jnp.take(dist_x, ind21)
    if return_mean:
        return geo_err.mean()
    else:
        return geo_err

def calculate_dense_correspondece_matches(shape_x, shape_y, corr_x, corr_y, dist_x):
    nn_indices = nn_query(shape_x, shape_y)
    geo_err = calculate_geodesic_error(dist_x, corr_x, corr_y, nn_indices, return_mean=False)
    count_zeros = jnp.count_nonzero(geo_err == 0.0)
    return count_zeros / len(geo_err), geo_err
