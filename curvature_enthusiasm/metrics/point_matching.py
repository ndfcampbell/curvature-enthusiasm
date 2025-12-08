import jax.numpy as jnp
import jax.random as jr
from jax import jit, vmap


def compute_closest_indices(set1, set2):
    """
    Finds the closest point in set1 to each point in set2.
    Returns the indices of the closest points.
    """
    distances = jnp.linalg.norm(set1[:, None, :] - set2[None, :, :], axis=-1)
    closest_indices = jnp.argmin(distances, axis=0)
    return closest_indices


@jit
def compute_match_score(set1, set2):
    """
    Computes the percentage of points in set2 that have their closest point in set1
    at the same index.
    """
    closest_indices = compute_closest_indices(set1, set2)
    correct_matches = jnp.sum(closest_indices == jnp.arange(len(set1)))
    score = correct_matches / len(set1)
    return score

# # Example usage:
# if __name__ == "__main__":
#     key = jr.PRNGKey(0)
#     set1 = jr.normal(key, (100, 3))  # 100 random 3D points
#     set2 = set1 + jr.normal(key, (100, 3)) * 0.2  # Offset with noise
#
#     match_score = compute_match_score(set1, set2)
#     print("Match Score:", match_score)
