import jax.numpy as jnp

def hausdorff_distance(x: jnp.ndarray, y: jnp.ndarray, batch_size: bool = True) -> jnp.ndarray:
    """
    Calculate the Hausdorff distance between two sets of point clouds, with optional batching.
    """
    if not batch_size:
        x = jnp.expand_dims(x, axis=0)
        y = jnp.expand_dims(y, axis=0)

    x_expanded = jnp.expand_dims(x, axis=2)
    y_expanded = jnp.expand_dims(y, axis=1)
    squared_distances = jnp.sum((x_expanded - y_expanded) ** 2, axis=-1)
    x_to_y = jnp.min(squared_distances, axis=2)
    y_to_x = jnp.min(squared_distances, axis=1)
    hausdorff_dist = jnp.maximum(jnp.max(x_to_y, axis=1), jnp.max(y_to_x, axis=1))
    if not batch_size:
        hausdorff_dist = hausdorff_dist[0]
    return hausdorff_dist