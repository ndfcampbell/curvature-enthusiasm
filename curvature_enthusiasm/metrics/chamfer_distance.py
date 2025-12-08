import jax
import jax.numpy as jnp

from pykeops.torch import LazyTensor
from torch2jax import torch2jax

def keops_squared_distances(x, y):
    """
    source, target -> squared distances
    (B, N, D), (B, M, D) -> (B, N, M)
    """
    B, N, D = x.shape # Batch size, number of source points, features
    _, M, _ = y.shape # Batch size, number of target points, features

    # Encode as symbolic tensors:
    x_i = LazyTensor(x.view(B, N, 1, D)) # (B, N, 1, D)
    y_j = LazyTensor(y.view(B, 1, M, D)) # (B, 1, M, D)

    # Symbolic matrix of squared distances:
    D_ij = ((x_i - y_j)**2).sum(-1) # (B, N, M), squared distances
    return D_ij

def keops_chamfer_loss(x, y):
    """
    source, target -> loss values
    # (B, N, D), (B, M, D) -> (B,)
    """
    D_ij = keops_squared_distances(x, y) # (B, N, M) symbolic matrix
    D_xy = D_ij.min(dim=2).sqrt() # (B, N), distances from x to y
    D_yx = D_ij.min(dim=1).sqrt() # (B, M), distances from y to x
    return (D_xy.mean(dim=1) + D_yx.mean(dim=1)).view(-1) / 2 # (B,)

def keops_chamfer_distance(x, y):

    cost = torch2jax(
        keops_chamfer_loss,
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        jax.ShapeDtypeStruct(y.shape, y.dtype),
        output_shapes=jax.ShapeDtypeStruct((1,), x.dtype),
    )(x, y)

    return cost

def keops_chamfer_loss(x, y):
    """
    source, target -> squared‐Chamfer loss
    (B, N, D), (B, M, D) -> (B,)
    """
    # 1) get the (B, N, M) symbolic matrix of squared distances
    # D_ij = keops_squared_distances(x, y)
    # # 2) for each x_i find min_j ‖x_i–y_j‖²  →  (B, N)
    # D_xy2 = D_ij.min(dim=2)
    # # 3) for each y_j find min_i ‖x_i–y_j‖²  →  (B, M)
    # D_yx2 = D_ij.min(dim=1)
    # # 4) sum of mean squared distances in both directions, no sqrt, no ½
    # return D_xy2.mean(dim=1) + D_yx2.mean(dim=1)   # → (B,)

    D_ij = keops_squared_distances(x, y)        # (B, N, M)
    D_xy2 = D_ij.min(dim=2)   # shape (B, N)
    D_yx2 = D_ij.min(dim=1)   # shape (B, M)

    # now average over the batch and wrap as a 1‐element tensor
    per_batch = D_xy2.mean(dim=1) + D_yx2.mean(dim=1)
    # 4) mean over batch → scalar, then wrap as (1,)
    return per_batch.mean().view(1)  # → (1,)              # -> (1,)

def j2t_keops_chamfer_loss(x, y):

    cost = torch2jax(
        keops_chamfer_loss,
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        jax.ShapeDtypeStruct(y.shape, y.dtype),
        output_shapes=jax.ShapeDtypeStruct((1,), x.dtype),
    )(x, y)

    return cost


def keops_per_vertex_chamfer_loss(x, y):

    # Convert to LazyTensors for efficient pairwise computation
    D_ij = keops_squared_distances(x, y)

    # Find the minimum for each source vertex
    min_dists = D_ij.min(axis=1).numpy().ravel()  # Shape [N,]

    return min_dists

def j2t_per_vertex_keops_chamfer_loss(x, y):

    cost = torch2jax(
        keops_per_vertex_chamfer_loss,
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        jax.ShapeDtypeStruct(y.shape, y.dtype),
        output_shapes=jax.ShapeDtypeStruct((x.shape[0],), x.dtype),
    )(x, y)

    return cost


def chamfer_distance(x: jnp.ndarray, y: jnp.ndarray, batch_size: bool = True) -> jnp.ndarray:
    """
    Calculate the Chamfer distance between two sets of point clouds, with optional batching.

    Args:
        x: First point cloud of shape (B, N, D) if batched or (N, D) if unbatched
           where B is batch size, N is number of points, and D is dimensionality
        y: Second point cloud of shape (B, M, D) if batched or (M, D) if unbatched
           where M is number of points
        batch_size: If True, expects batched input with shape (B, N, D), else (N, D)

    Returns:
        If batched: Tensor of shape (B,) containing Chamfer distances for each pair
        If unbatched: Scalar Chamfer distance
    """
    if not batch_size:
        # Add batch dimension if unbatched
        x = jnp.expand_dims(x, axis=0)
        y = jnp.expand_dims(y, axis=0)

    # x shape: (B, N, 1, D)
    # y shape: (B, 1, M, D)
    x_expanded = jnp.expand_dims(x, axis=2)
    y_expanded = jnp.expand_dims(y, axis=1)

    # Calculate squared distances
    # Result shape: (B, N, M)
    squared_distances = jnp.sum((x_expanded - y_expanded) ** 2, axis=-1)

    # For each point in x, find distance to nearest point in y
    # Shape: (B, N)
    x_to_y = jnp.min(squared_distances, axis=2)

    # For each point in y, find distance to nearest point in x
    # Shape: (B, M)
    y_to_x = jnp.min(squared_distances, axis=1)

    # Compute mean distances in both directions
    # Shape: (B,)
    forward_distance = jnp.mean(x_to_y, axis=1)
    backward_distance = jnp.mean(y_to_x, axis=1)

    # Chamfer distance is the sum of both directions
    # Shape: (B,)
    chamfer_dist = forward_distance + backward_distance

    if not batch_size:
        # Remove batch dimension if input was unbatched
        chamfer_dist = chamfer_dist[0]

    return chamfer_dist


def chamfer_distance_per_element(x: jnp.ndarray, y: jnp.ndarray, batch_size: bool = True) -> jnp.ndarray:
    """
    Calculate the Chamfer distance between two sets of point clouds, with optional batching.

    Args:
        x: First point cloud of shape (B, N, D) if batched or (N, D) if unbatched
           where B is batch size, N is number of points, and D is dimensionality
        y: Second point cloud of shape (B, M, D) if batched or (M, D) if unbatched
           where M is number of points
        batch_size: If True, expects batched input with shape (B, N, D), else (N, D)

    Returns:
        If batched: Tensor of shape (B,) containing Chamfer distances for each pair
        If unbatched: Scalar Chamfer distance
    """
    if not batch_size:
        x = jnp.expand_dims(x, axis=0)
        y = jnp.expand_dims(y, axis=0)

    x_expanded = jnp.expand_dims(x, axis=2)
    y_expanded = jnp.expand_dims(y, axis=1)
    squared_distances = jnp.sum((x_expanded - y_expanded) ** 2, axis=-1)

    # For each point in x, find distance to nearest point in y
    # Shape: (B, N)
    x_to_y = jnp.min(squared_distances, axis=2)

    # For each point in y, find distance to nearest point in x
    # Shape: (B, M)
    y_to_x = jnp.min(squared_distances, axis=1)

    chamfer_dist = x_to_y, y_to_x  # Return per-vertex distances
    if not batch_size:
        chamfer_dist = (chamfer_dist[0], chamfer_dist[1])
    return chamfer_dist