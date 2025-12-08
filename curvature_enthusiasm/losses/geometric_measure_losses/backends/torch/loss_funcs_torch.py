import torch
import numpy as np
from jaxtyping import Array, Float, Int
from typing import Tuple

from ..keops.loss_funcs_keops import varifold_kernel, normal_cycle_kernel


def tri_extr(
        pts: Float[torch.Tensor, "... V 3"],
        simplices: Int[torch.Tensor, "... F 3"],
) -> Tuple[
    Float[torch.Tensor, "... F 3"],
    Float[torch.Tensor, "... F 3"],
    Float[torch.Tensor, "... F 3"]
]:
    """
    Extract triangle vertices from mesh.

    Args:
        pts: Vertex positions, shape [V, 3] or [B, V, 3]
        simplices: Triangle vertex indices, shape [F, 3] or [B, F, 3]

    Returns:
        Tuple of (v1, v2, v3) where each has shape [F, 3] or [B, F, 3]

    Examples:
        >>> pts = torch.randn(100, 3)
        >>> faces = torch.randint(0, 100, (50, 3))
        >>> v1, v2, v3 = tri_extr(pts, faces)
        >>> v1.shape
        torch.Size([50, 3])
    """
    if pts.dim() == 2:  # Unbatched: [V, 3], [F, 3]
        return (
            pts[simplices[:, 0]],
            pts[simplices[:, 1]],
            pts[simplices[:, 2]]
        )

    # Batched: [B, V, 3], [B, F, 3]
    B, F = simplices.shape[:2]

    # More efficient: avoid creating large batch_ix tensor
    batch_idx = torch.arange(B, device=pts.device)[:, None, None]  # [B, 1, 1]

    v1 = pts[batch_idx, simplices[:, :, 0]]  # [B, F, 3]
    v2 = pts[batch_idx, simplices[:, :, 1]]
    v3 = pts[batch_idx, simplices[:, :, 2]]

    return v1, v2, v3


def extract(
        verts: Tuple[
            Float[torch.Tensor, "... F 3"],
            Float[torch.Tensor, "... F 3"],
            Float[torch.Tensor, "... F 3"]
        ]
) -> Tuple[
    Float[torch.Tensor, "... F 3"],  # centres
    Float[torch.Tensor, "... F 3"],  # unit_normals
    Float[torch.Tensor, "... F 1"]  # areas
]:
    """
    Compute triangle properties from vertices.

    Args:
        verts: Tuple of (v1, v2, v3) vertex positions, each shape [..., F, 3]

    Returns:
        centres: Triangle centroids, shape [..., F, 3]
        unit_normals: Unit normal vectors, shape [..., F, 3]
        areas: Triangle areas, shape [..., F, 1]

    Examples:
        >>> v1 = torch.tensor([[0., 0., 0.]])
        >>> v2 = torch.tensor([[1., 0., 0.]])
        >>> v3 = torch.tensor([[0., 1., 0.]])
        >>> c, n, a = extract((v1, v2, v3))
        >>> a.item()  # area = 0.5 for unit right triangle
        0.5
    """
    v1, v2, v3 = verts

    # Centroid
    centres = (v1 + v2 + v3) / 3.0

    # Cross product for normal and area
    u = v2 - v1
    v = v3 - v1
    cross = torch.linalg.cross(u, v, dim=-1)  # [..., F, 3]

    # ORIGINAL CODE
    # -------------
    # Area is half the magnitude of cross product
    cross_norm = torch.linalg.norm(cross, dim=-1, keepdim=True)  # [..., F, 1]
    areas = 0.5 * cross_norm

    # Normalize (handle degenerate triangles)
    eps = torch.finfo(cross.dtype).eps
    unit_normals = cross / cross_norm.clamp_min(eps)
    # ------------------

    # # NEW CODE CODE FOR BETTER GRADIENTS
    # # ---------------------------------
    # # Relative epsilon regularization
    # squared_norm = torch.sum(cross * cross, dim=-1, keepdim=True)
    # eps = torch.finfo(cross.dtype).eps
    # safe_squared_norm = squared_norm + eps * squared_norm.clamp_min(eps)
    # cross_norm = torch.sqrt(safe_squared_norm)
    # areas = 0.5 * cross_norm
    # unit_normals = cross / cross_norm
    # # ---------------------------------

    return centres, unit_normals, areas



def calc_varifold_loss_torch(
    x_centers: Float[Array, "M 3"],
    x_normals: Int[Array, "M 3"],
    x_weights: Float[Array, "M 1"],
    y_centers: Float[Array, "N 3"],
    y_normals: Float[Array, "N 3"],
    y_weights: Float[Array, "N 1"],
    kernel_params: Float[Array, "2"]
) -> Float[Array, "1 1"]:
    """
    Compute the varifold loss between a source mesh and target geometric data.

    This function implements the varifold metric, which is a way to compare
    geometric shapes by treating them as measures on the space of positions
    and orientations. The loss is computed as a Maximum Mean Discrepancy (MMD)
    in the varifold space.

    Args:
        x_centres: Precomputed centers of source shape elements with shape (M, 3)
        x_normaals: Precomputed unit normals of source shape elements with shape (M, 3)
        x_weights: Precomputed areas of source shape elements with shape (M,1)
        y_centers: Precomputed centers of target shape elements with shape (N, 3)
        y_normals: Precomputed unit normals of target shape elements with shape (N, 3)
        y_weights: Precomputed areas of target shape elements with shape (N,1)
        kernel_params: Varifold kernel parameters [sigma_space, sigma_sphere]

    Returns:
        Varifold loss value with shape (1, 1) for compatibility

    Mathematical Details:
        The varifold metric computes:
        L = ⟨μ_X - μ_Y, μ_X - μ_Y⟩_H
        = ⟨μ_X, μ_X⟩_H + ⟨μ_Y, μ_Y⟩_H - 2⟨μ_X, μ_Y⟩_H

        Where μ_X, μ_Y are varifold measures and ⟨·,·⟩_H is the inner product
        in the reproducing kernel Hilbert space defined by the varifold kernel.

    """

    # Compute varifold kernel matrices
    # K(X,X): self-similarity of source shape
    K_xx = varifold_kernel(x_centers, x_normals, x_centers, x_normals, kernel_params)

    # K(Y,Y): self-similarity of target shape
    K_yy = varifold_kernel(y_centers, y_normals, y_centers, y_normals, kernel_params)

    # K(X,Y): cross-similarity between source and target
    K_xy = varifold_kernel(x_centers, x_normals, y_centers, y_normals, kernel_params)

    # Each term corresponds to: ∫∫ K(x,y) dμ(x) dμ(y)
    xx_term = ((K_xx @ x_weights) * x_weights).sum()  # ⟨μ_X, μ_X⟩_H
    yy_term = ((K_yy @ y_weights) * y_weights).sum()  # ⟨μ_Y, μ_Y⟩_H
    xy_term = ((K_xy @ y_weights) * x_weights).sum()  # ⟨μ_X, μ_Y⟩_H

    # MMD² = ⟨μ_X, μ_X⟩ + ⟨μ_Y, μ_Y⟩ - 2⟨μ_X, μ_Y⟩
    varifold_loss = xx_term + yy_term - 2 * xy_term

    return varifold_loss.reshape(1, 1)


def calc_varifold_loss_torch_old_style(
    x_vertices: Float[Array, "V 3"],
    x_faces: Int[Array, "F 3"],
    y_centers: Float[Array, "M 3"],
    y_normals: Float[Array, "M 3"],
    y_weights: Float[Array, "M 1"],
    kernel_params: Float[Array, "2"]
) -> Float[Array, "1 1"]:
    """
    Compute the varifold loss between a source mesh and target geometric data.

    This function implements the varifold metric, which is a way to compare
    geometric shapes by treating them as measures on the space of positions
    and orientations. The loss is computed as a Maximum Mean Discrepancy (MMD)
    in the varifold space.

    Args:
        x_centres: Precomputed centers of source shape elements with shape (M, 3)
        x_normaals: Precomputed unit normals of source shape elements with shape (M, 3)
        x_weights: Precomputed areas of source shape elements with shape (M,1)
        y_centers: Precomputed centers of target shape elements with shape (N, 3)
        y_normals: Precomputed unit normals of target shape elements with shape (N, 3)
        y_weights: Precomputed areas of target shape elements with shape (N,1)
        kernel_params: Varifold kernel parameters [sigma_space, sigma_sphere]

    Returns:
        Varifold loss value with shape (1, 1) for compatibility

    Mathematical Details:
        The varifold metric computes:
        L = ⟨μ_X - μ_Y, μ_X - μ_Y⟩_H
        = ⟨μ_X, μ_X⟩_H + ⟨μ_Y, μ_Y⟩_H - 2⟨μ_X, μ_Y⟩_H

        Where μ_X, μ_Y are varifold measures and ⟨·,·⟩_H is the inner product
        in the reproducing kernel Hilbert space defined by the varifold kernel.

    """

    v1, v2, v3 = tri_extr(x_vertices, x_faces)
    x_centers, x_normals, x_areas = extract([v1, v2, v3])
    return calc_varifold_loss_torch(x_centers, x_normals, x_areas, y_centers, y_normals, y_weights, kernel_params)


def calc_normal_cycle_loss_torch(
    x_points: Float[Array, "N 3"],
    x_weights: Float[Array, "N F"],
    y_points: Float[Array, "M 3"],
    y_weights: Float[Array, "M F"],
    gamma: Float[Array, ""]
) -> Float[Array, "1"]:
    """
    Calculate the Normal Cycle loss between two weighted point sets.

    This implements a Maximum Mean Discrepancy (MMD) style loss in the Normal Cycle
    kernel space. It's commonly used for comparing geometric structures while
    accounting for both spatial and feature information.

    Args:
        x_points: Source points with shape (N, 3)
        x_weights: Source weights with shape (N, F)
        y_points: Target points with shape (M, 3)
        y_weights: Target weights with shape (M, F)
        gamma: Kernel bandwidth parameter

    Returns:
        Scalar loss value (reshaped to (1,) for compatibility)

    Mathematical Details:
        Loss =  ||μ_X - μ_Y||²_H = K(X,X) - 2K(X,Y) + K(Y,Y)
        where K is the normal cycle kernel, X=(x_points,x_weights), Y=(y_points,y_weights)
        This is the squared MMD distance in the kernel space.

    Note:
        Lower values indicate better similarity between the point sets.
        The loss is always non-negative due to the MMD formulation.
    """
    K_xx = normal_cycle_kernel(x_points, x_weights, x_points, x_weights, gamma)
    K_yy = normal_cycle_kernel(y_points, y_weights, y_points, y_weights, gamma)
    K_xy = normal_cycle_kernel(x_points, x_weights, y_points, y_weights, gamma)
    normal_cycle_loss = K_xx + K_yy - 2 * K_xy
    return normal_cycle_loss.reshape(-1)