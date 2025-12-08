from pykeops.torch import LazyTensor
from jaxtyping import Array, Float, Int

def varifold_kernel(
    x_points: Float[Array, "N 3"],
    x_normals: Float[Array, "N 3"],
    y_points: Float[Array, "M 3"],
    y_normals: Float[Array, "M 3"],
    kernel_params: Float[Array, "2"]):
    """
    Compute the varifold kernel between two geometric shapes.

    The varifold kernel combines spatial proximity with orientation similarity:
    K(x_i, y_j) = exp(-γ_s ||x_i - y_j||²) * exp(γ_n ⟨u_i, v_j⟩)

    Args:
        x_points: Centers/points of first shape with shape (N, 3)
        x_normals: Unit normal vectors of first shape with shape (N, 3)
        y_points: Centers/points of second shape with shape (M, 3)
        y_normals: Unit normal vectors of second shape with shape (M, 3)
        kernel_params: Kernel parameters [sigma_space, sigma_sphere] with shape (2,):
                      - sigma_space: Spatial scale parameter (controls spatial locality)
                      - sigma_sphere: Spherical scale parameter (controls orientation similarity)

    Returns:
        LazyTensor representing the varifold kernel matrix with shape (N, M).
        Each entry K[i,j] measures similarity between elements i and j considering
        both spatial distance and normal alignment.

    Mathematical Details:
        K(x_i, y_j) = exp(-||x_i - y_j||² / σ_space) * exp(σ_sphere * ⟨u_i, v_j⟩)
        where:
        - x_i, y_j are spatial positions
        - u_i, v_j are unit normal vectors
        - σ_space controls spatial bandwidth
        - σ_sphere controls orientation sensitivity

    Note:
        Uses KeOps LazyTensor for memory-efficient computation without materializing
        the full N×M kernel matrix in memory.
    """

    # sigma, sig = var_kernel_params
    gamma_spatial = kernel_params[0]
    gamma_normal = kernel_params[1]

    x_points_lazy = LazyTensor(x_points[:, None, :])
    y_points_lazy = LazyTensor(y_points[None, :, :])
    x_normals_lazy = LazyTensor(x_normals[:, None, :])
    y_normals_lazy = LazyTensor(y_normals[None, :, :])

    spatial_distances_sq = x_points_lazy.sqdist(y_points_lazy)
    normal_similarity = (x_normals_lazy * y_normals_lazy).sum()

    return (-spatial_distances_sq * gamma_spatial).exp()  * (normal_similarity * gamma_normal).exp()

    # normal_kernel = (normal_similarity * gamma_normal).exp()
    # spatial_kernel = (-spatial_distances_sq * gamma_spatial).exp()
    # return normal_kernel * spatial_kernel

def normal_cycle_kernel(
    x_points: Float[Array, "N 3"],
    x_weights: Float[Array, "N F"],
    y_points: Float[Array, "M 3"],
    y_weights: Float[Array, "M F"],
    gamma: Float[Array, ""]
) -> Float[Array, ""]:
    """
    Compute the normal cycle kernel between two point sets with associated weights.

    This kernel is used in geometric measure theory and shape analysis, particularly
    for comparing geometric structures like meshes or point clouds. It combines
    spatial proximity (via RBF kernel) with feature similarity (via weight dot products).

    Args:
        x_points: First point set with shape (N, D) where N is number of points and D is spatial dimension
        x_weights: Weights/features for first point set with shape (N, F) where F is feature dimension
        y_points: Second point set with shape (M, D)
        y_weights: Weights/features for second point set with shape (M, F)
        gamma: Kernel bandwidth parameter (scalar). Higher values create sharper kernels.

    Returns:
        Scalar kernel value representing similarity between the two weighted point sets.

    Mathematical Details:
        The kernel computes: Σᵢⱼ exp(-γ||x_pointsᵢ - y_pointsⱼ||²) * ⟨x_weightsᵢ, y_weightsⱼ⟩
        where the sum is over all pairs (i,j) of points from the two sets.

    Note:
        Uses KeOps LazyTensor for memory-efficient computation of large kernel matrices
        without materializing the full N×M distance matrix in memory.
    """

    # Convert to LazyTensors for efficient computation
    x_points = LazyTensor(x_points[:, None, :])
    y_points = LazyTensor(y_points[None, :, :])
    D = x_points.sqdist(y_points)
    K = (-D * gamma).exp()
    x_weights = LazyTensor(x_weights[:, None, :])
    y_weights = LazyTensor(y_weights[None, :, :])
    weight_similarity = ((x_weights * y_weights)).sum()
    res = K * weight_similarity
    return res.sum(0).sum()



