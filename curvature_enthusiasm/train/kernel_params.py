"""Kernel parameter generation utilities."""
import jax
import jax.numpy as jnp
from typing import Tuple
from jaxtyping import Array, Float


def calc_kernel_params(
        sigma: float,
        sigma_sph: float
) -> Tuple[float, float]:
    """
    Calculate kernel parameters from sigma values.

    Args:
        sigma: Spatial bandwidth parameter
        sigma_sph: Spherical/orientation bandwidth parameter

    Returns:
        (gamma_spatial, gamma_normal) where gamma = 1/(2σ²)
    """
    gamma_spatial = 1.0 / (2.0 * sigma ** 2)
    gamma_normal = 0.0 if sigma_sph == 0.0 else 1.0 / (2.0 * sigma_sph ** 2)
    return gamma_spatial, gamma_normal


def generate_kernel_params(
        sigma_values: list[float],
        sigma_sph_values: list[float] | None = None,
        dtype: jnp.dtype = jnp.float32
) -> Float[jax.Array, "N 2"]:
    """
    Generate array of kernel parameters from lists of sigma values.

    Args:
        sigma_values: List of spatial sigma values
        sigma_sph_values: List of spherical sigma values. If None, uses 0 (normal cycles)
        dtype: Data type for output array

    Returns:
        Array of shape (N, 2) with [gamma_spatial, gamma_normal] pairs

    Examples:
        >>> # Varifold kernels (spatial + orientation)
        >>> varifold_params = generate_kernel_params([1.0, 2.0], [0.5, 1.0])
        >>> # Normal cycle kernels (spatial only)
        >>> nc_params = generate_kernel_params([1.0, 2.0])
    """
    # If no spherical sigmas provided, use zeros (normal cycles case)
    if sigma_sph_values is None:
        sigma_sph_values = [0.0] * len(sigma_values)

    params = [
        calc_kernel_params(sigma, sigma_sph)
        for sigma, sigma_sph in zip(sigma_values, sigma_sph_values)
    ]
    return jnp.array(params, dtype=dtype)