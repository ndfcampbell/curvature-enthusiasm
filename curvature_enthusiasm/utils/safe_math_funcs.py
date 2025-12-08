from typing import Tuple

import jax.numpy as jnp
import optax
from jaxtyping import Array, Float


def safe_normalize(
    x: Float[Array, "..."],
    axis: int | tuple[int, ...] = -1,
    keepdims: bool = True,
    ord: int | None = None,
    eps: float | Float[Array, ""] | None = None,
) -> Float[Array, "..."]:

    """Normalize a vector with safe behavior near zero norm."""
    x = jnp.asarray(x)
    if eps is None:
        eps = jnp.finfo(x.dtype).eps * 10

    # Compute norm once (squared for efficiency if using L2)
    if ord is None or ord == 2:
        norm_sq = jnp.sum(x * x, axis=axis, keepdims=True)
        is_small = norm_sq <= eps * eps
        norm = jnp.sqrt(jnp.maximum(norm_sq, eps * eps))
    else:
        norm = jnp.linalg.norm(x, axis=axis, keepdims=True, ord=ord)
        is_small = norm <= eps
        norm = jnp.maximum(norm, eps)

    normalized = x / norm
    result = jnp.where(is_small, jnp.zeros_like(x), normalized)

    if not keepdims:
        result = jnp.squeeze(result, axis=axis)

    return result

def safe_norm(
    x: Float[Array, "..."],
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    eps: float | Float[Array, ""] | None = None,
) -> Float[Array, "..."]:

    """Gradient-safe norm: returns ||x|| for ||x|| > eps, else 0."""
    x = jnp.asarray(x)
    # dtype-aware epsilon (slightly larger than machine eps for stability)
    eps = jnp.asarray(jnp.finfo(x.dtype).eps * 10 if eps is None else eps, dtype=x.dtype)
    eps2 = eps * eps

    # Decision uses raw squared norm (no extra sqrt)
    n2 = jnp.sum(jnp.square(x), axis=axis, keepdims=keepdims)
    big = n2 > eps2

    # Good-gradient norm (clamped only to stabilize division/gradients)
    n_clamped = optax.safe_norm(x, axis=axis, keepdims=keepdims, min_norm=eps)

    return jnp.where(big, n_clamped, jnp.zeros_like(n_clamped))

def safe_normalize_with_norm(
    x: Float[Array, "..."],
    axis: int | tuple[int, ...] = -1,
    keepdims: bool = True,
    eps: float | Float[Array, ""] | None = None,
) -> Tuple[Float[Array, "..."], Float[Array, "..."]]:

    """Return (x/||x||, ||x||) with stable grads; tiny vectors -> (0, 0)."""
    x = jnp.asarray(x)
    eps = jnp.asarray(jnp.finfo(x.dtype).eps * 10 if eps is None else eps, dtype=x.dtype)
    eps2 = eps * eps

    # Squared norm for decision (no extra sqrt)
    n2 = jnp.sum(jnp.square(x), axis=axis, keepdims=True)
    is_big = n2 > eps2

    # Denominator with good gradients
    denom = optax.safe_norm(x, axis=axis, keepdims=True, min_norm=eps)
    y_unit = x / denom

    # Use raw norm (exact) for the returned magnitude
    n_raw = jnp.sqrt(n2)

    normalized_vectors = jnp.where(is_big, y_unit, jnp.zeros_like(x))
    vector_norms = jnp.where(is_big, n_raw, jnp.zeros_like(n_raw))

    if not keepdims:
        vector_norms = jnp.squeeze(vector_norms, axis=axis)
    return normalized_vectors, vector_norms

