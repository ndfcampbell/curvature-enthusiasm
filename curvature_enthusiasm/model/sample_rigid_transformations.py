import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..utils.quaternion_funcs import q_mult, q_conj, rotate_group, quat_mul_group


def _finfo_tol(dtype):
    # A dtype-aware tiny tolerance you can use elsewhere if you like
    return jnp.asarray(1e-4 if dtype == jnp.float32 else 1e-8, dtype=dtype)


def q_ln(
    q_wxyz: Float[Array, "... 4"],
    tol: Float[Array, ""] | float | None = None
) -> Float[Array, "... 4"]:
    """
    Quaternion log for q = (w,x,y,z), principal branch:
      log(q) = (0.5*log(||q||^2), theta * v_hat),
    with theta = atan2(||v||, w), v_hat = v/||v|| (or 0 if ||v||~0).
    """
    q = jnp.asarray(q_wxyz)
    dtype = q.dtype
    tol = jnp.asarray(1e-6 if dtype == jnp.float32 else 1e-12, dtype=dtype) if tol is None else jnp.asarray(tol, dtype=dtype)
    tol2 = tol * tol

    w = q[..., 0:1]
    v = q[..., 1:4]

    r2 = jnp.sum(v * v, axis=-1, keepdims=True)   # ||v||^2
    r  = jnp.sqrt(r2)                              # ||v||
    theta = jnp.arctan2(r, w)                      # angle to real axis

    # v_hat = v / r (safe), then scale by theta
    inv_r = jnp.where(r2 > tol2, jax.lax.rsqrt(r2), 0.0)
    v_scaled = v * (inv_r * theta)                 # combines both

    # scalar part: 0.5 * log(||q||^2) with guard
    q2 = w * w + r2
    s = 0.5 * jnp.log(jnp.maximum(q2, jnp.finfo(dtype).tiny))

    return jnp.concatenate([s, v_scaled], axis=-1)


def q_exp(
    q_wxyz: Float[Array, "... 4"],
    tol: Float[Array, ""] | float | None = None
) -> Float[Array, "... 4"]:
    """
    Quaternion exponential for q = (w, x, y, z).
    exp(q) = exp(w) * [cos(||v||), (sin(||v||)/||v||) * v],  v=(x,y,z).
    Stable near ||v|| -> 0.

    Args:
        q_wxyz: Quaternion(s) in [w,x,y,z] format, shape (..., 4)
        tol: Threshold for small-angle approximation. If None, uses
             1e-4 for float32, 1e-8 for float64

    Returns:
        Quaternion exponential, same shape as input
    """
    q = jnp.asarray(q_wxyz)
    dtype = q.dtype

    # dtype-aware small-angle switch
    if tol is None:
        tol = jnp.asarray(1e-4 if dtype == jnp.float32 else 1e-8, dtype=dtype)

    tol2 = tol * tol

    # Split quaternion into scalar and vector parts
    w = q[..., 0:1]  # (...,1)
    v = q[..., 1:4]  # (...,3)

    # Compute norm of vector part
    r2 = jnp.sum(v * v, axis=-1, keepdims=True)  # (...,1)
    r = jnp.sqrt(r2)  # (...,1)

    # Determine which computation path to use
    small = r2 <= tol2

    # Normal branch (guard denominator to avoid NaNs even if both branches eval)
    sinc_normal = jnp.sin(r) / jnp.maximum(r, jnp.finfo(dtype).tiny)
    cos_normal = jnp.cos(r)

    # Taylor series around 0 (use r2/r4, no extra sqrt)
    r4 = r2 * r2
    # sin(x)/x ≈ 1 - x²/6 + x⁴/120 - x⁶/5040
    sinc_taylor = 1.0 - r2 / 6.0 + r4 / 120.0 - (r2 * r4) / 5040.0
    # cos(x) ≈ 1 - x²/2 + x⁴/24 - x⁶/720
    cos_taylor = 1.0 - r2 / 2.0 + r4 / 24.0 - (r2 * r4) / 720.0

    # Select appropriate computation based on magnitude
    sinc = jnp.where(small, sinc_taylor, sinc_normal)
    c = jnp.where(small, cos_taylor, cos_normal)

    # Apply exponential scaling
    ew = jnp.exp(w)
    w_out = ew * c
    v_out = ew * (v * sinc)

    # Combine into output quaternion
    return jnp.concatenate([w_out, v_out], axis=-1)

def q_normalize(q: Float[Array, "... 4"]) -> Float[Array, "... 4"]:
    norm = jnp.linalg.norm(q, axis=-1, keepdims=True)
    # guard zero; keep direction if not zero
    return q / jnp.maximum(norm, jnp.finfo(q.dtype).tiny)


def q_shortest(q: Float[Array, "... 4"]) -> Float[Array, "... 4"]:
    # flip to ensure w >= 0 (shortest arc)
    sign = jnp.where(q[..., :1] < 0, -1.0, 1.0)
    return q * sign

def q_unitize_shortest(q: Float[Array, "... 4"]) -> Float[Array, "... 4"]:
    return q_shortest(q_normalize(q))


def q_slerp(
    q: Float[Array, "... 4"],
    t: Float[Array, "..."] | float
) -> Float[Array, "... 4"]:
    """Slerp from identity to q (rotation only)."""
    u = q_unitize_shortest(q)
    logu = q_ln(u)
    # Zero the scalar part to avoid tiny drift from numeric ln:
    logu = jnp.concatenate([jnp.zeros_like(logu[..., 0:1]), logu[..., 1:4]], axis=-1)
    out = q_exp(t * logu)
    # Renormalize to be safe
    return q_normalize(out)

# @jax.jit
def q_derivative(
    qt: Float[Array, "... 4"],
    q: Float[Array, "... 4"]
) -> Float[Array, "... 4"]:
    """Right-trivialized: d/dt q(t) = q(t) * log(u), with u unit."""
    u = q_unitize_shortest(q)
    logu = q_ln(u)
    logu = jnp.concatenate([jnp.zeros_like(logu[..., 0:1]), logu[..., 1:4]], axis=-1)
    return q_mult(qt, logu)

def field_points(
    x: Float[Array, "N 3"],
    qt: Float[Array, "... 4"],
    q: Float[Array, "... 4"],
    t: Float[Array, "..."] | float
) -> Float[Array, "N 3"]:

    zeros = jnp.zeros((x.shape[0], 1), dtype=x.dtype)
    p0 = jnp.concatenate([zeros, x], axis=-1)  # (N,4) pure quats

    qt_dot = q_derivative(qt, q)
    q_inverse = q_conj(q)
    qt_inverse = q_conj(qt)
    qt_conj_dot = q_derivative(qt_inverse, q_inverse)

    q0_deriv = quat_mul_group(quat_mul_group(qt_dot, p0), qt_inverse) \
               + quat_mul_group(quat_mul_group(qt, p0), qt_conj_dot)
    return q0_deriv[:, 1:]  # (N,3)

def quat_transformation_field(
    x: Float[Array, "N 3"],
    qt: Float[Array, "... 4"],
    q: Float[Array, "... 4"],
    translation: Float[Array, "... 3"],
    time: Float[Array, "..."] | float
) -> Float[Array, "N 3"]:
    rotation_field_points = field_points(x, qt, q, time)
    return translation + rotation_field_points

def transform_points(
    p0: Float[Array, "3"],
    pts: Float[Array, "N 3"],
    translation: Float[Array, "3"],
    qt: Float[Array, "4"]
) -> Float[Array, "N 3"]:
    local_pts = pts - p0
    local_ips = rotate_group(local_pts, qt)
    global_pts = (p0 + local_ips) + translation  # Changed order of operations
    return global_pts


def transform_points_inter(
    p0: Float[Array, "3"],
    pts: Float[Array, "N 3"],
    translation: Float[Array, "3"],
    q: Float[Array, "4"],
    time: float
) -> Float[Array, "N 3"]:
    local_pts = pts - p0
    t = time * translation
    qt = q_slerp(q, time)
    local_ips = rotate_group(local_pts, qt)
    global_ips = (p0 + local_ips) + t
    return global_ips

