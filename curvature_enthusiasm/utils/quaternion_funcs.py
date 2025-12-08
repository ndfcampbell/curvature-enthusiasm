from functools import partial

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float


@partial(jax.jit, static_argnames=("hemisphere", "order"))
def pin_to_hemisphere(q, hemisphere: str = "positive", order: str = "wxyz", eps: float = 0.0):
    """
    Pin quaternion(s) to a consistent hemisphere.

    Args
    ----
    q : (..., 4) array
        Quaternion(s). Order controlled by `order`.
    hemisphere : {"positive","negative"}
        "positive" -> enforce w >= 0 (within eps)
        "negative" -> enforce w <= 0 (within eps)
    order : {"wxyz","xyzw"}
        Quaternion component order.
    eps : float
        Deadband around zero to avoid flipping on tiny numerical noise.

    Returns
    -------
    (..., 4) array with the same order as input.
    """
    q = jnp.asarray(q)
    want_positive = (hemisphere == "positive")
    w = q[..., 0] if order == "wxyz" else q[..., -1]

    if eps > 0.0:
        cond = (w < -eps) if want_positive else (w > eps)
    else:
        cond = (w < 0.0) if want_positive else (w > 0.0)

    # If cond, flip; otherwise keep. Equivalent to multiply by ±1.
    flip = jnp.where(cond, -1.0, 1.0)
    return q * flip[..., None]


def safe_normalize_quaternion(q_wxyz, eps=None):
    """Normalize quaternion(s); if ‖q‖ <= eps, return identity [1,0,0,0]."""
    q = jnp.asarray(q_wxyz)
    dtype = q.dtype
    eps = jnp.asarray(jnp.finfo(dtype).eps * 10 if eps is None else eps, dtype=dtype)

    # Decide near-zero using squared norm (no sqrt)
    n2 = jnp.sum(q * q, axis=-1, keepdims=True)
    is_near_zero = n2 <= eps * eps

    # Gradient-safe normalization
    denom = optax.safe_norm(q, axis=-1, keepdims=True, min_norm=eps)
    normalized = q / denom

    # Map near-zero to identity
    identity = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=dtype)
    identity = jnp.broadcast_to(identity, q.shape)

    return jnp.where(is_near_zero, identity, normalized)


def rotate_group(
    vecs: Float[Array, "n 3"],
    quat: Float[Array, "4"],
) -> Float[Array, "n 3"]:
    """Rotate a batch of 3D vectors by the same unit quaternion.

    Args:
       vecs: Array of shape (n, 3). Each row is a 3D vector to be rotated.
       quat: Array of shape (4,). A single quaternion (s, x, y, z) assumed to be unit length.

    Returns:
       Array of shape (n, 3), where each row is the corresponding rotated vector.

    Notes:
       - The quaternion `quat` should be normalized (‖quat‖ ≈ 1) for a proper rotation.
       - The rotation formula is derived from:
           v' = (s² - u·u) v + 2 (u·v) u + 2 s (u × v)
         where q = (s, u).
    """
    s = quat[0]
    u = quat[1:]
    dot = vecs @ u  # a bit cleaner than sum(..., axis=1)
    uu = u @ u
    ss = s * s
    return vecs * (ss - uu) + 2.0 * (dot[:, None] * u + s * jnp.cross(u, vecs))

  # # Broadcasting the unit vector and scalar part
  # # Ensuring proper handling of broadcasting and dot product
  # dot_product = jnp.sum(vecs * u, axis=1)  # Proper dot product for each vec with u
  # uv = dot_product[:, None] * u  # Shape (n, 3)
  # uu = jnp.dot(u, u)  # Scalar, since u is (3,)
  # cross_product = jnp.cross(u, vecs)
  # # Calculate the rotated vector
  # r = vecs * (s ** 2 - uu) + 2 * uv + 2 * s * cross_product
  # return r


def q_mult(q1: Float[Array, "4"], q2: Float[Array, "4"]) -> Float[Array, "4"]:
    """
    Quaternion multiplication.

    Computes the Hamilton product of two quaternions in (w, x, y, z) format.

    Args:
        q1: First quaternion with shape (4,) in (w, x, y, z) format
        q2: Second quaternion with shape (4,) in (w, x, y, z) format

    Returns:
        Product quaternion q1 * q2 with shape (4,) in (w, x, y, z) format

    Note:
        Quaternion multiplication is non-commutative: q1 * q2 ≠ q2 * q1
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return jnp.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])

def quat_mul_group(u: Float[Array, "N 4"], v: Float[Array, "N 4"]) -> Float[Array, "N 4"]:
    """Multiplies two quaternions.

    Args:
    u: (4,) quaternion (w,x,y,z)
    v: (n,4) quaternion (w,x,y,z)

    Returns:
    A quaternion u * v.
    """
    w = u[..., 0] * v[..., 0] - u[..., 1] * v[..., 1] - u[..., 2] * v[..., 2] - u[..., 3] * v[..., 3]
    x = u[..., 0] * v[..., 1] + u[..., 1] * v[..., 0] + u[..., 2] * v[..., 3] - u[..., 3] * v[..., 2]
    y = u[..., 0] * v[..., 2] - u[..., 1] * v[..., 3] + u[..., 2] * v[..., 0] + u[..., 3] * v[..., 1]
    z = u[..., 0] * v[..., 3] + u[..., 1] * v[..., 2] - u[..., 2] * v[..., 1] + u[..., 3] * v[..., 0]
    return jnp.stack([w, x, y, z], axis=-1)


def q_conj(q: Float[Array, "4"]) -> Float[Array, "4"]:
    """
    Quaternion conjugate.

    Computes the conjugate of a quaternion by negating the vector part
    while keeping the scalar part unchanged.

    Args:
        q: Quaternion with shape (4,) in (w, x, y, z) format

    Returns:
        Conjugate quaternion with shape (4,) in (w, -x, -y, -z) format

    Mathematical Details:
        For quaternion q = w + xi + yj + zk, the conjugate is:
        q* = w - xi - yj - zk = (w, -x, -y, -z)

    Note:
        For unit quaternions representing rotations, the conjugate
        represents the inverse rotation.
    """
    return jnp.array([q[0], -q[1], -q[2], -q[3]])



