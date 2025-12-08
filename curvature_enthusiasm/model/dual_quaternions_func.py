from typing import Tuple

import jax
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, Float

from ..utils.safe_math_funcs import safe_normalize, safe_norm
from ..utils.quaternion_funcs import q_mult, q_conj, safe_normalize_quaternion

def check_dual_quaternion(
    dq: Float[Array, "8"],
    eps: float = 1e-6
) -> Float[Array, "8"]:
    """Input validation of dual quaternion representation using safe norm."""
    # dq = jnp.asarray(dq)
    # Use the norm of the real part safely.
    real_norm = safe_norm(dq[:4]) #jp_safe_norm(dq[:4]) + eps
    dq = dq / real_norm
    return dq


def check_screw_parameters(
        q: Float[Array, "3"],
        s_axis: Float[Array, "3"],
        h: Float[Array, ""]
) -> Tuple[Float[Array, "3"], Float[Array, "3"], Float[Array, ""]]:
    """
    Validate and normalize screw motion parameters.

    Processes screw motion parameters to handle special cases and ensure
    proper normalization. Screw motions combine rotation and translation
    along/around a common axis, commonly used in robotics and mechanism analysis.

    Args:
        q: Point on the screw axis with shape (3,) as (x, y, z).
           For pure translations (infinite pitch), this will be set to zero.
        s_axis: Screw axis direction vector with shape (3,) as (x, y, z).
               Will be normalized to unit length.
        h: Screw pitch parameter with shape ().
           - Finite values represent helical motion (rotation + translation)
           - Infinite values represent pure translation

    Returns:
        Tuple of validated parameters:
        - q: Validated point on screw axis with shape (3,)
        - s_axis: Normalized screw axis with shape (3,)
        - h: Original pitch parameter with shape ()

    Mathematical Details:
        Screw motion is characterized by:
        - s_axis: Unit vector defining the screw axis direction
        - q: Any point on the screw axis (defines axis position)
        - h: Pitch = translation_distance / rotation_angle

        Special cases:
        - h = ∞: Pure translation, q becomes irrelevant (set to zero)
        - h = 0: Pure rotation around axis through point q

    Process:
        1. Check if pitch h is infinite (pure translation case)
        2. If infinite, set q to zero vector (axis position irrelevant)
        3. If finite, keep q unchanged (axis position matters)
        4. Normalize s_axis to ensure unit length
        5. Return processed parameters

    Note:
        - Uses jax.lax.cond for conditional execution in JAX
        - safe_normalize handles potential zero vectors in s_axis
        - Pure translation case (h=∞) makes axis position irrelevant
        - Result can be used directly in screw motion computations

    Example:
        >>> q = jnp.array([1.0, 2.0, 3.0])
        >>> s_axis = jnp.array([1.0, 1.0, 0.0])  # Non-unit vector
        >>> h = jnp.array(jnp.inf)               # Pure translation
        >>> q_out, s_axis_out, h_out = check_screw_parameters(q, s_axis, h)
        >>> # Result: q_out = [0,0,0], s_axis_out ≈ [0.707,0.707,0], h_out = inf
    """

    # If h is infinite (pure translation), set q to the zero vector.
    q = jax.lax.cond(
        jnp.isinf(h),
        lambda _: jnp.zeros_like(q),
        lambda _: q,
        operand=None
    )

    # Normalize the screw axis.
    s_axis = safe_normalize(s_axis)

    return q, s_axis, h


def concatenate_quaternions(
    q1: Float[Array, "4"],
    q2: Float[Array, "4"]
) -> Float[Array, "4"]:
    r"""Concatenate two quaternions.

    We concatenate two quaternions by quaternion multiplication
    :math:`\boldsymbol{q}_1\boldsymbol{q}_2`.

    We use Hamilton's quaternion multiplication.

    If the two quaternions are divided up into scalar part and vector part
    each, i.e.,
    :math:`\boldsymbol{q} = (w, \boldsymbol{v}), w \in \mathbb{R},
    \boldsymbol{v} \in \mathbb{R}^3`, then the quaternion product is

    .. math::

        \boldsymbol{q}_{12} =
        (w_1 w_2 - \boldsymbol{v}_1 \cdot \boldsymbol{v}_2,
        w_1 \boldsymbol{v}_2 + w_2 \boldsymbol{v}_1
        + \boldsymbol{v}_1 \times \boldsymbol{v}_2)

    with the scalar product :math:`\cdot` and the cross product :math:`\times`.

    Args:
        q1: First quaternion with shape (4,) in (w, x, y, z) format.
            Represents the first rotation to be applied.
        q2: Second quaternion with shape (4,) in (w, x, y, z) format.
            Represents the second rotation to be applied.

    Returns:
        Product quaternion q12 with shape (4,) representing the concatenated
        rotation q1 * q2 in (w, x, y, z) format.

    Mathematical Details:
        For quaternions q1 = (w1, v1) and q2 = (w2, v2) where w is the scalar
        part and v is the 3D vector part, Hamilton's multiplication gives:

        .. math::

            \boldsymbol{q}_{12} = \boldsymbol{q}_1 \boldsymbol{q}_2 =
            (w_1 w_2 - \boldsymbol{v}_1 \cdot \boldsymbol{v}_2,
            w_1 \boldsymbol{v}_2 + w_2 \boldsymbol{v}_1
            + \boldsymbol{v}_1 \times \boldsymbol{v}_2)

        where:
        - :math:`\cdot` denotes the scalar (dot) product
        - :math:`\times` denotes the cross product

    Implementation:
        - Scalar part: w1*w2 - dot(v1, v2)
        - Vector part: w1*v2 + w2*v1 + cross(v1, v2)
        - Uses jnp.r_ to concatenate scalar and vector parts

    Note:
        - Quaternion multiplication is non-commutative: q1*q2 ≠ q2*q1
        - The order matters for rotation composition
        - Result represents applying q1 rotation first, then q2
        - No automatic normalization is performed on inputs or output

    Example:
        >>> q1 = jnp.array([1.0, 0.0, 0.0, 0.0])  # Identity rotation
        >>> q2 = jnp.array([0.707, 0.707, 0.0, 0.0])  # 90° around x-axis
        >>> q12 = concatenate_quaternions(q1, q2)  # Result: q2 (since q1 is identity)

    """

    # q1 = check_quaternion(q1, unit=False)
    # q2 = check_quaternion(q2, unit=False)
    # q12 = jnp.empty(4)
    q12_0 = q1[0] * q2[0] - jnp.dot(q1[1:], q2[1:])
    q12_1 = q1[0] * q2[1:] + q2[0] * q1[1:] + jnp.cross(q1[1:], q2[1:])
    return jnp.r_[q12_0, q12_1]


def dq_q_conj(
    dq: Float[Array, "8"],
) -> Float[Array, "8"]:
    """Quaternion conjugate of dual quaternion.

    For unit dual quaternions that represent transformations, this function
    is equivalent to the inverse of the corresponding transformation matrix.

    There are three different conjugates for dual quaternions. The one that we
    use here converts (pw, px, py, pz, qw, qx, qy, qz) to
    (pw, -px, -py, -pz, qw, -qx, -qy, -qz). It is the quaternion conjugate
    applied to each of the two quaternions.

    Parameters
    ----------
    dq : array-like, shape (8,)
        Unit dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)

    Returns
    -------
    dq_q_conjugate : array, shape (8,)
        Conjugate of dual quaternion: (pw, -px, -py, -pz, qw, -qx, -qy, -qz)

    See Also
    --------
    dq_conj
        Conjugate of a dual quaternion.
    """
    dq = check_dual_quaternion(dq)
    return jnp.r_[dq[0], -dq[1:4], dq[4], -dq[5:]]


def concatenate_dual_quaternions(
    dq1: Float[Array, "8"],
    dq2: Float[Array, "8"],
    unit: bool = True,
) -> Float[Array, "8"]:
    r"""Concatenate dual quaternions.

    We concatenate two dual quaternions by dual quaternion multiplication

    .. math::

        (\boldsymbol{p}_1 + \epsilon \boldsymbol{q}_1)
        (\boldsymbol{p}_2 + \epsilon \boldsymbol{q}_2)
        = \boldsymbol{p}_1 \boldsymbol{p}_2 + \epsilon (
        \boldsymbol{p}_1 \boldsymbol{q}_2 + \boldsymbol{q}_1 \boldsymbol{p}_2)

    using Hamilton multiplication of quaternions.

    .. warning::

        Note that the order of arguments is different than the order in
        :func:`concat`.

    Parameters
    ----------
    dq1 : array-like, shape (8,)
        Dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)

    dq2 : array-like, shape (8,)
        Dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)

    unit : bool, optional (default: True)
        Normalize the dual quaternion so that it is a unit dual quaternion.
        A unit dual quaternion has the properties
        :math:`p_w^2 + p_x^2 + p_y^2 + p_z^2 = 1` and
        :math:`p_w q_w + p_x q_x + p_y q_y + p_z q_z = 0`.

    Returns
    -------
    dq3 : array, shape (8,)
        Product of the two dual quaternions:
        (pw, px, py, pz, qw, qx, qy, qz)

    See Also
    --------
    pytransform3d.rotations.concatenate_quaternions
        Quaternion multiplication.
    """
    # dq1 = check_dual_quaternion(dq1, unit=unit)
    # dq2 = check_dual_quaternion(dq2, unit=unit)
    dq1 = check_dual_quaternion(dq1)
    dq2 = check_dual_quaternion(dq2)
    real = concatenate_quaternions(dq1[:4], dq2[:4])
    dual = (concatenate_quaternions(dq1[:4], dq2[4:]) +
            concatenate_quaternions(dq1[4:], dq2[:4]))
    return jnp.hstack((real, dual))


def axis_angle_from_quaternion(
    q: Float[Array, "4"],
    eps: float = 1e-8,
) -> Float[Array, "4"]:
    """Compute axis-angle from quaternion.

    This operation is called logarithmic map.

    We usually assume active rotations.

    Parameters
    ----------
    q : array-like, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)

    Returns
    -------
    a : array, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle). The angle is
        constrained to [0, pi) so that the mapping is unique.
    """
    # eps = 1e-8  # Small epsilon to prevent division issues
    # q = check_unit_quaternion(q)
    q = safe_normalize_quaternion(q)
    p = q[1:]
    # p_norm = jp_safe_norm(p) + eps  # Ensures p_norm is never zero
    #
    # # Normalize axis safely
    # axis = p / p_norm

    axis = safe_normalize(p)

    # Clamp w to avoid numerical issues in arccos
    w_clamped = jnp.clip(q[0], -1.0 + eps, 1.0 - eps)
    angle = 2.0 * jnp.arccos(w_clamped)

    out = jnp.concatenate((axis, jnp.atleast_1d(angle)))

    return out


def dq_conj(
    dq: Float[Array, "8"],
) -> Float[Array, "8"]:
    """Conjugate of dual quaternion.

    There are three different conjugates for dual quaternions. The one that we
    use here converts (pw, px, py, pz, qw, qx, qy, qz) to
    (pw, -px, -py, -pz, -qw, qx, qy, qz). It is a combination of the quaternion
    conjugate and the dual number conjugate.

    Parameters
    ----------
    dq : array-like, shape (8,)
        Unit dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)

    Returns
    -------
    dq_conjugate : array, shape (8,)
        Conjugate of dual quaternion: (pw, -px, -py, -pz, -qw, qx, qy, qz)

    See Also
    --------
    dq_q_conj
        Quaternion conjugate of dual quaternion.
    """
    dq = check_dual_quaternion(dq)
    return jnp.r_[dq[0], -dq[1:5], dq[5:]]


def screw_parameters_from_dual_quaternion(
    dq: Float[Array, "8"],
) -> tuple[
    Float[Array, "3"],  # q
    Float[Array, "3"],  # s_axis
    float,              # h
    float               # theta
]:
    """Compute screw parameters from dual quaternion using JAX.

    Parameters
    ----------
    dq : array-like, shape (8,)
        Unit dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)

    Returns
    -------
    q : array, shape (3,)
        Vector to a point on the screw axis

    s_axis : array, shape (3,)
        Direction vector of the screw axis

    h : float
        Pitch of the screw. Infinite pitch indicates pure translation.

    theta : float
        Parameter of the transformation: theta is the angle of rotation and
        h * theta the translation.
    """
    dq = check_dual_quaternion(dq)
    real = dq[:4]
    dual = dq[4:]

    # Extract axis and angle from the real quaternion.
    a = axis_angle_from_quaternion(real)
    s_axis = a[:3]
    theta = a[3]

    translation = 2 * concatenate_quaternions(dual, q_conj(real))[1:]
    eps = jnp.finfo(float).eps
    # When rotation angle is nearly zero, treat as pure translation.
    def pure_translation_branch(_):
        d = safe_norm(translation)
        s_axis_new = jnp.where(d < eps, jnp.array([1.0, 0.0, 0.0]), translation / d)
        q_out = jnp.zeros(3)
        theta_out = d
        h_out = jnp.inf
        return q_out, s_axis_new, h_out, theta_out

    def rotation_branch(_):
        distance = jnp.dot(translation, s_axis)
        moment = 0.5 * (jnp.cross(translation, s_axis) +
                        (translation - distance * s_axis) / jnp.tan(0.5 * theta))
        # The dual part as computed here is not used later except in deriving h.
        h_out = distance / theta
        return moment, s_axis, h_out, theta

    q, s_axis_out, h_out, theta_out = jax.lax.cond(jnp.abs(theta) < eps,
                                               pure_translation_branch,
                                               rotation_branch,
                                               operand=None)
    return q, s_axis_out, h_out, theta_out


def dual_quaternion_from_screw_parameters(
    q: Float[Array, "3"],
    s_axis: Float[Array, "3"],
    h: Float[Array, ""],
    theta: Float[Array, ""],
) -> Float[Array, "8"]:
    """Compute dual quaternion from screw parameters using JAX.

    Parameters
    ----------
    q : array-like, shape (3,)
        Vector to a point on the screw axis

    s_axis : array-like, shape (3,)
        Direction vector of the screw axis

    h : float or scalar array
        Pitch of the screw. Infinite pitch indicates pure translation.

    theta : float or scalar array
        Parameter of the transformation: theta is the angle of rotation and
        h * theta the translation.

    Returns
    -------
    dq : array, shape (8,)
        Unit dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)
    """
    q, s_axis, h = check_screw_parameters(q, s_axis, h)

    # Use lax.cond to safely branch on whether h is infinite.
    def pure_translation(_):
        # Pure translation: d = theta and set theta = 0.
        return theta, 0.5 * theta  # returning (theta_out, half_distance)

    def rotation_branch(_):
        return theta, 0.5 * (h * theta)

    theta_out, half_distance = jax.lax.cond(jnp.isinf(h),
                                        pure_translation,
                                        rotation_branch,
                                        operand=None)
    # When h is infinite (pure translation), we want theta to be zero.
    d = jax.lax.cond(jnp.isinf(h), lambda _: theta, lambda _: h * theta, operand=None)

    # For pure translation, we already set theta_out = theta but then override to 0.
    theta_used = jax.lax.cond(jnp.isinf(h), lambda _: 0.0, lambda _: theta, operand=None)

    moment = jnp.cross(q, s_axis)

    sin_half_angle = jnp.sin(0.5 * theta_used)
    cos_half_angle = jnp.cos(0.5 * theta_used)

    real_w = cos_half_angle
    real_vec = sin_half_angle * s_axis
    dual_w = -half_distance * sin_half_angle
    dual_vec = sin_half_angle * moment + half_distance * cos_half_angle * s_axis

    # Concatenate the scalar and vector parts appropriately.
    dq = jnp.concatenate([jnp.array([real_w]), real_vec,
                          jnp.array([dual_w]), dual_vec])
    return dq



def dual_quaternion_power(
    dq: Float[Array, "8"],
    t: Float[Array, ""],
    small_angle_thresh: float = 1e-4,
) -> Float[Array, "8"]:
    """
    Compute the power of a unit dual quaternion with a continuous treatment
    for small rotation angles.
    """
    # dq = check_dual_quaternion(dq)
    # dual_val, s_axis, h, theta = screw_parameters_from_dual_quaternion(dq)
    #
    # jax.debug.print("theta: {x}", x=theta)
    # jax.debug.print("dual_val: {x}", x=dual_val)
    #
    # # Smoothly handle the case when theta is very small (near-identity)
    # # We blend between a rotation and pure translation scenario.
    # weight = jnp.clip(theta / small_angle_thresh, 0.0, 1.0)
    # effective_theta = weight * theta + (1 - weight) * (jp_safe_norm(dual_val) * 2.0)
    # jax.debug.print("effective_theta: {x}", x=effective_theta)
    #
    # # Scale theta by t
    # new_theta = effective_theta * t
    #
    # # Reconstruct the dual quaternion using the scaled rotation.
    # dq_t = dual_quaternion_from_screw_parameters(dual_val, s_axis, h, new_theta)
    # return dq_t

    dq = check_dual_quaternion(dq)
    q, s_axis, h, theta = screw_parameters_from_dual_quaternion(dq)
    return dual_quaternion_from_screw_parameters(q, s_axis, h, theta * t)


def dual_quaternion_sclerp(
    start: Float[Array, "8"],
    end: Float[Array, "8"],
    t: Float[Array, ""],
) -> Float[Array, "8"]:
    """Screw linear interpolation (ScLERP) for dual quaternions.

    Although linear interpolation of dual quaternions is possible, this does
    not result in constant velocities. If you want to generate interpolations
    with constant velocity, you have to use ScLERP.

    Parameters
    ----------
    start : array-like, shape (8,)
        Unit dual quaternion to represent start pose:
        (pw, px, py, pz, qw, qx, qy, qz)

    end : array-like, shape (8,)
        Unit dual quaternion to represent end pose:
        (pw, px, py, pz, qw, qx, qy, qz)

    t : float in [0, 1]
        Position between start and goal

    Returns
    -------
    dq_t : array, shape (8,)
        Interpolated unit dual quaternion: (pw, px, py, pz, qw, qx, qy, qz)

    References
    ----------
    .. [1] Kavan, L., Collins, S., O'Sullivan, C., Zara, J. (2006).
       Dual Quaternions for Rigid Transformation Blending, Technical report,
       Trinity College Dublin,
       https://users.cs.utah.edu/~ladislav/kavan06dual/kavan06dual.pdf

    See Also
    --------
    transform_sclerp :
        ScLERP for transformation matrices.

    pq_slerp :
        An alternative approach is spherical linear interpolation (SLERP) with
        position and quaternion.
    """
    start = check_dual_quaternion(start)
    end = check_dual_quaternion(end)
    diff = concatenate_dual_quaternions(dq_q_conj(start), end)
    return concatenate_dual_quaternions(start, dual_quaternion_power(diff, t))

# ------------------------------------------------------------------------------
# MOVED FROM ik_system_dq.py
# ------------------------------------------------------------------------------


def dq_mult(dq1: Float[Array, "2 4"], dq2: Float[Array, "2 4"]) -> Float[Array, "2 4"]:
    """
    Dual quaternion multiplication.

    Computes the product of two dual quaternions using the dual number
    multiplication rule: (a + εb)(c + εd) = ac + ε(ad + bc).

    Args:
        dq1: First dual quaternion with shape (2, 4) where:
             - dq1[0] is the real quaternion part (w, x, y, z)
             - dq1[1] is the dual quaternion part (w', x', y', z')
        dq2: Second dual quaternion with shape (2, 4) in the same format

    Returns:
        Product dual quaternion dq1 * dq2 with shape (2, 4), normalized

    Mathematical Details:
        For dual quaternions dq1 = q_r + εq_d and dq2 = p_r + εp_d:
        dq1 * dq2 = (q_r * p_r) + ε(q_r * p_d + q_d * p_r)

        where * denotes quaternion multiplication and ε is the dual unit
        with the property ε² = 0.

    Note:
        The result is normalized to ensure it represents a valid
        rigid transformation (rotation + translation).
    """
    real = q_mult(dq1[0], dq2[0])
    dual = q_mult(dq1[0], dq2[1]) + q_mult(dq1[1], dq2[0])
    return dq_normalize(jnp.stack([real, dual]))


def dq_conjugate(dq: Float[Array, "2 4"]) -> Float[Array, "2 4"]:
    """
    Dual quaternion conjugate.

    Computes the conjugate of a dual quaternion by taking the quaternion
    conjugate of both the real and dual parts.

    Args:
        dq: Dual quaternion with shape (2, 4) where:
            - dq[0] is the real quaternion part (w, x, y, z)
            - dq[1] is the dual quaternion part (w', x', y', z')

    Returns:
        Conjugate dual quaternion with shape (2, 4) where:
        - result[0] = conjugate of real part = (w, -x, -y, -z)
        - result[1] = conjugate of dual part = (w', -x', -y', -z')

    Mathematical Details:
        For dual quaternion dq = q_r + εq_d, the conjugate is:
        dq* = q_r* + εq_d*

        where q* denotes quaternion conjugate and ε is the dual unit.

    Note:
        For unit dual quaternions representing rigid transformations,
        the conjugate represents the inverse transformation.
    """
    return jnp.stack([q_conj(dq[0]), q_conj(dq[1])])


def dq_transform_point(dq: Float[Array, "2 4"], point: Float[Array, "3"]) -> Float[Array, "3"]:
    """
    Apply a dual quaternion transformation to a point.

    Transforms a 3D point using a dual quaternion representing a rigid
    transformation (rotation + translation).

    Args:
        dq: Dual quaternion with shape (2, 4) where:
            - dq[0] is the real part representing rotation (w, x, y, z)
            - dq[1] is the dual part encoding translation (w', x', y', z')
        point: 3D point to transform with shape (3,) as (x, y, z)

    Returns:
        Transformed 3D point with shape (3,) as (x', y', z')

    Mathematical Details:
        The transformation consists of:
        1. Rotation: q * p * q* where p = (0, x, y, z) is the pure quaternion
        2. Translation: extracted as 2 * dual_part * conjugate(real_part)
        3. Final point = rotated_point + translation

    Process:
        1. Convert point to pure quaternion (0, x, y, z)
        2. Apply rotation using quaternion sandwich product
        3. Extract translation from dual quaternion structure
        4. Combine rotation and translation results

    Note:
        This implements the standard dual quaternion point transformation
        formula used in computer graphics and robotics.
    """
    p = jnp.array([0, *point])
    rotated_point = q_mult(q_mult(dq[0], p), q_conj(dq[0]))
    translation = 2 * q_mult(dq[1], q_conj(dq[0]))[1:]
    return rotated_point[1:] + translation

dq_transform_point_vmap = vmap(dq_transform_point, in_axes=(0, None))


def dq_normalize(dq: Float[Array, "2 4"], eps: float = 1e-6) -> Float[Array, "2 4"]:
    """
    Normalize a dual quaternion.

    Normalizes a dual quaternion by dividing both real and dual parts by the
    norm of the real part. This ensures the dual quaternion represents a
    valid unit dual quaternion for rigid transformations.

    Args:
        dq: Dual quaternion with shape (2, 4) where:
            - dq[0] is the real quaternion part (w, x, y, z)
            - dq[1] is the dual quaternion part (w', x', y', z')
        eps: Small epsilon value to prevent division by zero. Defaults to 1e-6.

    Returns:
        Normalized dual quaternion with shape (2, 4) where the real part
        has unit norm and the dual part is scaled accordingly.

    Mathematical Details:
        For dual quaternion dq = q_r + εq_d, normalization ensures:
        ||q_r|| = 1 (unit norm for real part)

        The normalization is computed as:
        dq_normalized = (q_r / ||q_r||) + ε(q_d / ||q_r||)

    Process:
        1. Compute safe norm of real part: ||q_r|| + eps
        2. Divide both real and dual parts by this norm
        3. Result represents valid unit dual quaternion

    Note:
        The eps parameter prevents numerical instability when the real
        part has very small norm. Unit dual quaternions are required
        for proper rigid transformation representation.
    """
    safe_real_norm = safe_norm(dq[0]) + eps
    real_normalized = dq[0] / safe_real_norm
    dual_normalized = dq[1] / safe_real_norm
    return jnp.stack([real_normalized, dual_normalized])


def dq_from_rot_trans(
        rotation: Float[Array, "4"],
        translation: Float[Array, "3"]
) -> Float[Array, "2 4"]:
    """
    Construct a dual quaternion from rotation quaternion and translation vector.

    Creates a dual quaternion representing a rigid transformation using the
    standard convention: dq = [r, 0.5 * (t_quat * r)]

    Args:
        rotation: Unit quaternion representing rotation with shape (4,)
                 in (w, x, y, z) format
        translation: 3D translation vector with shape (3,) as (x, y, z)

    Returns:
        Normalized dual quaternion with shape (2, 4) where:
        - result[0] = rotation quaternion (real part)
        - result[1] = 0.5 * translation_quat * rotation (dual part)

    Mathematical Details:
        The dual quaternion is constructed as:
        dq = q_r + ε * q_d

        where:
        - q_r = rotation quaternion (real part)
        - q_d = 0.5 * t * q_r (dual part)
        - t = (0, tx, ty, tz) is translation as pure quaternion
        - ε is the dual unit with ε² = 0

    Process:
        1. Convert translation vector to pure quaternion (0, x, y, z)
        2. Compute dual part as 0.5 * translation_quat * rotation
        3. Stack real and dual parts to form dual quaternion
        4. Normalize to ensure valid unit dual quaternion

    Note:
        The resulting dual quaternion represents the rigid transformation
        that first rotates by the rotation quaternion, then translates
        by the translation vector.
    """
    translation_quat = jnp.concatenate([jnp.array([0.0]), translation])
    dual = 0.5 * q_mult(translation_quat, rotation)
    return dq_normalize(jnp.stack([rotation, dual]))


