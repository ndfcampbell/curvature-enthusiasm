import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array


from ..utils import safe_normalize


def orthogonal_axes_loss(
    A: Float[Array, "N D"],
    B: Float[Array, "N D"],
    w1: float = 1.0,
    w2: float = 1.0,
    w3: float = 1.0
) -> Float[Array, ""]:
    """
     Compute a loss function that encourages matrix B to have orthogonal axes with consistent lengths.

    This loss function is designed for learning coordinate frames or basis vectors where
    orthogonality and consistent scaling are desired properties. It's commonly used in
    applications like learning rotation matrices, coordinate transformations, or
    orthonormal basis construction.

    The loss consists of two main components:
    1. Axis alignment loss: Penalizes deviation from orthogonality by measuring how
       close the Gram matrix of normalized B is to the identity matrix.
    2. Length consistency loss: Encourages all axes in B to have the same length as
       the first axis in the reference matrix A.

    Args:
        A: Reference matrix with shape (N, D) where N is the number of axes/vectors
           and D is the dimensionality. Used to determine the target length for axes in B.
        B: Predicted matrix with shape (N, D) that should be regularized toward
           orthogonality and consistent axis lengths.
        w1: Weight for the axis alignment (orthogonality) loss component.
            Higher values more strongly enforce orthogonality. Defaults to 1.0.
        w2: Weight for the length consistency loss component.
            Higher values more strongly enforce uniform axis lengths. Defaults to 1.0.
        w3: Currently unused weight parameter (likely reserved for future extensions).
            Defaults to 1.0.

    Returns:
        Scalar loss value combining both orthogonality and length consistency penalties.
        Lower values indicate better orthogonality and length consistency.

    Mathematical Details:
        - Target length is computed as ||A[0]|| (norm of first axis in A)
        - Gram matrix G = B_norm @ B_norm^T where B_norm has unit-length rows
        - Axis alignment loss = ||G - I||_F^2 (Frobenius norm squared)
        - Length consistency loss = Σ(target_length - ||B[i]||)^2 / target_length^2
        - Total loss = w1 * axis_alignment_loss + w2 * length_loss

    Example:
        >>> A = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # Identity
        >>> B = jnp.array([[0.9, 0.1, 0.0], [0.1, 0.9, 0.1], [0.0, 0.1, 0.9]])  # Nearly orthogonal
        >>> loss = orthogonal_axes_loss(A, B)
        >>> print(f"Loss: {loss:.4f}")  # Small positive
    """

    # Ensure A and B have the same shape
    assert A.shape == B.shape, "Matrices A and B must have the same shape"

    # If target_length is not provided, use the length of axes in A
    # if target_length is None:
    target_length = jnp.linalg.norm(A[0])

    B_normalized = safe_normalize(B)
    gram_matrix = jnp.dot(B_normalized, B_normalized.T)

    # Axis alignment loss
    identity_matrix = jnp.eye(B.shape[0])
    axis_alignment_loss = jnp.sum((gram_matrix - identity_matrix) ** 2)

    # Length consistency loss
    length_loss = jnp.sum((target_length - jnp.linalg.norm(B, axis=1))**2 / (target_length ** 2))

    # Combine losses with weights
    total_loss = w1 * axis_alignment_loss + w2 * length_loss

    return total_loss

def quat_multiply(quaternion1, quaternion2):
  """Multiplies two quaternions.

   Args:
     quaternion1:  A tensor of shape `[A1, ..., An, 4]`, where the last dimension
       represents a quaternion.
     quaternion2:  A tensor of shape `[A1, ..., An, 4]`, where the last dimension
       represents a quaternion.

   Returns:
     A tensor of shape `[A1, ..., An, 4]` representing quaternions.
   """
  quaternion1 = jnp.asarray(quaternion1)
  quaternion2 = jnp.asarray(quaternion2)

  x1, y1, z1, w1 = jnp.split(quaternion1, 4, axis=-1)
  x2, y2, z2, w2 = jnp.split(quaternion2, 4, axis=-1)
  x = x1 * w2 + y1 * z2 - z1 * y2 + w1 * x2
  y = -x1 * z2 + y1 * w2 + z1 * x2 + w1 * y2
  z = x1 * y2 - y1 * x2 + z1 * w2 + w1 * z2
  w = -x1 * x2 - y1 * y2 - z1 * z2 + w1 * w2
  return jnp.concatenate((x, y, z, w), axis=-1)


def quat_conjugate(quaternion):
  """Computes the conjugate of a quaternion.

  Args:
    quaternion: A tensor of shape `[A1, ..., An, 4]`, where the last dimension
      represents a normalized quaternion.

  Returns:
    A tensor of shape `[A1, ..., An, 4]`, where the last dimension represents
    a normalized quaternion.
  """
  quaternion = jnp.asarray(quaternion)

  # xyz, w = jnp.split(quaternion, [3, quaternion.shape[-1] - 3], axis=-1)

  xyz = quaternion[..., :3]
  w = quaternion[..., 3:]

  return jnp.concatenate((-xyz, w), axis=-1)


def quat_rotate(point, quaternion):
  """Rotates a point using a quaternion.

  Args:
    point: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents a 3d point.
    quaternion: A tensor of shape `[A1, ..., An, 4]`, where the last dimension
      represents a normalized quaternion.

  Returns:
    A tensor of shape `[A1, ..., An, 3]`, where the last dimension represents a
    3d point.
  """
  point = jnp.asarray(point)
  quaternion = jnp.asarray(quaternion)

  # Padding logic
  padding = ((0, 0),) * (point.ndim - 1) + ((0, 1),)
  point = jnp.pad(point, padding, mode="constant")

  point = quat_multiply(quaternion, point)
  point = quat_multiply(point, quat_conjugate(quaternion))
  # xyz, _ = jnp.split(point, [3, point.shape[-1] - 3], axis=-1)

  xyz = point[..., :3]

  return xyz


def l2_normalize(x, axis=-1, eps=1e-12):
  """L2 normalize the input x along the specified axis."""
  norm = jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(x), axis=axis, keepdims=True), eps))
  return x / norm


def quat_normalize(quaternion, eps=1e-8):
  """Normalizes a quaternion.

  Args:
    quaternion:  A tensor of shape `[A1, ..., An, 4]`, where the last dimension
      represents a quaternion.
    eps: A lower bound value for the norm that defaults to 1e-12.

  Returns:
    A N-D tensor of shape `[?, ..., ?, 1]` where the quaternion elements have
    been normalized.
  """
  quaternion = jnp.asarray(quaternion)
  return l2_normalize(quaternion, axis=-1, eps=eps)


def quat_from_euler(angles):
  """Converts an Euler angle representation to a quaternion.

  Uses the z-y-x rotation convention (Tait-Bryan angles).

  Args:
    angles: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the three Euler angles. `[..., 0]` is the angle about `x` in
      radians, `[..., 1]` is the angle about `y` in radians and `[..., 2]` is
      the angle about `z` in radians.

  Returns:
    A tensor of shape `[A1, ..., An, 4]`, where the last dimension represents
    a normalized quaternion.
  """
  angles = jnp.asarray(angles)

  half_angles = angles / 2.0
  cos_half_angles = jnp.cos(half_angles)
  sin_half_angles = jnp.sin(half_angles)
  return _build_quaternion_from_sines_and_cosines(sin_half_angles,
                                                  cos_half_angles)


def _build_quaternion_from_sines_and_cosines(sin_half_angles, cos_half_angles):
  """Builds a quaternion from sines and cosines of half Euler angles.

  Args:
    sin_half_angles: A tensor of shape `[A1, ..., An, 3]`, where the last
      dimension represents the sine of half Euler angles.
    cos_half_angles: A tensor of shape `[A1, ..., An, 3]`, where the last
      dimension represents the cosine of half Euler angles.

  Returns:
    A tensor of shape `[A1, ..., An, 4]`, where the last dimension represents
    a quaternion.
  """
  c1, c2, c3 = jnp.split(cos_half_angles, 3, axis=-1)
  s1, s2, s3 = jnp.split(sin_half_angles, 3, axis=-1)
  w = c1 * c2 * c3 + s1 * s2 * s3
  x = -c1 * s2 * s3 + s1 * c2 * c3
  y = c1 * s2 * c3 + s1 * c2 * s3
  z = -s1 * s2 * c3 + c1 * c2 * s3
  return jnp.concatenate((x, y, z, w), axis=-1)

def acap_energy(vertices_rest_pose,
           vertices_deformed_pose,
           quaternions,
           edges,
           vertex_weight: None,
           edge_weight: None,
           conformal_energy: bool = True,
           aggregate_loss: bool = True):

  vertices_rest_pose = jnp.asarray(vertices_rest_pose)
  vertices_deformed_pose = jnp.asarray(vertices_deformed_pose)
  quaternions = jnp.asarray(quaternions)
  edges = jnp.asarray(edges)

  if vertex_weight is not None:
    vertex_weight = jnp.asarray(vertex_weight)
  if edge_weight is not None:
    edge_weight = jnp.asarray(edge_weight)

  # JAX doesn't have static shape checking similar to TensorFlow. Most of the
  # shape-related error checking is done dynamically.

  if not conformal_energy:
    quaternions = quat_normalize(quaternions)
  indices_i, indices_j = edges.T
  vertices_i_rest = vertices_rest_pose[indices_i]
  vertices_j_rest = vertices_rest_pose[indices_j]
  vertices_i_deformed = vertices_deformed_pose[indices_i]
  vertices_j_deformed = vertices_deformed_pose[indices_j]

  weights_shape = vertices_i_rest.shape[-2]
  if vertex_weight is not None:
    weight_i = vertex_weight[indices_i]
    weight_j = vertex_weight[indices_j]
  else:
    weight_i = weight_j = jnp.ones(weights_shape, dtype=vertices_rest_pose.dtype)
  weight_i = jnp.expand_dims(weight_i, axis=-1)
  weight_j = jnp.expand_dims(weight_j, axis=-1)
  if edge_weight is not None:
    weight_ij = edge_weight
  else:
    weight_ij = jnp.ones(weights_shape, dtype=vertices_rest_pose.dtype)
  weight_ij = jnp.expand_dims(weight_ij, axis=-1)

  quaternion_i = quaternions[indices_i]
  quaternion_j = quaternions[indices_j]

  deformed_ij = vertices_i_deformed - vertices_j_deformed
  rotated_rest_ij = quat_rotate((vertices_i_rest - vertices_j_rest), quaternion_i)
  energy_ij = weight_i * weight_ij * (deformed_ij - rotated_rest_ij)

  deformed_ji = vertices_j_deformed - vertices_i_deformed
  rotated_rest_ji = quat_rotate((vertices_j_rest - vertices_i_rest), quaternion_j)
  energy_ji = weight_j * weight_ij * (deformed_ji - rotated_rest_ji)

  # energy_ij_squared = jnp.dot(energy_ij, energy_ij)
  # energy_ji_squared = jnp.dot(energy_ji, energy_ji)

  energy_ij_squared = jnp.sum(energy_ij ** 2, axis=-1, keepdims=True)
  energy_ji_squared = jnp.sum(energy_ji ** 2, axis=-1, keepdims=True)

  rigid_res = quaternion_j - quaternion_i
  rigid_energy = jnp.sum(jnp.mean(jnp.square(rigid_res), axis=0))

  if aggregate_loss:
    average_energy_ij = jnp.mean(energy_ij_squared, axis=-1)
    average_energy_ji = jnp.mean(energy_ji_squared, axis=-1)
    return (average_energy_ij + average_energy_ji) / 2.0, rigid_energy

  return jnp.concatenate((energy_ij_squared, energy_ji_squared), axis=-1), rigid_energy


class ACAP_Surface_Energy(eqx.Module):
    edges: Float[Array, "E 2"]  # edge indices

    def __init__(self, edges: Float[Array, "E 2"]):
        self.edges = edges

    def __call__(
            self,
            initial_points: Float[Array, "V 3"],
            y_pred_traj: Float[Array, "T V 3"],
            Q: Float[Array, "V 3 3"],
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        acap, rigid = jax.vmap(acap_energy, (None, 0, 0, None, None, None))(
            initial_points, y_pred_traj, Q, jax.lax.stop_gradient(self.edges), None, None)
        return jnp.mean(acap), jnp.mean(rigid)


class ACAP_Sample_Energy(eqx.Module):
    offset: float = eqx.field(static=True)
    offset_matrix: Float[Array, "3 3"]  # axis-aligned offset matrix

    def __init__(self, offset=0.01):
        self.offset = offset
        self.offset_matrix = jnp.eye(3) * offset

    def calc_acap_sample_energy(
        self,
        deformed_axis: Float[Array, "3"],
        Q: Float[Array, "3 3"],
    ) -> Float[Array, ""]:
        rotated_deformed_axis = quat_rotate(deformed_axis, Q)
        return orthogonal_axes_loss(jax.lax.stop_gradient(self.offset_matrix), rotated_deformed_axis)

    def acap_sample_energy_per_step(
        self,
        samples_pred: Float[Array, "N 12"],   # reshaped later to (N, 4, 3)
        Q: Float[Array, "N 3 3"],
    ) -> tuple[
        Float[Array, "N"],                    # acap_energy
        Float[Array, "N 3 3"],                # offset_preds
        Float[Array, "N 3 3"],                # diffs
    ]:
        samples_pred = samples_pred.reshape(-1, 4, 3)
        y_pred = samples_pred[:, 0, :]
        offset_preds = samples_pred[:, 1:, :]
        diffs = (offset_preds - y_pred[:, jnp.newaxis, :]) + 1e-8
        acap_energy = jax.vmap(self.calc_acap_sample_energy, (0, 0))(diffs, Q)
        return acap_energy, offset_preds, diffs

    def __call__(
        self,
        samples_traj: Float[Array, "T N 12"],
        Q: Float[Array, "T N 3 3"],
        offset: float = 0.01,
    ) -> tuple[
        Float[Array, "T N"],        # acap_energy over trajectory
        Float[Array, "N 3 3"],      # offset_preds at last step
        Float[Array, "N 3 3"],      # diffs at last step
    ]:
        acap_energy, offset_preds, diffs = jax.vmap(self.acap_sample_energy_per_step, (0, 0))(
            samples_traj, Q)
        return acap_energy, offset_preds[-1], diffs[-1]
