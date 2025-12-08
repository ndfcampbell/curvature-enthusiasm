import jax
import jax.nn as jnn
import jax.numpy as jnp
import equinox as eqx
import jaxlie
from jaxtyping import Array, Float

from ..utils import safe_normalize, pin_to_hemisphere, default_floating_dtype

from .lip_mlp import Lip_MLP

# GSO+6
def compute_rot_matrix_from_ortho6d(ortho6d: Float[Array, "B 6"]) -> Float[Array, "B 3 3"]:
    """
    Convert 6D orthogonal representation to rotation matrices.

    Implements the method from Zhou et al. 'On the Continuity of Rotation Representations'
    which provides a continuous representation of SO(3) rotations using 6 parameters.

    Args:
        ortho6d: 6D orthogonal representation with shape (B, 6).
                First 3 elements represent the first column vector,
                last 3 elements represent the second column vector.

    Returns:
        Rotation matrices with shape (B, 3, 3) where each matrix is orthogonal
        with determinant +1 (proper rotation matrix).

    Note:
        The algorithm uses Gram-Schmidt orthogonalization to ensure orthogonality
        between the first two column vectors, then computes the third via cross product
        to form a right-handed coordinate frame.
    """
    B = ortho6d.shape[0]
    x_raw = ortho6d[:, 0:3]
    y_raw = ortho6d[:, 3:6]

    # Normalize first vector
    x = safe_normalize(x_raw, axis=-1)               # (B,3)
    # Make y orthogonal to x via Gram–Schmidt (more stable than cross-first)
    y = y_raw - (jnp.sum(x * y_raw, axis=-1, keepdims=True) * x)
    y = safe_normalize(y, axis=-1)                   # (B,3)

    # z = x × y gives a right-handed frame; it’s already unit if x,y are orthonormal
    z = jnp.cross(x, y, axis=-1)                     # (B,3)

    # Stack columns to form rotation matrix (B, 3, 3) with columns [x y z]
    R = jnp.stack([x, y, z], axis=-1)
    return R


class GSO_MLP(eqx.Module):
    """
    Gramian-Schmidt Orthogonalization MLP for learning rotation representations.

    This module learns to predict 6D orthogonal representations which are then
    converted to quaternions. The 6D representation provides a continuous,
    singularity-free parameterization of SO(3) rotations.

    The network outputs small perturbations (scaled by 1e-1) around the identity
    rotation to encourage stable training dynamics.

    Attributes:
        mlp: The underlying Lipschitz-constrained MLP that maps input states
             to 6D orthogonal representations.
    """

    mlp: Lip_MLP

    def __init__(
        self,
        in_size: int,
        *,
        key,
        # configuration options
        width: int | None = 128,
        depth: int | None = 3,
        hidden: list[int] | None = None,
        activation=jnn.gelu,
        out_size: int = 6,
        dtype=None,
        **kwargs,
    ):
        """
        Initialize the GSO MLP module.

        Args:
            in_size: Dimension of input state vectors.
            key: PRNG key for parameter initialization.
            width: Width of hidden layers if `hidden` is None. Defaults to 128.
            depth: Number of hidden layers if `hidden` is None. Defaults to 3.
            hidden: Explicit list of hidden layer sizes. If provided, overrides
                   `width` and `depth` parameters.
            activation: Activation function for hidden layers. Defaults to GELU.
            out_size: Output dimension. Should be 6 for 6D orthogonal representation.
                     Defaults to 6.
            dtype: Data type for parameters. If None, uses default floating dtype.
            **kwargs: Additional arguments passed to parent class.

        Note:
            The MLP is Lipschitz-constrained to ensure stable training dynamics
            and bounded outputs.
        """
        super().__init__(**kwargs)

        self.mlp = Lip_MLP(
            in_size=in_size,
            key=key,
            width=width,
            depth=depth,
            hidden=hidden,
            activation=activation,
            out_size=out_size,
            dtype=dtype,
        )

    def __call__(self, states: Float[Array, "B D"]) -> Float[Array, "B 4"]:
        """
        Forward pass: convert input states to normalized quaternions.

        Args:
            states: Input state vectors with shape (B, D) where B is batch size
                   and D is the input dimension.

        Returns:
            Normalized quaternions with shape (B, 4) in XYZW order.
            Quaternions are pinned to the positive hemisphere for uniqueness
            and normalized to unit length.

        Note:
            The forward pass involves:
            1. MLP prediction of 6D orthogonal representation
            2. Scaling by 1e-1 and adding identity bias
            3. Converting to rotation matrix via Gram-Schmidt
            4. Converting to quaternion via SO(3) representation
            5. Pinning to positive hemisphere and normalizing

            The output quaternions use XYZW ordering, which differs from
            some other conventions (WXYZ). The comment notes a difference
            between ACAP quaternions ([0,0,0,1]) and rigid transformation
            quaternions ([1,0,0,0]) for identity representation.
        """
        ortho6d = 1e-1 * jax.vmap(self.mlp)(states).reshape(-1, 6)

        # Add identity bias: [1,0,0,1,0,0] represents identity rotation
        identity6 = jnp.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=ortho6d.dtype)
        pred_rot = ortho6d + identity6
        R = compute_rot_matrix_from_ortho6d(pred_rot)

        # Convert to SO(3) representation and extract quaternion
        so3 = jaxlie.SO3.from_matrix(R)
        # NOTE ACAP QUATERNIONS ARE W = [0,0,0,1], RIGID TRNANFORMATION QUATERNIONS ARE W = [1,0,0,0]
        quat = so3.as_quaternion_xyzw()  # (x,y,z,w)

        # Ensure quaternion uniqueness by pinning to positive hemisphere
        quat = pin_to_hemisphere(quat, hemisphere='positive', order='xyzw')
        quat = quat.reshape(states.shape[0], 4)

        # Normalize to unit quaternion
        quat = safe_normalize(quat, axis=-1, keepdims=True)

        return quat

