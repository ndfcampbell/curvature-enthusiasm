import jax.numpy as jnp
import jax.random as jr
from jaxlie import SO3, SE3

from jaxtyping import Float, Array

from ..utils import safe_normalize

def make_bone_transform(
    x_axis: Float[Array, "3"],
    y_axis: Float[Array, "3"],
    z_axis: Float[Array, "3"],
    origin: Float[Array, "3"],
):
    """
    Build an SE3 transform from local axes + origin.

    Parameters
    ----------
    x_axis, y_axis, z_axis : (3,) array-like
        Local axes (will be normalized).
    origin : (3,) array-like
        Origin in world coordinates.
    safe_normalize : callable
        Function to normalize vectors safely.

    Returns
    -------
    SE3
        Rigid transform from local -> world.
    """
    x_axis = safe_normalize(x_axis)
    y_axis = safe_normalize(y_axis)
    z_axis = safe_normalize(z_axis)

    R_mat = jnp.stack([x_axis, y_axis, z_axis], axis=1)  # columns are local axes
    so3 = SO3.from_matrix(R_mat)
    return SE3.from_rotation_and_translation(so3, jnp.asarray(origin))



def generate_samples_around_bone(
    p0: Float[Array, "3"],
    p1: Float[Array, "3"],
    init_local_axes: tuple[Float[Array, "3"], Float[Array, "3"], Float[Array, "3"]],
    radius_proportion: float,
    min_length_proportion: float = 0.1,
    max_length_proportion: float = 0.9,
    num_length_samples: int = 5,
    num_radial_samples: int = 2,
    num_angular_samples: int = 5,
    dtype=jnp.float32,
    safe_normalize=None,
) -> Float[Array, "N 3"]:
    """
    Deterministic cylindrical lattice of samples around a bone segment.

    Parameters
    ----------
    p0, p1 : (3,) array-like
        Endpoints of the bone segment in world coordinates.
    init_local_axes : tuple of (3,) array-like
        Initial local coordinate frame axes (x, y, z).
    radius_proportion : float
        Radius relative to bone length for sampling cylinder.
    min_length_proportion : float, optional
        Minimum proportion along bone length to start sampling.
    max_length_proportion : float, optional
        Maximum proportion along bone length to stop sampling.
    num_length_samples : int, optional
        Number of samples along bone length.
    num_radial_samples : int, optional
        Number of samples along radial direction.
    num_angular_samples : int, optional
        Number of angular samples around cylinder.
    dtype : dtype, optional
        Data type for computation.
    safe_normalize : callable, required
        Function to normalize vectors safely.

    Returns
    -------
    samples : (N, 3) array
        Sample points in world coordinates.
    """
    assert safe_normalize is not None, "Please pass your safe_normalize function"

    p0 = jnp.asarray(p0, dtype)
    p1 = jnp.asarray(p1, dtype)
    x_axis, y_axis, z_axis = [jnp.asarray(a, dtype) for a in init_local_axes]

    # Bone metrics
    bone_len = jnp.linalg.norm(p1 - p0)
    max_r = radius_proportion * bone_len

    # Local grid in cylindrical coords: (length, radius, angle)
    z_lin = jnp.linspace(min_length_proportion * bone_len,
                         max_length_proportion * bone_len,
                         num_length_samples, dtype=dtype)

    num_radial_samples = int(jnp.maximum(1, num_radial_samples))
    r_lin = jnp.linspace(0.0, max_r, num_radial_samples, dtype=dtype)

    theta = jnp.linspace(0.0, 2.0 * jnp.pi, num_angular_samples,
                         endpoint=False, dtype=dtype)

    Z, R, TH = jnp.meshgrid(z_lin, r_lin, theta, indexing="ij")
    local_pts = jnp.stack(
        [Z, R * jnp.cos(TH), R * jnp.sin(TH)], axis=-1
    ).reshape(-1, 3)

    # Apply rigid transform
    T = make_bone_transform(x_axis, y_axis, z_axis, p0)
    return T.apply(local_pts)



def generate_random_samples_around_bone(
    p0: Float[Array, "3"],
    p1: Float[Array, "3"],
    init_local_axes: tuple[Float[Array, "3"], Float[Array, "3"], Float[Array, "3"]],
    radius_proportion: float,
    min_length_proportion: float = 0.1,
    max_length_proportion: float = 0.9,
    num_samples: int = 50,
    key: Array | None = None,
    dtype=jnp.float32,
) -> Float[Array, "N 3"]:
    """
        Random point cloud around a bone:
          - A deterministic set of samples along the central axis (20% of total, min 2).
          - Remaining points uniformly sampled in the disk of radius max_r for each z.

        Parameters
        ----------
        p0, p1 : (3,) array-like
            Endpoints of the bone segment in world coordinates.
        init_local_axes : tuple of (3,) array-like
            Local coordinate frame axes (x, y, z).
        radius_proportion : float
            Radius relative to bone length for sampling cylinder.
        min_length_proportion : float, optional
            Minimum proportion along bone length to start sampling.
        max_length_proportion : float, optional
            Maximum proportion along bone length to stop sampling.
        num_samples : int, optional
            Total number of samples (default 50).
        key : PRNGKey
            Random key for sampling.
        dtype : dtype, optional
            Data type for computation.

        Returns
        -------
        world_pts : (N, 3) array
            Sampled points in world coordinates.
    """

    p0 = jnp.asarray(p0, dtype)
    p1 = jnp.asarray(p1, dtype)
    x_axis, y_axis, z_axis = [jnp.asarray(a, dtype) for a in init_local_axes]

    # Bone metrics
    bone_len = jnp.linalg.norm(p1 - p0)
    max_r = radius_proportion * bone_len

    # Central-axis samples (deterministic)
    n_central = max(2, int(0.20 * int(num_samples)))
    n_rand = int(num_samples) - n_central

    z_central = jnp.linspace(
        min_length_proportion * bone_len,
        max_length_proportion * bone_len,
        n_central,
        dtype=dtype,
    )
    central_local = jnp.stack(
        [z_central, jnp.zeros_like(z_central), jnp.zeros_like(z_central)], axis=-1
    )  # (n_central, 3)

    # Random ring samples (uniform over disk area)
    key, kz, kr, kth = jr.split(key, 4)

    z = jr.uniform(
        kz,
        (n_rand,),
        minval=min_length_proportion * bone_len,
        maxval=max_length_proportion * bone_len,
        dtype=dtype,
    )
    r = jnp.sqrt(jr.uniform(kr, (n_rand,), dtype=dtype)) * max_r
    th = jr.uniform(kth, (n_rand,), minval=0.0, maxval=2.0 * jnp.pi, dtype=dtype)

    ring_local = jnp.stack([z, r * jnp.cos(th), r * jnp.sin(th)], axis=-1)  # (n_rand, 3)

    # Concatenate local points
    local_pts = jnp.concatenate([central_local, ring_local], axis=0)  # (N,3)

    # Apply rigid transform local->world
    T = make_bone_transform(x_axis, y_axis, z_axis, p0)
    world_pts = T.apply(local_pts)

    return world_pts





