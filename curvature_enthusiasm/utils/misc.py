import jax
import jax.numpy as jnp
import numpy as np
from plyfile import PlyData, PlyElement

tiny_val = np.float32(np.finfo(np.float32).tiny)
min_val = np.float32(np.finfo(np.float32).min)
max_val = np.float32(np.finfo(np.float32).max)

def scalar_float(x):
    return jnp.array(x, dtype=jnp.float64)

def scalar_float32(x):
    return jnp.array(x, dtype=jnp.float32)

def scalar_int(x):
    return jnp.array(x, dtype=jnp.int32)

def default_floating_dtype():
    if jax.config.jax_enable_x64:  # pyright: ignore
        return jnp.float64
    else:
        return jnp.float32

def save_skeleton_to_ply(joints, bones, filename):
    # Ensure joints is a 2D numpy array
    joints = np.asarray(joints)
    if joints.ndim != 2 or joints.shape[1] != 3:
        raise ValueError("joints should be a 2D array with shape (n, 3)")

    # Prepare vertex data
    vertex_data = np.zeros(joints.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_data['x'] = joints[:, 0]
    vertex_data['y'] = joints[:, 1]
    vertex_data['z'] = joints[:, 2]

    # Ensure bones is a 2D numpy array
    bones = np.asarray(bones)
    if bones.ndim != 2 or bones.shape[1] != 2:
        raise ValueError("bones should be a 2D array with shape (m, 2)")

    # Prepare edge data
    edge_data = np.zeros(bones.shape[0], dtype=[('vertex1', 'i4'), ('vertex2', 'i4')])
    edge_data['vertex1'] = bones[:, 0]
    edge_data['vertex2'] = bones[:, 1]

    # Create PlyElements
    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    edge_element = PlyElement.describe(edge_data, 'edge')

    # Create PlyData object and write to file
    ply_data = PlyData([vertex_element, edge_element], text=True)
    ply_data.write(filename)


def remove_padding(arr, pad_value=jnp.nan):
    """
    Remove rows that are entirely equal to pad_value.

    Args:
        arr: jnp.ndarray of shape (M, D) or (N, M, D)
        pad_value: value used for padding (default -1)

    Returns:
        If input is (M,D): returns (M',D) with only valid rows.
        If input is (N,M,D): returns a Python list of (Mi',D) arrays (since lengths differ).
    """
    if arr.ndim == 2:
        mask = ~jnp.all(arr == pad_value, axis=-1)   # (M,)
        return arr[mask]
    elif arr.ndim == 3:
        return [remove_padding(a, pad_value) for a in arr]  # list of arrays
    else:
        raise ValueError("Expected array of shape (M,D) or (N,M,D)")

def pad_variable_length(arr_list, pad_value=-1.0):
    """
    Pad a list of (Mi, D) JAX arrays to shape (N, maxM, D) with pad_value.

    Args:
        arr_list: list of jnp.ndarray, each shape (Mi, D) with variable Mi
        pad_value: value used for padding

    Returns:
        jnp.ndarray of shape (N, maxM, D)
    """
    N = len(arr_list)
    if N == 0:
        return jnp.zeros((0, 0, 0))

    D = arr_list[0].shape[1]
    maxM = max(a.shape[0] for a in arr_list)

    padded = []
    for a in arr_list:
        Mi = a.shape[0]
        pad_len = maxM - Mi
        if pad_len > 0:
            pad_block = jnp.full((pad_len, D), pad_value, dtype=a.dtype)
            a = jnp.concatenate([a, pad_block], axis=0)
        padded.append(a)

    return jnp.stack(padded, axis=0)


