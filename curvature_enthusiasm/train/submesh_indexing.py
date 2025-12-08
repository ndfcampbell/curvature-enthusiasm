import numpy as np
import jax.numpy as jnp


def setup_triangle_tracker(f, tracked_face_ids):
    """
    Setup tracking arrays (do this once, outside JAX).

    Parameters:
    -----------
    f : ndarray, shape (M, 3)
        Face indices
    tracked_face_ids : ndarray
        Triangle IDs to track

    Returns:
    --------
    unique_vertex_ids : ndarray, shape (N_unique,)
        Unique vertex IDs used by tracked faces
    local_face_indices : ndarray, shape (K, 3)
        Face indices remapped to local vertex array
    """
    tracked_face_ids = np.asarray(tracked_face_ids)
    tracked_faces = f[tracked_face_ids]

    # Get unique vertex IDs
    unique_vertex_ids = np.unique(tracked_faces)

    # Create mapping array: global_vid -> local_idx
    max_vid = f.max()
    vertex_id_map = np.full(max_vid + 1, -1, dtype=np.int32)
    vertex_id_map[unique_vertex_ids] = np.arange(len(unique_vertex_ids))

    # Remap faces to local indices
    local_face_indices = vertex_id_map[tracked_faces]

    return unique_vertex_ids, local_face_indices


def extract_local_vertices(v, unique_vertex_ids):
    """
    Extract subset of vertices (JAX-compatible).

    Parameters:
    -----------
    v : array, shape (N, 3)
        Full vertex array
    unique_vertex_ids : array, shape (N_unique,)
        Vertex IDs to extract

    Returns:
    --------
    local_vertices : array, shape (N_unique, 3)
    """
    return v[unique_vertex_ids]


def compute_centers(local_vertices, local_face_indices):
    """
    Compute triangle centers from local vertices (JAX-compatible).

    Parameters:
    -----------
    local_vertices : array, shape (N_unique, 3)
        Local vertex positions (already updated/received)
    local_face_indices : array, shape (K, 3)
        Face indices in local coordinates

    Returns:
    --------
    centers : array, shape (K, 3)
        Triangle centers
    """
    triangle_verts = local_vertices[local_face_indices]  # (K, 3, 3)
    centers = triangle_verts.mean(axis=1)  # (K, 3)
    return centers
