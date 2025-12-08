from collections import defaultdict
from typing import Dict, Tuple, Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from curvature_enthusiasm.utils import safe_normalize, safe_normalize_with_norm

# Pre-compute indices since they're constant
INDICES_1 = jnp.array([i - 1 for i in range(1, 6) for j in range(i + 1, 7)])
INDICES_2 = jnp.array([j - 1 for i in range(1, 6) for j in range(i + 1, 7)])


def normalize(shape):
  return (shape - jnp.mean(shape, axis=0)) / jnp.mean(jnp.std(shape, axis=0))

def coordinates(
    a: Float[Array, "N 6"],
    b: Float[Array, "N 6"]
) -> Float[Array, "N 15"]:
    """
    Compute coordinates using pre-computed index pairs.

    Args:
        a: First input array of shape (N, 6)
        b: Second input array of shape (N, 6)

    Returns:
        Array of shape (N, 15) containing computed coordinates
    """
    # Vectorized computation using pre-computed indices
    result = (a[:, INDICES_1] * b[:, INDICES_2] -
              a[:, INDICES_2] * b[:, INDICES_1])

    # Reshape to final dimensions
    return result.reshape(-1, 15)


# def pre_comp_nc_vec_con(
#     M1: Int[Array, "F 3"],
#     ex_bdry: bool
# ) -> Dict[str, Any]:
#     """takes in shape, mesh, bdry boolean and extracts
#         required info for downline tasks with Normal Cycles.
#
#     Args:
#         gen_pts (array): vertices of input shape of size (n,3)
#         M1 (array): mesh structure of input shape of size (m,3)
#         ex_bdry (bool): True if input shape has boundary, false otherwise.
#
#     Returns:
#        dict: A dictionary of useful structural indices for NC metric.
#     """
#     max_ = 0
#     edges_for_verts_inds = []
#
#     # for each unique edge compute triangles attached
#     Acc = [[0, 1], [2, 0], [1, 2]]
#     # Rej = [[1, 0], [0, 2], [2, 1]]
#
#     e1 = jnp.column_stack([M1[:, 0], M1[:, 1]])
#     e2 = jnp.column_stack([M1[:, 1], M1[:, 2]])
#     e3 = jnp.column_stack([M1[:, 2], M1[:, 0]])
#
#     # indices of unique edges - using numpy for unique operations
#     all_edges = jnp.vstack([e1, e2, e3])
#     all_edges_np = np.array(all_edges)
#     sorted_edges_np = np.sort(all_edges_np, axis=1)
#     u_edge_inds = np.unique(sorted_edges_np, axis=0)
#     u_edge_inds = jnp.array(u_edge_inds)  # convert back to JAX array
#
#     VS = []
#     lens = []
#
#     # Using regular Python dictionary since JAX doesn't support defaultdict
#     dic = defaultdict(list)
#
#     print("looping on triangulation")
#
#     # Convert to numpy for iteration
#     M1_np = np.array(M1)
#
#     for i in range(len(M1_np)):
#         e_1 = tuple(sorted((int(M1_np[i, 0]), int(M1_np[i, 1]))))
#         e_2 = tuple(sorted((int(M1_np[i, 1]), int(M1_np[i, 2]))))
#         e_3 = tuple(sorted((int(M1_np[i, 2]), int(M1_np[i, 0]))))
#
#         dic[e_1].append(i)
#         dic[e_2].append(i)
#         dic[e_3].append(i)
#
#     # store boundary edges
#     bdry = []
#     print('collecting bdry edges')
#     print(u_edge_inds.shape[0])
#
#     u_edge_inds_np = np.array(u_edge_inds)
#
#     for inds in range(u_edge_inds_np.shape[0]):
#         edge_tuple = tuple(sorted((int(u_edge_inds_np[inds][0]), int(u_edge_inds_np[inds][1]))))
#         vs = dic[edge_tuple]
#
#         if len(vs) == 1:
#             bdry.append(u_edge_inds_np[inds])
#             VS.append(np.array((vs[0], -1, -1, -1)))
#             lens.append(True)
#         else:
#             VS.append(np.append(np.array(vs), [-1] * (4 - len(vs))))
#             lens.append(False)
#
#     stack = np.vstack(VS).astype(int)
#     stack = jnp.array(stack)  # convert back to JAX array
#     lens = jnp.array(lens)
#
#     if ex_bdry and bdry:
#         bdry_stack = jnp.vstack([jnp.array(b) for b in bdry])
#         bdry_stack_np = np.array(bdry_stack)
#         bdry_verts_np = np.unique(bdry_stack_np.reshape(-1))
#         bdry_verts = jnp.array(bdry_verts_np)
#
#         # Calculate indices where boundary vertices appear in boundary edges
#         lis = [np.where((bdry_stack_np == bdry_verts_np[i]).sum(1))[0].shape for i in range(len(bdry_verts_np))]
#         max_lis = max(lis)
#         min_lis = min(lis)
#
#         if max_lis == min_lis:
#             bdry_vert_ed_inds = np.vstack(
#                 [np.where((bdry_stack_np == bdry_verts_np[i]).sum(1))[0] for i in range(len(bdry_verts_np))]
#             )
#             bdry_vert_edges = bdry_stack_np[bdry_vert_ed_inds]
#         else:
#             # Handle case where vertices have different numbers of boundary edges
#             bdry_vert_ed_inds = np.vstack(
#                 [np.append(np.where((bdry_stack_np == bdry_verts_np[i]).sum(1))[0], [0] * (max_lis[0] - lis[i][0]))
#                  for i in range(len(bdry_verts_np))]
#             )
#             bdry_vert_edges = bdry_stack_np[bdry_vert_ed_inds]
#
#             for i in range(len(bdry_vert_edges)):
#                 pad_length = int(max_lis[0] - lis[i][0])
#                 if pad_length > 0:
#                     bdry_vert_edges[i, -pad_length:, :] = 0.0
#
#         bdry_vert_edges = jnp.array(bdry_vert_edges)
#         bdry_vert_ed_inds = jnp.array(bdry_vert_ed_inds)
#         print(bdry_vert_edges)
#     else:
#         bdry_verts = None
#         bdry_stack = None
#         bdry_vert_ed_inds = None
#         bdry_vert_edges = None
#
#     # Initialize coordinates array
#     coords = [jnp.zeros((stack.shape[0], 1)) for i in range(4)]
#
#     # Convert for iteration
#     u_edge_inds_np = np.array(u_edge_inds)
#     stack_np = np.array(stack)
#     M1_np = np.array(M1)
#
#     # Initialize numpy arrays for later conversion
#     coords_np = [np.zeros((stack.shape[0], 1)) for i in range(4)]
#
#     for index, edge in enumerate(u_edge_inds_np):
#         for i in range(4):
#             if stack_np[index][i] == -1:
#                 continue
#
#             # Find indices of edge vertices in the triangle
#             triangle = M1_np[stack_np[index][i]]
#             v0_idx = np.where(triangle == edge[0])[0][0]
#             v1_idx = np.where(triangle == edge[1])[0][0]
#             test = [v0_idx, v1_idx]
#
#             # Check if this ordering is accepted
#             if test in Acc:
#                 coords_np[i][index] = 1.0
#             else:
#                 coords_np[i][index] = -1.0
#
#     # Convert back to JAX arrays
#     coords = [jnp.array(coord) for coord in coords_np]
#
#     # Return results as dictionary
#     return {
#         'stack': stack,
#         'edges_for_verts_inds': jnp.array(edges_for_verts_inds),
#         'u_edge_inds': u_edge_inds,
#         'max_': max_,
#         'lens': lens,
#         'coords': coords,
#         'bdry_verts': bdry_verts,
#         'bdry_vert_edges': bdry_vert_edges
#     }


def pre_comp_nc_vec_con(
    M1,            # Int[Array, "F 3"] — accepts array-like; converted to NumPy internally
    ex_bdry: bool
) -> Dict[str, Any]:
    """
    NumPy-only computation that reproduces the first implementation's logic/quirks,
    then converts all returned values to JAX arrays at the end.
    """

    # --- to NumPy (and int dtype) ---
    M1 = np.asarray(M1)
    if not np.issubdtype(M1.dtype, np.integer):
        M1 = M1.astype(np.int64)
    F = M1.shape[0]

    max_ = 0
    edges_for_verts_inds = []  # preserved for parity

    # Accepted directed (local) edge orientations in a triangle
    Acc = [[0, 1], [2, 0], [1, 2]]

    # directed edges per face (for sign test via Acc later)
    e1 = np.column_stack([M1[:, 0], M1[:, 1]])
    e2 = np.column_stack([M1[:, 1], M1[:, 2]])
    e3 = np.column_stack([M1[:, 2], M1[:, 0]])

    # unique undirected edges
    all_edges = np.vstack([e1, e2, e3])
    sorted_edges = np.sort(all_edges, axis=1)
    u_edge_inds = np.unique(sorted_edges, axis=0)

    # Build map: undirected edge -> incident faces (by index)
    dic = defaultdict(list)
    for i in range(F):
        e_1 = tuple(sorted((int(M1[i, 0]), int(M1[i, 1]))))
        e_2 = tuple(sorted((int(M1[i, 1]), int(M1[i, 2]))))
        e_3 = tuple(sorted((int(M1[i, 2]), int(M1[i, 0]))))
        dic[e_1].append(i)
        dic[e_2].append(i)
        dic[e_3].append(i)

    # For each unique edge, collect incident faces (pad to 4 with -1)
    VS = []
    lens = []
    bdry = []

    for inds in range(u_edge_inds.shape[0]):
        edge_tuple = tuple(sorted((int(u_edge_inds[inds, 0]), int(u_edge_inds[inds, 1]))))
        vs = dic[edge_tuple]
        if len(vs) == 1:
            bdry.append(u_edge_inds[inds])
            VS.append(np.array((vs[0], -1, -1, -1), dtype=int))
            lens.append(True)
        else:
            padded = np.append(np.array(vs, dtype=int), [-1] * (4 - len(vs)))
            VS.append(padded)
            lens.append(False)

    stack = np.vstack(VS).astype(int)   # (E,4), -1 padded
    lens = np.array(lens, dtype=bool)   # (E,)

    # --- Boundary aggregates (match first version's quirks) ---
    if ex_bdry and len(bdry) > 0:
        bdry_stack = np.vstack(bdry).astype(int)     # (Be,2)
        bdry_verts = np.unique(bdry_stack.reshape(-1)).astype(int)

        # For each boundary vertex, collect indices of boundary edges that contain it
        edge_idx_lists = []
        shape_list = []
        for v in bdry_verts:
            rows = np.where((bdry_stack == v).sum(1))[0]
            edge_idx_lists.append(rows)
            shape_list.append(rows.shape)   # tuple e.g. (k,)

        max_lis = max(shape_list)  # tuple compare like the original
        min_lis = min(shape_list)

        if max_lis == min_lis:
            bdry_vert_ed_inds = np.vstack(edge_idx_lists).astype(int)       # (B, k)
            bdry_vert_edges   = bdry_stack[bdry_vert_ed_inds].astype(int)   # (B, k, 2) int
        else:
            k_max = int(max_lis[0])
            padded_idx_rows = []
            for rows, shp in zip(edge_idx_lists, shape_list):
                k = int(shp[0])
                if k < k_max:
                    rows = np.append(rows, [0] * (k_max - k))
                padded_idx_rows.append(rows)
            bdry_vert_ed_inds = np.vstack(padded_idx_rows).astype(int)      # (B, k_max)
            bdry_vert_edges = bdry_stack[bdry_vert_ed_inds].astype(float)   # becomes float per original
            # zero-out padded tail as 0.0
            for i, shp in enumerate(shape_list):
                k = int(shp[0])
                pad_len = k_max - k
                if pad_len > 0:
                    bdry_vert_edges[i, -pad_len:, :] = 0.0
    else:
        bdry_verts = None
        bdry_stack = None
        bdry_vert_ed_inds = None
        bdry_vert_edges = None

    # --- Orientation signs per slot (matches first via Acc test) ---
    E = u_edge_inds.shape[0]
    coords_np = [np.zeros((E, 1), dtype=float) for _ in range(4)]
    for index, edge in enumerate(u_edge_inds):
        a, b = int(edge[0]), int(edge[1])
        for i in range(4):
            face_id = stack[index, i]
            if face_id == -1:
                continue
            tri = M1[face_id]
            v0_idx = int(np.where(tri == a)[0][0])
            v1_idx = int(np.where(tri == b)[0][0])
            test = [v0_idx, v1_idx]
            coords_np[i][index, 0] = 1.0 if test in Acc else -1.0

    # --- Convert everything to JAX at the end ---
    # Dtypes:
    # - ints -> int32
    # - floats -> float32
    j_stack  = jnp.asarray(stack, dtype=jnp.int32)
    j_edges  = jnp.asarray(u_edge_inds, dtype=jnp.int32)
    j_lens   = jnp.asarray(lens)  # bool preserved
    j_coords = [jnp.asarray(c, dtype=jnp.float32) for c in coords_np]

    j_edges_for_verts_inds = jnp.asarray(np.array(edges_for_verts_inds), dtype=jnp.int32)
    j_max_ = jnp.asarray(max_, dtype=jnp.int32)

    if bdry_verts is None:
        j_bdry_verts = None
        j_bdry_vert_edges = None
    else:
        j_bdry_verts = jnp.asarray(bdry_verts, dtype=jnp.int32)
        # Preserve the original quirk: dtype int if uniform degree, else float with zero padding.
        if bdry_vert_edges.dtype.kind == 'f':
            j_bdry_vert_edges = jnp.asarray(bdry_vert_edges, dtype=jnp.float32)
        else:
            j_bdry_vert_edges = jnp.asarray(bdry_vert_edges, dtype=jnp.int32)

    return {
        'stack': j_stack,                                  # (E,4) int32
        'edges_for_verts_inds': j_edges_for_verts_inds,    # empty (0,) int32
        'u_edge_inds': j_edges,                             # (E,2) int32
        'max_': j_max_,                                     # scalar int32
        'lens': j_lens,                                     # (E,) bool
        'coords': j_coords,                                 # list of 4 × (E,1) float32
        'bdry_verts': j_bdry_verts,                         # (B,) int32 or None
        'bdry_vert_edges': j_bdry_vert_edges                # (B,k,2) int32 or float32, or None
    }

# CLEANER PURE NUMPY VERSION (UNTESTED)
# def pre_comp_nc_vec_con(
#         M1,  # numpy array of shape (F, 3)
#         ex_bdry: bool
# ) -> Dict[str, Any]:
#     """
#     Takes in mesh structure and boundary boolean, extracts required info for
#     Normal Cycles tasks using pure NumPy operations, then converts to JAX.
#
#     Args:
#         M1 (array): mesh structure of input shape of size (F, 3)
#         ex_bdry (bool): True if input shape has boundary, false otherwise.
#
#     Returns:
#        dict: A dictionary of useful structural indices for NC metric (as JAX arrays).
#     """
#     # Convert to NumPy and ensure int32
#     M1 = np.asarray(M1)
#     if M1.dtype != np.int32:
#         M1 = M1.astype(np.int32)
#
#     max_ = 0
#     edges_for_verts_inds = []
#
#     # for each unique edge compute triangles attached
#     Acc = [[0, 1], [2, 0], [1, 2]]
#
#     # Extract all edges from triangles
#     e1 = np.column_stack([M1[:, 0], M1[:, 1]])
#     e2 = np.column_stack([M1[:, 1], M1[:, 2]])
#     e3 = np.column_stack([M1[:, 2], M1[:, 0]])
#
#     # Get unique edges (sorted for canonical representation)
#     all_edges = np.vstack([e1, e2, e3])
#     sorted_edges = np.sort(all_edges, axis=1)
#     u_edge_inds = np.unique(sorted_edges, axis=0).astype(np.int32)
#
#     VS = []
#     lens = []
#
#     # Build dictionary mapping edges to incident triangles
#     dic = defaultdict(list)
#
#     print("looping on triangulation")
#
#     for i in range(len(M1)):
#         e_1 = tuple(sorted((int(M1[i, 0]), int(M1[i, 1]))))
#         e_2 = tuple(sorted((int(M1[i, 1]), int(M1[i, 2]))))
#         e_3 = tuple(sorted((int(M1[i, 2]), int(M1[i, 0]))))
#
#         dic[e_1].append(i)
#         dic[e_2].append(i)
#         dic[e_3].append(i)
#
#     # Store boundary edges
#     bdry = []
#     print('collecting bdry edges')
#     print(u_edge_inds.shape[0])
#
#     for inds in range(u_edge_inds.shape[0]):
#         edge_tuple = tuple(sorted((int(u_edge_inds[inds][0]), int(u_edge_inds[inds][1]))))
#         vs = dic[edge_tuple]
#
#         if len(vs) == 1:
#             bdry.append(u_edge_inds[inds])
#             VS.append(np.array((vs[0], -1, -1, -1)))
#             lens.append(True)
#         else:
#             VS.append(np.append(np.array(vs), [-1] * (4 - len(vs))))
#             lens.append(False)
#
#     stack = np.vstack(VS).astype(np.int32)
#     lens = np.array(lens, dtype=bool)
#
#     # Handle boundary information
#     if ex_bdry and bdry:
#         bdry_stack = np.vstack([np.array(b) for b in bdry])
#         bdry_verts = np.unique(bdry_stack.reshape(-1)).astype(np.int32)
#
#         # Calculate indices where boundary vertices appear in boundary edges
#         lis = [np.where((bdry_stack == bdry_verts[i]).sum(1))[0].shape for i in range(len(bdry_verts))]
#         max_lis = max(lis)
#         min_lis = min(lis)
#
#         if max_lis == min_lis:
#             bdry_vert_ed_inds = np.vstack(
#                 [np.where((bdry_stack == bdry_verts[i]).sum(1))[0] for i in range(len(bdry_verts))]
#             )
#             bdry_vert_edges = bdry_stack[bdry_vert_ed_inds]
#         else:
#             # Handle case where vertices have different numbers of boundary edges
#             bdry_vert_ed_inds = np.vstack(
#                 [np.append(np.where((bdry_stack == bdry_verts[i]).sum(1))[0], [0] * (max_lis[0] - lis[i][0]))
#                  for i in range(len(bdry_verts))]
#             )
#             bdry_vert_edges = bdry_stack[bdry_vert_ed_inds]
#
#             for i in range(len(bdry_vert_edges)):
#                 pad_length = int(max_lis[0] - lis[i][0])
#                 if pad_length > 0:
#                     bdry_vert_edges[i, -pad_length:, :] = 0
#
#         bdry_vert_edges = bdry_vert_edges.astype(np.int32)
#         print(bdry_vert_edges)
#     else:
#         bdry_verts = None
#         bdry_stack = None
#         bdry_vert_edges = None
#
#     # Initialize coordinates array
#     coords_np = [np.zeros((stack.shape[0], 1), dtype=np.float32) for i in range(4)]
#
#     print("computing edge orientations")
#
#     for index, edge in enumerate(u_edge_inds):
#         for i in range(4):
#             if stack[index][i] == -1:
#                 continue
#
#             # Find indices of edge vertices in the triangle
#             triangle = M1[stack[index][i]]
#             v0_idx = np.where(triangle == edge[0])[0][0]
#             v1_idx = np.where(triangle == edge[1])[0][0]
#             test = [v0_idx, v1_idx]
#
#             # Check if this ordering is accepted
#             if test in Acc:
#                 coords_np[i][index] = 1.0
#             else:
#                 coords_np[i][index] = -1.0
#
#     # Convert all outputs to JAX arrays
#     return {
#         'stack': jnp.array(stack),
#         'edges_for_verts_inds': jnp.array(edges_for_verts_inds, dtype=jnp.int32),
#         'u_edge_inds': jnp.array(u_edge_inds),
#         'max_': jnp.array(max_, dtype=jnp.int32),
#         'lens': jnp.array(lens),
#         'coords': [jnp.array(coord) for coord in coords_np],
#         'bdry_verts': None if bdry_verts is None else jnp.array(bdry_verts),
#         'bdry_vert_edges': None if bdry_vert_edges is None else jnp.array(bdry_vert_edges),
#     }


def calc_parts_and_weights(
    v: Float[Array, "V 3"],
    f: Int[Array, "F 3"],
    mesh_struct: Dict[str, Any]
) -> Tuple[Float[Array, "E 3"], Float[Array, "E 15"]]:
    gen_parts = get_nc_parts(v, f, mesh_struct)
    weights = get_nc_weights(gen_parts)
    return gen_parts['cs'], weights

def newell_normal_mesh_jax(
    vertices: Float[Array, "V 3"],
    faces: Int[Array, "F 3"]
) -> Float[Array, "F 3"]:
    """
    Calculate normals for triangular faces in a mesh using Newell's method, implemented in JAX.

    Parameters:
    vertices -- JAX array of shape (num_vertices, 3) containing vertex coordinates
    faces -- JAX array of shape (num_faces, 3) containing vertex indices for each triangular face

    Returns:
    face_normals -- JAX array of shape (num_faces, 3) containing normalized normal vectors
    """
    # Gather the vertices for each face
    # This creates a (num_faces, 3, 3) array where each face has 3 vertices with xyz coordinates
    face_vertices = vertices[faces]

    # Define a function to compute the normal for a single triangle
    def newell_normal_single(tri_vertices):
        # Get current and next vertices using roll
        current = tri_vertices
        next_vertex = jnp.roll(tri_vertices, -1, axis=0)

        # Calculate differences and sums
        diff_y = current[:, 1] - next_vertex[:, 1]
        diff_z = current[:, 2] - next_vertex[:, 2]
        diff_x = current[:, 0] - next_vertex[:, 0]

        sum_z = current[:, 2] + next_vertex[:, 2]
        sum_x = current[:, 0] + next_vertex[:, 0]
        sum_y = current[:, 1] + next_vertex[:, 1]

        # Compute normal components
        normal_x = jnp.sum(diff_y * sum_z)
        normal_y = jnp.sum(diff_z * sum_x)
        normal_z = jnp.sum(diff_x * sum_y)

        # Combine into normal vector and scale by 0.5
        normal = jnp.array([normal_x, normal_y, normal_z]) * 0.5

        return safe_normalize(normal)

        # Gradient-safe normalization
        # length = jnp.linalg.norm(normal)
        # epsilon = 1e-10
        # safe_length = jnp.maximum(length, epsilon)
        #
        # return normal / safe_length

    # Vectorize the function to process all faces
    batch_normal = jax.vmap(newell_normal_single)

    # Compute normals for all faces
    face_normals = batch_normal(face_vertices)

    return face_normals

def get_nc_parts(
    v: Float[Array, "V 3"],
    f: Int[Array, "F 3"],
    struct: Dict[str, Any]
) -> Dict[str, Float[Array, "..."]]:
    # stack, edges_for_verts_inds, u_edge_inds, max_, lens, coords, _, _ = struct

    stack = struct['stack']
    u_edge_inds = struct['u_edge_inds']
    coords = struct['coords']

    # v1, v2, v3 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    # g1, g2 = v2 - v1, v3 - v1
    # v_norms = 0.5 * jnp.cross(g1, g2)
    # v_normals = safe_normalize(v_norms)

    v_normals = newell_normal_mesh_jax(v, f)



    v_normal_mod = jnp.vstack([v_normals, jnp.zeros((1, 3))])

    v_re = sum(coords[i] * v_normal_mod[stack[:, i]] for i in range(4))

    fs = v[u_edge_inds][:, 1, :] - v[u_edge_inds][:, 0, :]
    cs = (v[u_edge_inds][:, 1, :] + v[u_edge_inds][:, 0, :]) / 2

    normals = v_re

    return {
        'fs': fs,
        'cs': cs,
        'normals': v_re,
    }



# def get_nc_weights(fs, v_re):
def get_nc_weights(S1: Dict[str, Float[Array, "..."]]) -> Float[Array, "E 15"]:
    fs = S1['fs']
    normals = S1['normals']
    normalized, norms_ = safe_normalize_with_norm(fs)
    norms_ = norms_.reshape((-1,1))
    a1 = jnp.hstack([normalized, jnp.zeros((fs.shape[0], 3))])
    a2 = jnp.hstack([jnp.zeros((fs.shape[0], 3)), normals])
    coords = coordinates(a1, a2)
    return norms_ * coords



def calc_parts_and_weights_brdy(
    v: Float[Array, "V 3"],
    f: Int[Array, "F 3"],
    mesh_struct: Dict[str, Any]
) -> Tuple[Float[Array, "... 3"], Float[Array, "... 15"]]:
    gen_parts = get_nc_parts_brdy(v, f, mesh_struct)
    weights = get_nc_weights_brdy(gen_parts)
    gen_pts = gen_parts['gen_pts_brdy']
    gen_cs = gen_parts['cs']
    centres = jnp.vstack([gen_pts, gen_cs])
    return centres, weights

def get_nc_parts_brdy(
    v: Float[Array, "V 3"],
    f: Int[Array, "F 3"],
    args: Dict[str, Any]
) -> Dict[str, Float[Array, "..."]]:

    stack = args['stack']
    u_edge_inds = args['u_edge_inds']
    coords = args['coords']
    bdry_verts = args['bdry_verts']
    bdry_vert_edges = args['bdry_vert_edges']

    # Extract boundary edge vectors
    e_for_verts = v[bdry_vert_edges[:, :, 1]] - v[bdry_vert_edges[:, :, 0]]
    # f_norm = jnp.linalg.norm(e_for_verts, axis=2)
    # f_norm_1 = jnp.where(f_norm != 0.0, f_norm, jnp.inf)

    sev_vec = safe_normalize(e_for_verts)


    sum_edges = jnp.sum(sev_vec, axis=1)

    # Vertex positions for triangles
    v1 = v[f[:, 0]]
    v2 = v[f[:, 1]]
    v3 = v[f[:, 2]]

    # Edge vectors
    g1 = v2 - v1
    g2 = v3 - v1
    g3 = v3 - v2

    # Compute face normals
    v_norms = 0.5 * jnp.cross(g1, g2)
    # v_normals = v_norms / jnp.expand_dims(jnp.linalg.norm(v_norms, axis=1), axis=-1)

    v_normals = safe_normalize(v_norms)

    # Add zero normal and stack
    v_normal_mod = jnp.vstack([v_normals, jnp.array([0.0, 0.0, 0.0])])

    # Initialize result array
    v_re = jnp.zeros((stack.shape[0], 3))


    v_re = jnp.zeros((stack.shape[0], 3))
    for i in range(4):
        v_re += coords[i] * v_normal_mod[stack[:, i]]
    v_res = v_re

    # Compute edge vectors and centers
    fs = v[u_edge_inds][:, 1, :] - v[u_edge_inds][:, 0, :]
    cs = (v[u_edge_inds][:, 1, :] + v[u_edge_inds][:, 0, :]) / 2
    normals = v_res

    return {
        'fs': fs,
        'gen_pts_brdy': v[bdry_verts],
        'cs': cs,
        'normals': normals,
        'sum_edges': sum_edges
    }


def get_nc_weights_brdy(S1: Dict[str, Float[Array, "..."]]) -> Float[Array, "... 15"]:
    """
    JAX implementation of the Embed function

    Args:
        S1: Tuple containing shape information and boundary data

    Returns:
        Embedded coordinates with boundary weights if applicable
    """
    # Compute weights with gradient-safe normalization

    fs = S1['fs']
    gen_pts_brdy = S1['gen_pts_brdy']
    cs = S1['cs']
    normals = S1['normals']
    sum_edges = S1['sum_edges']

    a1 = jnp.zeros((fs.shape[0], 6))
    normalized, norms_ = safe_normalize_with_norm(fs)

    # Instead of using .at[:, :3].set()
    a1_first_part = normalized
    a1_second_part = jnp.zeros((fs.shape[0], 3))
    a1 = jnp.concatenate([a1_first_part, a1_second_part], axis=1)

    # Initialize a2 directly
    a2_first_part = jnp.zeros((fs.shape[0], 3))
    a2_second_part = sum_norm = normals
    a2 = jnp.concatenate([a2_first_part, a2_second_part], axis=1)

    # Create fill array directly
    fill_first_part = jnp.zeros((gen_pts_brdy.shape[0], 12))
    fill = jnp.concatenate([fill_first_part, sum_edges], axis=1)

    # Return the stacked result
    return jnp.vstack([fill, norms_ * coordinates(a1, a2)])