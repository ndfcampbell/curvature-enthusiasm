import numpy as np
import pytetwild
import pyvista as pv
from tqdm import tqdm


def compute_tet_volumes(vertices, tets):
    """
    Compute volumes for all tetrahedra in one vectorized call.
    vertices: (n,3) array.
    tets: (m,4) array of indices.
    """
    tet_vertices = vertices[tets]  # shape: (m, 4, 3)
    v0 = tet_vertices[:, 1] - tet_vertices[:, 0]
    v1 = tet_vertices[:, 2] - tet_vertices[:, 0]
    v2 = tet_vertices[:, 3] - tet_vertices[:, 0]
    volumes = np.abs(np.einsum('ij,ij->i', np.cross(v0, v1), v2)) / 6.0
    return volumes


def adaptive_refine_tet_mesh(vertices, tets, max_volume_ratio=1.25, max_edge_ratio=2.0, max_iterations=10):
    """
    Refine the tetrahedral mesh adaptively.
    A tetrahedron is flagged for refinement if its volume is too large compared
    to the current average OR if its longest-to-shortest edge ratio is too high.
    Vectorized computations are used for volumes and edge lengths, and only the
    flagged tetrahedra are processed in a Python loop.
    """
    # Use a Python list for vertices to allow fast dynamic appending.
    vertices_list = vertices.tolist()
    tets = np.array(tets)

    # Mapping from an edge's index in the list of 6 edges to its vertex pair
    # (local indices inside a tetrahedron). Order is:
    # (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    mapping = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])

    for iteration in tqdm(range(max_iterations), desc="Adaptive Refinement Iterations"):
        # Convert current vertices to an array for vectorized computation.
        vertices_array = np.array(vertices_list)

        # Compute volumes for all tetrahedra.
        volumes = compute_tet_volumes(vertices_array, tets)
        avg_volume = volumes.mean() if volumes.size > 0 else 0

        # Vectorized computation of edge lengths:
        v0 = vertices_array[tets[:, 0]]
        v1 = vertices_array[tets[:, 1]]
        v2 = vertices_array[tets[:, 2]]
        v3 = vertices_array[tets[:, 3]]
        e01 = np.linalg.norm(v0 - v1, axis=1)
        e02 = np.linalg.norm(v0 - v2, axis=1)
        e03 = np.linalg.norm(v0 - v3, axis=1)
        e12 = np.linalg.norm(v1 - v2, axis=1)
        e13 = np.linalg.norm(v1 - v3, axis=1)
        e23 = np.linalg.norm(v2 - v3, axis=1)
        edges = np.stack([e01, e02, e03, e12, e13, e23], axis=1)
        longest_edge_lengths = np.max(edges, axis=1)
        min_edge_lengths = np.min(edges, axis=1)
        edge_ratios = longest_edge_lengths / (min_edge_lengths + 1e-12)  # avoid div-by-zero

        # Identify tetrahedra that do not meet quality.
        volume_condition = volumes > avg_volume * max_volume_ratio
        edge_condition = edge_ratios > max_edge_ratio
        flagged = volume_condition | edge_condition

        # If no tetrahedra are flagged, stop iterating.
        if not np.any(flagged):
            tqdm.write("No more refinement needed, stopping early.")
            break

        flagged_indices = np.where(flagged)[0]
        # We'll build a new list of tetrahedra for the next iteration.
        new_tets_list = []
        # Add all tetrahedra that pass the quality check unchanged.
        unflagged_indices = np.where(~flagged)[0]
        if unflagged_indices.size > 0:
            new_tets_list.extend(tets[unflagged_indices].tolist())

        # For the flagged tetrahedra, determine which edge to split along.
        flagged_edges = edges[flagged_indices]
        # Get the index (0-5) of the longest edge for each flagged tet.
        longest_edge_idx = np.argmax(flagged_edges, axis=1)

        # Process each flagged tetrahedron.
        for idx, local_longest in zip(flagged_indices, longest_edge_idx):
            tet = tets[idx]
            # Determine which pair of vertices form the longest edge.
            edge_pair = mapping[local_longest]
            v_i = tet[edge_pair[0]]
            v_j = tet[edge_pair[1]]
            # Compute the midpoint.
            midpoint = (vertices_array[v_i] + vertices_array[v_j]) / 2.0
            new_vertex_index = len(vertices_list)
            vertices_list.append(midpoint.tolist())
            # Identify the remaining two vertices (local indices not in edge_pair).
            remaining = [i for i in range(4) if i not in edge_pair]
            # Split into two new tetrahedra:
            # First tet uses (new vertex, v_i, remaining[0], remaining[1])
            # Second tet uses (new vertex, v_j, remaining[0], remaining[1])
            new_tet1 = [new_vertex_index, v_i, tet[remaining[0]], tet[remaining[1]]]
            new_tet2 = [new_vertex_index, v_j, tet[remaining[0]], tet[remaining[1]]]
            new_tets_list.extend([new_tet1, new_tet2])

        # Update the tetrahedra array with the new set.
        tets = np.array(new_tets_list)
        tqdm.write(f"Iteration {iteration}: refined {len(flagged_indices)} tets, total tets now: {len(tets)}")

    return np.array(vertices_list), tets


def create_tet_template(V, F):
    """
    Generate a tetrahedral mesh from the surface mesh (V, F),
    refine it adaptively for better quality, and compute the tetrahedron centers.
    Optionally visualize using polyscope.
    """
    # Convert the surface mesh data to PyVista format.
    surface_mesh = pv.PolyData(V, np.hstack([np.full((F.shape[0], 1), 3), F]).astype(np.int64))

    # Set tetrahedralization parameters.
    length_ratio = 0.05  # Adjust tetrahedron quality.
    tetrahedral_mesh = pytetwild.tetrahedralize_pv(surface_mesh, edge_length_fac=length_ratio, optimize=True)

    TV = np.array(tetrahedral_mesh.points)
    TT = np.array(tetrahedral_mesh.cells_dict[pv.CellType.TETRA])

    # Apply adaptive refinement.
    TV, TT = adaptive_refine_tet_mesh(TV, TT, max_volume_ratio=1.25, max_edge_ratio=1.25, max_iterations=2)

    # Compute the centers of the tetrahedra.
    tetra_centers = np.mean(TV[TT], axis=1)

    return TV, TT, tetra_centers

# Example usage:
# V, F = load_your_mesh("path/to/mesh.obj")
# TV, TT, centers = create_tet_template(V, F, visualize=True)


