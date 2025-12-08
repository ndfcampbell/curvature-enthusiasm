import numpy as np
from .ik_skeleton_funcs import find_parent_edges, calculate_bone_lengths, calculate_bone_vectors, calc_local_axes, \
    axes_to_quaternion

def setup_tgf_skeleton(joints_positions, bone_edges):

    link_lengths = np.array(calculate_bone_lengths(joints_positions, bone_edges))
    bone_vectors = calculate_bone_vectors(joints_positions, bone_edges)

    base_position = joints_positions[0]
    base_rotation = np.array([1.0, 0.0, 0.0, 0.0])
    num_bones = bone_edges.shape[0]

    list_of_bone_edges = [tuple(row) for row in bone_edges]
    print("# num_joints", np.max(bone_edges) + 1)
    print("# num_bones", num_bones)
    print("edges", list_of_bone_edges)
    parent_edges = find_parent_edges(list_of_bone_edges)
    parent_edges = np.array(parent_edges)
    # parent_edges += 1

    list_of_rotations = []
    list_of_local_axes = []
    list_of_rotations.append(base_rotation)
    unit_axes = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    list_of_local_axes.append(unit_axes)

    for i, parent_edge in enumerate(parent_edges):

        parent_axes = list_of_local_axes[parent_edge + 1]
        ux, uy, uz = calc_local_axes(joints_positions[bone_edges[i]])

        u_axes = np.array([ux, uy, uz])

        # u_axes = np.array([ux, uy, uz])
        quaternion = axes_to_quaternion(parent_axes.T, u_axes.T)
        # comparison = test_quaternion_transform(parent_axes.T, u_axes.T, quaternion)
        list_of_local_axes.append(u_axes)
        list_of_rotations.append(quaternion)

        joint_rotations = np.array(list_of_rotations[1:])
        init_local_axes = np.array(list_of_local_axes[1:])

    skeleton_config = {
        'bone_edges': bone_edges,
        'joint_rotations': joint_rotations,
        'init_local_axes': init_local_axes,
        'root_position': base_position,
        'root_rotation': base_rotation,
        'joints_positions': joints_positions,
        'link_lengths': link_lengths,
        'init_local_axes': init_local_axes,
        'parent_edges': parent_edges}

    return skeleton_config


def load_skeleton_tgf(path):

    with open(path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    if '#' not in lines:
        raise ValueError("Missing '#' separator between nodes and edges.")

    first_sep = lines.index('#')
    second_sep = lines[first_sep + 1:].index('#') + first_sep + 1 if '#' in lines[first_sep + 1:] else None

    node_lines = lines[:first_sep]
    edge_lines = lines[first_sep + 1: second_sep if second_sep is not None else None]

    # Map from file node ID to 0-based index
    node_ids = []
    node_coords = {}
    for line in node_lines:
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f"Malformed node line: {line}")
        node_id = parts[0]
        x, y, z = map(float, parts[1:])
        node_ids.append(node_id)
        node_coords[node_id] = (x, y, z)

    id_map = {node_id: idx for idx, node_id in enumerate(node_ids)}
    positions = np.array([node_coords[nid] for nid in node_ids], dtype=np.float32)

    edges = np.array([
        [id_map[a], id_map[b]]
        for line in edge_lines
        for a, b in [line.split()]
    ], dtype=np.int32)

    return positions, edges

def save_skeleton_tgf(joints_positions, bone_edges, filename):
    """
    Save skeleton data to a TGF file.

    Parameters:
    -----------
    joints_positions : np.ndarray
        Array of joint positions with shape (n, 3)
    bone_edges : np.ndarray
        Array of bone connections with shape (m, 2)
    filename : str
        Path to the output TGF file
    """
    joints_positions = np.asarray(joints_positions)
    bone_edges = np.asarray(bone_edges)

    if joints_positions.ndim != 2 or joints_positions.shape[1] != 3:
        raise ValueError("joints_positions should be a 2D array with shape (n, 3)")
    if bone_edges.ndim != 2 or bone_edges.shape[1] != 2:
        raise ValueError("bone_edges should be a 2D array with shape (m, 2)")

    with open(filename, 'w') as f:
        # Write node lines (using 1-based indexing to match common TGF convention)
        for i, pos in enumerate(joints_positions):
            f.write(f"{i} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")

        # Write separator
        f.write("#\n")

        # Write edge lines
        for edge in bone_edges:
            f.write(f"{edge[0]} {edge[1]}\n")