import numpy as np
from .misc import default_floating_dtype

def find_connected_edges(kinematic_tree):
    connected_edges = []

    for edge1 in kinematic_tree:
        for edge2 in kinematic_tree:
            # Check if the head of edge1 is the tail of edge2 and they are not the same edge
            if edge1[1] == edge2[0] and edge1[0] != edge2[1]:
                connected_edges.append([edge1[0], edge1[1], edge2[0], edge2[1]])

    return np.array(connected_edges)

def perp_stark(u, dtype=None):

    dtype = default_floating_dtype() if dtype is None else dtype

    # Convert the input vector to an absolute value version for comparison
    a = np.abs(u)

    uyx = a[0] < a[1]
    uzx = a[0] < a[2]
    uzy = a[1] < a[2]

    # Apply logic to determine the mask for each component
    xm = uyx & uzx
    ym = (~xm) & uzy
    zm = ~(xm | ym)

    # Create the vector for the cross product
    mask_vector = np.array([xm, ym, zm], dtype=dtype)

    # Calculate the cross product to find the perpendicular vector
    v = np.cross(u, mask_vector)

    return v

def calculate_bone_lengths(joint_positions, edges):
    """Calculate and return an array of bone lengths based on a matrix of joint positions and a list of edges."""
    lengths = []
    for parent_idx, child_idx in edges:
        vector = joint_positions[child_idx] - joint_positions[parent_idx]
        length = np.linalg.norm(vector)
        lengths.append(length)
    return np.array(lengths)

def calculate_bone_vectors(joint_positions, edges):
    """Calculate bone vectors using a matrix of joint positions and an (n, 2) matrix of edges."""
    bone_vectors = {}
    for i in range(edges.shape[0]):
        parent_idx, child_idx = edges[i]
        bone_vectors[(parent_idx, child_idx)] = joint_positions[child_idx] - joint_positions[parent_idx]
    return bone_vectors


def find_parent_edges(edges):
    # Mapping from a vertex ID to the list of edges that input into this vertex
    to_vertex_to_edges = {}
    for edge_index, (from_vertex, to_vertex) in enumerate(edges):
        if to_vertex in to_vertex_to_edges:
            to_vertex_to_edges[to_vertex].append(edge_index)
        else:
            to_vertex_to_edges[to_vertex] = [edge_index]

    parent_edges = [-1] * len(edges)  # Initialize with -1 indicating no parent edge

    for edge_index, (from_vertex, to_vertex) in enumerate(edges):
        if from_vertex in to_vertex_to_edges:
            # We choose the last added edge as the parent for simplicity; could be adjusted if needed
            parent_edges[edge_index] = to_vertex_to_edges[from_vertex][-1]

    return parent_edges

def calc_local_axes(bone, u_axes_dir=None):
    # Calculate the local axes and transformation matrix
    start_point = bone[0]
    end_point = bone[1]

    bone_vec = end_point - start_point
    bone_vec_axis_1 = bone_vec / np.linalg.norm(bone_vec)

    if u_axes_dir is None:
        bone_vec_axis_2 = perp_stark(bone_vec_axis_1)
    else:
        bone_vec_axis_2 = np.cross(bone_vec_axis_1, u_axes_dir)

    bone_vec_axis_2 /= np.linalg.norm(bone_vec_axis_2)

    bone_vec_axis_3 = np.cross(bone_vec_axis_1, bone_vec_axis_2)
    bone_vec_axis_3 /= np.linalg.norm(bone_vec_axis_3)

    return bone_vec_axis_1, bone_vec_axis_2, bone_vec_axis_3


def axes_to_quaternion(u_axes, v_axes):
    # Compute the rotation matrix R
    R = np.dot(v_axes, u_axes.T)

    # Calculate the trace of the rotation matrix
    trace = np.trace(R)

    if trace > -1:
        qw = 0.5 * np.sqrt(1 + trace)
        qx = (R[2, 1] - R[1, 2]) / (4 * qw)
        qy = (R[0, 2] - R[2, 0]) / (4 * qw)
        qz = (R[1, 0] - R[0, 1]) / (4 * qw)
    else:
        # Trace is close to -1, handle near 180 degree rotations
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s

    quaternion = np.array([qw, qx, qy, qz])
    quaternion /= np.linalg.norm(quaternion)

    if qw < 0:
        quaternion = -quaternion

    return quaternion
