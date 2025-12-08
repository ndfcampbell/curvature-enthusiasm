import numpy as np
import pandas as pd

from curvature_enthusiasm.utils.ik_skeleton_funcs import find_parent_edges, calculate_bone_lengths, calculate_bone_vectors, calc_local_axes, \
    axes_to_quaternion


# def load_mano_model(v_1):
#     model_obj = pd.read_pickle(r'data/MANO/MANO_RIGHT.pkl')
#     kintree_table = model_obj['kintree_table']
#     parents = list(kintree_table[0].tolist())
#     edges = []
#     for i in range(1, len(parents)):
#         edges.append([parents[i], i])
#     J_regressor = model_obj['J_regressor'].todense()
#     joints_positions = np.matmul(np.asarray(J_regressor), v_1)
#     return joints_positions


def setup_mano_skeleton(v_1, add_extra_bones=True):

    model_obj = pd.read_pickle(r'data/MANO/MANO_RIGHT.pkl')
    kintree_table = model_obj['kintree_table']
    parents = list(kintree_table[0].tolist())

    edges = []
    for i in range(1, len(parents)):
        edges.append([parents[i], i])

    J_regressor = model_obj['J_regressor'].todense()
    joints_positions = np.matmul(np.asarray(J_regressor), v_1)

    bone_edges = np.array(edges)
    b_axes_dir = np.load('data/MANO/b_axes_dir.npy')[0]
    u_axes_dir = np.load('data/MANO/u_axes_dir.npy')[0]
    l_axes_dir = np.load('data/MANO/l_axes_dir.npy')[0]

    if add_extra_bones:
        extra_bones = np.array([[9, 16], [12, 17], [6, 18], [3, 19], [15, 20]])
        bone_edges = np.concatenate([bone_edges, extra_bones], axis=0)

        joints_positions = np.vstack([joints_positions,
                                   v_1[672, :],
                                   v_1[555, :],
                                   v_1[444, :],
                                   v_1[320, :],
                                   v_1[745, :],
                                   ])

        link_lengths = np.array(calculate_bone_lengths(joints_positions, bone_edges))
        # Calculate bone vectors
        bone_vectors = calculate_bone_vectors(joints_positions, bone_edges)

    num_bones = bone_edges.shape[0]

    base_position = joints_positions[0]
    base_rotation = np.array([1.0, 0.0, 0.0, 0.0])

    list_of_bone_edges = [tuple(row) for row in bone_edges]
    print("# num_joints", np.max(bone_edges)+1)
    print("# num_bones", num_bones)
    print("edges", list_of_bone_edges)
    parent_edges = find_parent_edges(list_of_bone_edges)
    parent_edges = np.array(parent_edges)

    list_of_rotations = []
    list_of_local_axes = []
    list_of_rotations.append(base_rotation)
    unit_axes = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    list_of_local_axes.append(unit_axes)

    for i, parent_edge in enumerate(parent_edges):

        parent_axes = list_of_local_axes[parent_edge + 1]

        if i < 16:
            ux, uy, uz = calc_local_axes(joints_positions[bone_edges[i]], u_axes_dir[i])
        else:
            ux, uy, uz = calc_local_axes(joints_positions[bone_edges[i]], u_axes_dir[parent_edge])

        u_axes = np.array([ux, uy, uz])
        quaternion = axes_to_quaternion(parent_axes.T, u_axes.T)
        # comparison = test_quaternion_transform(parent_axes.T, u_axes.T, quaternion)
        list_of_local_axes.append(u_axes)
        list_of_rotations.append(quaternion)

    joint_rotations = np.array(list_of_rotations[1:])
    init_local_axes = np.array(list_of_local_axes[1:])

    return {'bone_edges': bone_edges,
            'joint_rotations': joint_rotations,
            'init_local_axes': init_local_axes,
            'root_position': base_position,
            'root_rotation': base_rotation,
            'joints_positions': joints_positions,
            'link_lengths': link_lengths,
            'parent_edges': parent_edges}
