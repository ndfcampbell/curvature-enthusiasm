import numpy as np
import pandas as pd

from curvature_enthusiasm.utils.ik_skeleton_funcs import find_parent_edges, calculate_bone_lengths, calculate_bone_vectors, calc_local_axes, \
    axes_to_quaternion

def determine_gender(id):
    male = [50002, 50007, 50009, 50026, 50027]
    female = [50004, 50020, 50021, 50022, 50025]

    if id in male:
        return 'male'
    elif id in female:
        return 'female'
    else:
        return 'unknown'

def load_smpl_model(v_1, gender='male'):

    if gender == 'male':
        model_obj = pd.read_pickle(r'data/DFAUST/SMPLH_male.pkl')
    else:
        model_obj = pd.read_pickle(r'data/DFAUST/SMPLH_female.pkl')
    #
    # v_template = model_obj['v_template']
    kintree_table = model_obj['kintree_table']
    parents = np.transpose(model_obj['kintree_table'][0, :])
    parents[0] = 0
    J_regressor = model_obj['J_regressor'].todense()

    joints_positions = np.matmul(np.asarray(J_regressor), v_1)

    return joints_positions

def setup_faust_smpl_skeleton(joints_positions, gender='male'):

    if gender == 'male':
        source = pd.read_pickle(r'data/DFAUST/SMPLH_male.pkl')
    else:
        source = pd.read_pickle(r'data/DFAUST/SMPLH_female.pkl')

    # v_template = source['v_template']
    parents = np.transpose(source['kintree_table'][0, :])
    parents[0] = 0
    kintree_table = source['kintree_table']
    J_regressor = source['J_regressor'].todense()

    parents = list(kintree_table[0].tolist())

    # TEMP FUDGE
    # v_1 = v_template

    # edges = []
    # for i in range(1, len(parents)):
    #     edges.append([parents[i], i])
    #
    # bone_edges = np.array(edges)

    # joints_positions = np.matmul(np.asarray(J_regressor), v_1)

    # joints_positions = np.vstack([
    #     joints_positions,
    #     v_1[6206, :],  # right thumb
    #     v_1[5782, :],  #
    #     v_1[5905, :],  #
    #     v_1[6016, :],  #
    #     v_1[6133, :],  #
    #     v_1[2746, :],  # left thumb
    #     v_1[2319, :],  #
    #     v_1[2445, :],  #
    #     v_1[2556, :],  #
    #     v_1[2673, :],  #
    #     v_1[411, :]  # head
    # ])
    #
    # extra_bones = np.array([[51, 52], [39, 53], [42, 54], [48, 55], [45, 56],
    #                             [36, 57], [24, 58], [27, 59], [33, 60], [30, 61], [15, 62]]).T

    # ALTER FINGER JOINTS TO MAKE THEM BETTER

    # LEFT HAND
    # joints_positions[36, :] = v_1[2746, :]
    # joints_positions[24, :] = v_1[2319, :]
    # joints_positions[27, :] = v_1[2445, :]
    # joints_positions[33, :] = v_1[2556, :]
    # joints_positions[30, :] = v_1[2673, :]
    #
    # # RIGHT HAND
    # joints_positions[51, :] = v_1[6206, :]
    # joints_positions[39, :] = v_1[5782, :]
    # joints_positions[42, :] = v_1[5905, :]
    # joints_positions[48, :] = v_1[6016, :]
    # joints_positions[45, :] = v_1[6133, :]
    #
    # joints_positions = np.vstack([
    #     joints_positions,
    #     v_1[411, :]  # head
    # ])

    extra_bones = np.array([[15, 52]]).T

    kintree_table = np.c_[(
        kintree_table,
        extra_bones
    )]

    # kintree = np.array(kintree)
    bones = kintree_table[:, 1:].T


    bone_edges = np.array(bones).astype(np.int32)
    num_bones = bone_edges.shape[0]

    # Number of bones
    # num_bones = bone_edges.shape[0]
    # bone_positions_1 = []

    # Calculate bone vectors
    link_lengths = np.array(calculate_bone_lengths(joints_positions, bone_edges))
    bone_vectors = calculate_bone_vectors(joints_positions, bone_edges)

    # Calculate quaternions and convert to a matrix format
    # init_rotation = calculate_quaternions_as_matrix(bone_vectors, num_bones)

    base_position = joints_positions[0]
    base_rotation = np.array([1.0, 0.0, 0.0, 0.0])


    list_of_bone_edges = [tuple(row) for row in bone_edges]
    print("# num_joints", np.max(bone_edges)+1)
    print("# num_bones", num_bones)
    print(f"edges :", [(int(x), int(y)) for x, y in bone_edges])
    parent_edges = find_parent_edges(list_of_bone_edges)
    parent_edges = np.array(parent_edges)
    print("parent edges",parent_edges)
    # parent_edges += 1

    list_of_rotations = []
    list_of_local_axes = []
    list_of_rotations.append(base_rotation)
    unit_axes = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    list_of_local_axes.append(unit_axes)

    for i, parent_edge in enumerate(parent_edges):

        parent_axes = list_of_local_axes[parent_edge+1]
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

def setup_smpl_skeleton_no_hands(joints_positions):


    # SCAPE_R
    bones = np.array([(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12), (9, 13), (9, 14), (12, 15), (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21), (20, 22), (21, 23), (15, 24)])

    # skel_no_hands_edges = np.array([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 6), (6, 7), (7, 8), (8, 9), (9 ,10), (0, 11), (11, 12), (12, 13), (12, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (12, 20), (20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (13,26)])


    bone_edges = np.array(bones).astype(np.int32)
    num_bones = bone_edges.shape[0]

    # Number of bones
    # num_bones = bone_edges.shape[0]
    # bone_positions_1 = []

    # Calculate bone vectors
    link_lengths = np.array(calculate_bone_lengths(joints_positions, bone_edges))
    bone_vectors = calculate_bone_vectors(joints_positions, bone_edges)

    base_position = joints_positions[0]
    base_rotation = np.array([1.0, 0.0, 0.0, 0.0])

    list_of_bone_edges = [tuple(row) for row in bone_edges]
    print("# num_joints", np.max(bone_edges) + 1)
    print("# num_bones", num_bones)
    print(f"edges :", [(int(x), int(y)) for x, y in bone_edges])
    parent_edges = find_parent_edges(list_of_bone_edges)
    parent_edges = np.array(parent_edges)
    print("parent edges", parent_edges)
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