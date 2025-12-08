import pickle as pkl
import numpy as np

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

def load_skel_model(gender='male'):

    if gender == 'male':
        skel_file = '../SKEL/models/skel_models_v1.1/skel_male.pkl'
    else:
        skel_file = '../SKEL/models/skel_models_v1.1/skel_female.pkl'

    model_obj = pkl.load(open(skel_file, 'rb'))

    kintree_table = model_obj['osim_kintree_table']
    parents = np.transpose(model_obj['osim_kintree_table'][0, :])
    parents[0] = 0
    J_regressor = model_obj['J_regressor_osim'].todense()

    template = model_obj['skin_template_v']

    return kintree_table, parents, J_regressor, template

def extract_smpl_hands(smpl_skel):

    # smpl_left_hand_bones = np.array([[23, 24], [20, 25], [25, 26], [26, 27], [20, 28], [28, 29], [29, 30], [20, 31], [31, 32], [32, 33], [20, 34], [34, 35], [35, 36]]).T
    # smpl_right_hand_bones = np.array([[21, 37], [37, 38], [38, 39], [21, 40], [40, 41], [41, 42], [21, 43], [43, 44], [44, 45], [21, 46], [46, 47], [47, 48], [21, 49], [49, 50], [50, 51]]).T

    smpl_left_hand_joints_ids = np.arange(22,37)
    smpl_right_hand_joints_ids = np.arange(37,52)

    # extrat the left hand joints
    smpl_left_hand_joints = smpl_skel['joints_positions'][smpl_left_hand_joints_ids]
    smpl_right_hand_joints = smpl_skel['joints_positions'][smpl_right_hand_joints_ids]

    return smpl_left_hand_joints, smpl_right_hand_joints



def setup_skel_skeleton(v_1, kintree_table, joints_positions, smpl_skel):

    # if gender == 'male':
    #     source = pd.read_pickle(r'data/DFAUST/SMPLH_male.pkl')
    # else:
    #     source = pd.read_pickle(r'data/DFAUST/SMPLH_female.pkl')
    #
    # v_template = source['v_template']
    # parents = np.transpose(source['kintree_table'][0, :])
    # parents[0] = 0
    # kintree_table = source['kintree_table']
    # J_regressor = source['J_regressor'].todense()
    #
    # parents = list(kintree_table[0].tolist())

    smpl_left_hand_joints, smpl_right_hand_joints = extract_smpl_hands(smpl_skel)


    # ALTER FINGER JOINTS TO MAKE THEM BETTER

    smpl_left_hand_bones = np.array([[23, 24], [24, 25], [25, 26], [23, 27], [27, 28], [28, 29], [23, 30], [30, 31], [31, 32], [23, 33], [33, 34], [34, 35], [23, 36], [36, 37], [37, 38]]).T

    smpl_right_hand_bones = np.array(
        [[18, 39], [39, 40], [40, 41], [18, 42], [42, 43], [43, 44], [18, 45], [45, 46], [46, 47], [18, 48], [48, 49],
         [49, 50], [18, 51], [51, 52], [52, 53]]).T



    joints_positions = np.vstack([
        joints_positions,
        smpl_left_hand_joints,
        smpl_right_hand_joints,
        v_1[411, :]  # head
    ])

    head_bone = np.array([[13, 54]]).T

    kintree_table = np.c_[(
        kintree_table,
        smpl_left_hand_bones,
        smpl_right_hand_bones,
        head_bone
    )]

    # kintree = np.array(kintree)
    bones = kintree_table[:, 1:].T


    bone_edges = bones
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
