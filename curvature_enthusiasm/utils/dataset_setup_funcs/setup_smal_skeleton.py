import numpy as np
import pickle as pkl

from curvature_enthusiasm.utils import find_parent_edges, calculate_bone_lengths, calculate_bone_vectors, calc_local_axes, \
    axes_to_quaternion

# POSE_1 = "0_0-38"
# POSE_2 = "5_0-38"
#
# N_T_INDUCING_POINTS = 5
# NUM_IP_PER_AXIS = 5
#
# LOAD_MODEL = True
#
# regular_samples = jnp.reshape(jnp.linspace(0.0, 1.0, n_sample_points_per_bone + 2, endpoint=True), (-1, 1))
# regular_samples = regular_samples[1:n_sample_points_per_bone + 1]

bone_positions_1 = []
bone_positions_2 = []


# def construct_transformation_matrix(x2, y2, z2, O2, scale_factors):
#     """
#     Construct a 4x4 transformation matrix using homogeneous coordinates, including scaling.
#
#     Parameters:
#     - x1, y1, z1: np.array, unit vectors of Frame 1 axes.
#     - x2, y2, z2: np.array, unit vectors of Frame 2 axes.
#     - O1, O2: np.array, origins of Frame 1 and Frame 2.
#     - scale_factors: tuple or np.array, scaling factors along x, y, and z axes.
#
#     Returns:
#     - transformation_matrix: np.array, a 4x4 transformation matrix.
#     """
#     # Construct the rotation matrix
#     rotation_matrix = np.array([x2, y2, z2]).T
#
#     # Incorporate scaling into the rotation matrix
#     scaled_rotation_matrix = rotation_matrix * scale_factors
#
#     # Create the transformation matrix
#     transformation_matrix = np.eye(4)
#     transformation_matrix[:3, :3] = scaled_rotation_matrix
#     transformation_matrix[:3, 3] = O2
#
#     return transformation_matrix



# def shortest_distance_to_line(p1, p2, p3):
#
#     distances = []
#     for P in p3:
#         distance = np.linalg.norm(np.abs(np.cross(p2-p1, p1-P)) / np.linalg.norm(p2-p1))
#         distances.append(distance)
#
#     return distances


# def subdivide(uv, uf, levels=1):
#     print('Subdivide input meshes...')
#     for n in range(levels):
#         uv, uf = igl.upsample(uv, uf)
#     return uv, uf


def setup_smal_skeleton(v_1, ADD_EXTRA_BONES=False):

    with open('data/SMAL/my_smpl_00781_4_all.pkl', 'rb') as f:
        source = pkl.load(f, encoding='latin1')
    # J_regressor = dd['J_regressor']
    J_regressor = source['J_regressor'].todense()

    parents = np.transpose(source['kintree_table'][0, :])
    parents[0] = 0
    kintree_table = source['kintree_table'].astype(np.int32)

    # v_1, f_1 = igl.read_triangle_mesh("data/SMAL/" + POSE_1 + ".obj")
    # v_2, f_2 = igl.read_triangle_mesh("data/SMAL/"+ POSE_2 + ".obj")

    # Jx_1 = v_1[:, 0] @ J_regressor.T
    # Jy_1 = v_1[:, 1] @ J_regressor.T
    # Jz_1 = v_1[:, 2] @ J_regressor.T
    # joints_1 = np.stack([Jx_1, Jy_1, Jz_1], axis=1)

    joints_positions = np.matmul(np.asarray(J_regressor), v_1)

    # joints_positions[0] -= np.array([0.0, 0.0, 5.0])
    # joints_positions[16] -= np.array([0.0, 0.0, 5.0])
    # joints_positions[20] -= np.array([0.0, 0.0, 5.0])
    # joints_positions[24] -= np.array([0.0, 0.0, 5.0])

    if ADD_EXTRA_BONES:
        joints_positions = np.vstack([
            joints_positions,
            v_1[1863, :],  # end_of_nose
            v_1[26,:],  # chin
            v_1[2124,:],  # right ear tip
            v_1[150,:],  # left ear tip
            v_1[3055,:],  # left eye
            v_1[1097,:],  # right eye
        v_1[2592], # front right paw
        v_1[628], # front left paw
        v_1[2819], # rear right paw
        v_1[855] #  rear left paw
        ])

        # move joint 31 to v_template[28]
        joints_positions[31] = v_1[28]

        kintree_table = np.c_[(
            kintree_table,
            np.array([33, 38]),
            np.array([34, 37]),
            np.array([16, 35]),
            np.array([16, 39]),
            np.array([16, 40]),
            np.array([32, 36]),
            np.array([20, 44]),
            np.array([24, 43]),
            np.array([14, 41]),
            np.array([10, 42]),
        )]

    for i in range(kintree_table.shape[1]):
        if kintree_table[0, i] == 0:
            kintree_table[0, i] = 1



    # kintree_table = np.c_[(
    #     kintree_table,
    #     extra_bones
    # )]

    kintree = kintree_table[:, 2:]
    bones = (kintree - 1).T
    joints_positions = joints_positions[1:]

    # TEMP FUDGE - ORIGINAL SKELETON RIDES THE SURFACE
    joints_positions[0] -= np.array([0.0, 0.0, 0.25])
    joints_positions[1] -= np.array([0.0, 0.0, 0.25])
    joints_positions[16] -= np.array([0.0, 0.0, 0.25])
    joints_positions[20] -= np.array([0.0, 0.0, 0.25])
    joints_positions[24] -= np.array([0.0, 0.0, 0.25])

    # bones = kintree_table[:, 1:].T
    bone_edges = bones
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


    # kintree = kintree[:, 2:]
    # bones = (kintree - 1).T
    # joints_1 = joints_1[1:]

    # joints_2 = joints_2[1:]


    # rescale the mesh
    # v_1, joints_1 = rescale_mesh_trimesh(v_1, f_1, joints_1)
    # v_2, joints_2 = rescale_mesh_trimesh(v_2, f_2, joints_2)


    # align.py v_2 with v_1
    # offset = np.mean(v_2[bottom_template_vert_ids] - v_1[bottom_template_vert_ids], axis=0)
    # offset = joints_2[0] - joints_1[0]
    # v_2 -= offset
    # joints_2 -= offset

    # sample points linearly along each bone

    # bones_position_1 = joints_1[kintree]
    # bones_position_2 = joints_2[kintree]

    # bone_positions_1.append(hand_joints_1)
    # bone_positions_2.append(hand_joints_2)

    # bone_axes = []
    # global_inducing_points = []
    # global_inducing_points_2 = []
    #
    #
    # transform_matrix_list = []
    # for i, bone in enumerate(bones):
    #     print('Processing bone', i)
    #
    #     start_pos = joints_1[bone[0]]
    #     end_pos = joints_1[bone[1]]
    #     bone_vec = end_pos - start_pos
    #
    #     bone_centre = start_pos + 0.5 * bone_vec
    #     bone_vec_axis_1 = bone_vec / np.linalg.norm(bone_vec)
    #     bone_vec_axis_2 = perp_stark(bone_vec_axis_1)
    #     bone_vec_axis_2 /= np.linalg.norm(bone_vec_axis_2)
    #
    #     # assert(np.abs(np.dot(bone_vec_axis_1, bone_vec_axis_2)) < 1e-6)
    #
    #     bone_vec_axis_3 = np.cross(bone_vec_axis_1, bone_vec_axis_2)
    #     bone_vec_axis_3 /= np.linalg.norm(bone_vec_axis_3)
    #
    #     # x1, y1, z1 = np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])
    #     # O1 = np.array([0, 0, 0])
    #     O2 = bone_centre
    #     scale_factor = np.linalg.norm(0.5 * bone_vec)
    #     transformation_matrix = construct_transformation_matrix(bone_vec_axis_1, bone_vec_axis_2, bone_vec_axis_3, O2, scale_factor)
    #
    #     if i == 0:
    #         print('Bone Centre:', bone_centre)
    #         print('Bone Vec :', bone_vec)
    #         print('Bone Vec Axis 1:', bone_vec_axis_1)
    #         print('Bone Vec Axis 2:', bone_vec_axis_2)
    #         print('Bone Vec Axis 3:', bone_vec_axis_3)
    #         print('Scale Factor:', scale_factor)
    #         print('Transformation Matrix:', transformation_matrix)
    #
    #     transform_matrix_list.append(transformation_matrix)
    #     # transformation_matrix = construct_transformation_matrix(x1, y1, z1, bone_vec_axis_1, bone_vec_axis_2, bone_vec_axis_3, O1, O2, scale_factor)
    #
    #     # transformed_point = convert_points_homogeneous(grid_ip, transformation_matrix)
    #     # global_inducing_points.append(transformed_point)
    #
    #     # Inverse transformation matrix for converting from local to global
    #     # inverse_transformation_matrix = np.linalg.inv(transformation_matrix)
    #     # global_inducing_points.append(convert_points(inverse_transformation_matrix, grid_ip))
    #
    #     # temp_a = convert_points(inverse_transformation_matrix, grid_ip)
    #     # temp_b = bone_centre+bone_vec_axis_1
    #
    #     bone_local_axis = np.stack([bone_vec_axis_1, bone_vec_axis_2], axis=0)
    #
    #     bone_axes.append(bone_local_axis)
    #
    #     # sample points along the bone
    #     # points_1 = start_pos + regular_samples * bone_vec
    #
    #     # start_pos = joints_2[bone[0]]
    #     # end_pos = joints_2[bone[1]]
    #     # bone_vec = end_pos - start_pos
    #     #
    #     # bone_centre = start_pos + 0.5 * bone_vec
    #     # bone_vec_axis_1 = bone_vec / np.linalg.norm(bone_vec)
    #     # bone_vec_axis_2 = perp_stark(bone_vec_axis_1)
    #     # bone_vec_axis_2 /= np.linalg.norm(bone_vec_axis_2)
    #     # bone_vec_axis_3 = np.cross(bone_vec_axis_1, bone_vec_axis_2)
    #     # bone_vec_axis_3 /= np.linalg.norm(bone_vec_axis_3)
    #     # O2 = bone_centre
    #     # scale_factor = np.linalg.norm(0.5 * bone_vec)
    #     #
    #     # # add points to form local axes
    #     # local_axis_points = []
    #     # bone_local_axes = np.stack([bone_vec_axis_1, bone_vec_axis_2, bone_vec_axis_3], axis=0)
    #     #
    #     #
    #     # local_x_points = points_1 + 0.01 * bone_vec_axis_1
    #     # local_y_points = points_1 + 0.01 * bone_vec_axis_2
    #     # local_z_points = points_1 + 0.01 * bone_vec_axis_3
    #     # local_axes_points = np.concatenate([local_x_points, local_y_points, local_z_points], axis=0)
    #     #
    #     # transformation_matrix = construct_transformation_matrix(bone_vec_axis_1, bone_vec_axis_2, bone_vec_axis_3, O2,
    #     #                                                         scale_factor)
    #     #
    #     # transformed_point = convert_points_homogeneous(grid_ip, transformation_matrix)
    #     # global_inducing_points_2.append(transformed_point)
    #     #
    #     #
    #     # # sample points along the bone
    #     # points_2 = start_pos + regular_samples * bone_vec
    #     #
    #     # bone_positions_1.append(points_1)
    #     # bone_positions_2.append(points_2)


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
