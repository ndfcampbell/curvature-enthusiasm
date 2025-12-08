from typing import List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap, lax
from jaxtyping import Array, Int

from .dual_quaternions_func import dual_quaternion_sclerp, dq_mult, dq_normalize, dq_from_rot_trans, \
    dq_transform_point_vmap
from ..utils.quaternion_funcs import q_mult, q_conj
from .sample_rigid_transformations import transform_points_inter
from .sampling_bone_funcs import generate_samples_around_bone, generate_random_samples_around_bone


def compute_parents_jax(edges: List[Tuple[int, int]] | Int[Array, "E 2"]) -> Int[Array, " N"]:
    """
    Compute parent relationships for a tree structure using JAX-based breadth-first search.

    Given a set of edges representing an undirected tree, this function performs BFS
    starting from node 0 (root) to establish parent-child relationships. Each node's
    parent is the node from which it was first discovered during the traversal.

    Args:
        edges: Edge list representing the tree structure. Can be:
               - List of tuples: [(parent, child), ...]
               - JAX array with shape (E, 2) where E is number of edges
               Each edge connects two nodes bidirectionally.

    Returns:
        Parent array with shape (N,) where N is the number of nodes.
        parents[i] contains the parent index of node i, or -1 for the root node (0).

    Algorithm Details:
        1. Build adjacency matrix from edge list
        2. Initialize BFS with root node 0
        3. For each node, mark its unvisited neighbors as children
        4. Continue until all nodes are processed

    Process:
        - Creates adjacency matrix for neighbor lookup
        - Uses BFS queue implemented with fixed-size arrays
        - Processes neighbors using JAX's functional operations
        - Maintains visited status to avoid cycles

    Note:
        - Assumes input represents a valid tree (connected, no cycles)
        - Root node is always node 0
        - Uses JAX control flow (lax.while_loop, lax.fori_loop) for efficient compilation
        - All arrays are pre-allocated with fixed sizes for JAX compatibility

    Example:
        edges = [(0, 1), (0, 2), (1, 3), (1, 4)]  # Tree: 0-1-3, 0-2, 1-4
        parents = compute_parents_jax(edges)      # Result: [-1, 0, 0, 1, 1]
    """

    # Convert edges to JAX array
    edges = jnp.array(edges, dtype=jnp.int32)

    # Get number of nodes
    num_nodes = jnp.max(edges) + 1

    # Create adjacency matrix
    adjacency = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)
    adjacency = adjacency.at[edges[:, 0], edges[:, 1]].set(1)
    adjacency = adjacency.at[edges[:, 1], edges[:, 0]].set(1)

    # Initialize arrays
    parents = -jnp.ones(num_nodes, dtype=jnp.int32)
    visited = jnp.zeros(num_nodes, dtype=jnp.int32)
    queue = jnp.full(num_nodes, -1, dtype=jnp.int32)
    queue = queue.at[0].set(0)  # Start from root (node 0)
    queue_start = jnp.array(0, dtype=jnp.int32)
    queue_size = jnp.array(1, dtype=jnp.int32)

    def bfs_step(carry):
        parents, visited, queue, queue_start, queue_size = carry

        # Get current node
        current = queue[queue_start]
        visited = visited.at[current].set(1)

        # Process all possible neighbors
        def process_neighbor(j, state):
            parents, queue, queue_end = state
            is_neighbor = adjacency[current, j]
            is_unvisited = 1 - visited[j]
            should_process = is_neighbor * is_unvisited

            # Update parent if needed
            parents = parents.at[j].set(
                jnp.where(should_process == 1, current, parents[j])
            )

            # Add to queue if needed
            queue = queue.at[queue_end].set(
                jnp.where(should_process == 1, j, queue[queue_end])
            )

            return (parents, queue, queue_end + should_process)

        # Process all nodes as potential neighbors
        init_state = (parents, queue, queue_size)
        parents, queue, new_queue_size = lax.fori_loop(
            0, num_nodes, process_neighbor, init_state
        )

        return (parents, visited, queue, queue_start + 1, new_queue_size)

    def condition(carry):
        parents, visited, queue, queue_start, queue_size = carry
        return queue_start < queue_size

    # Run BFS until queue is empty
    final_parents, _, _, _, _ = lax.while_loop(
        condition,
        bfs_step,
        (parents, visited, queue, queue_start, queue_size)
    )

    return final_parents

class IK_System_DQ(eqx.Module):

    bone_edges: jax.Array = eqx.field(static=True)
    parents: jax.Array = eqx.field(static=True)
    n_bones: int
    n_joints: int

    local_translations: jax.Array = eqx.field(static=True)
    local_dual_quats: jax.Array = eqx.field(static=True)
    xyz_local_axes: jax.Array = eqx.field(static=True)
    rest_positions: jax.Array = eqx.field(static=True)

    axis_radii: jax.Array = eqx.field(static=True)

    n_rigid_samples_per_bone : int
    n_tissue_samples_per_iter : int
    NUM_MAJOR_AXIS_IPS: int
    NUM_RADIAL_IPS: int
    NUM_ANGULAR_SAMPLES: int

    def __init__(self, ik_skeleton, sampling_settings, dtype=None):
        # Use double precision by default for better numerical stability
        dtype = jnp.float32 if dtype is None else dtype
        int_dtype = jnp.int32

        self.bone_edges = jnp.array(ik_skeleton['bone_edges']).astype(int_dtype)
        # NEED TO CALCULATE, CAREFUL WITH THIS
        self.parents = compute_parents_jax(self.bone_edges)
        self.n_bones = self.bone_edges.shape[0]
        self.n_joints = int(jnp.max(self.bone_edges)) + 1
        self.xyz_local_axes = jnp.array(ik_skeleton['init_local_axes']).astype(dtype)

        positions = jnp.array(ik_skeleton['joints_positions'])

        self.local_translations = self.compute_local_translations(positions)
        self.local_dual_quats = self.compute_local_dual_quaternions(positions)

        # identity_rotations = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (self.n_joints, 1)).astype(dtype)
        # # Initial forward kinematics to set up base positions
        # rest_positions = self.apply_fk(identity_rotations)

        identity = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=dtype)
        identity_rotations = jnp.broadcast_to(identity, (self.n_joints, 4))
        rest_positions = self.apply_fk(identity_rotations)

        self.rest_positions = rest_positions

        self.n_rigid_samples_per_bone = sampling_settings['N_RIGID_SAMPLES_PER_BONE']
        self.n_tissue_samples_per_iter = sampling_settings['N_TISSUE_SAMPLES_PER_ITER']
        self.axis_radii = jnp.array(sampling_settings['AXIS_RADII']).astype(dtype)

        self.NUM_MAJOR_AXIS_IPS = 5  # 5
        self.NUM_RADIAL_IPS = 3
        self.NUM_ANGULAR_SAMPLES = 5


    def compute_local_translations(self, positions):
        def get_local_translation(i, parent_idx):
            return jax.lax.cond(
                parent_idx == -1,
                lambda: positions[i],
                lambda: positions[i] - positions[parent_idx]
            )

        return vmap(get_local_translation)(jnp.arange(self.n_joints), self.parents)

    def compute_local_dual_quaternions(self, positions):
        """Compute local dual quaternions from the rest pose using a consistent convention."""

        def compute_dq(i, parent_idx):
            translation = jax.lax.cond(
                parent_idx == -1,
                lambda: positions[i],  # For the root, use the absolute position
                lambda: positions[i] - positions[parent_idx]
                # Otherwise, use the relative offset
            )
            rotation = jnp.array([1.0, 0.0, 0.0, 0.0])  # Identity rotation for the rest pose
            translation_quat = jnp.concatenate([jnp.array([0.0]), translation])
            dual_part = 0.5 * q_mult(translation_quat, rotation)
            return dq_normalize(jnp.stack([rotation, dual_part]))

        local_dqs = vmap(compute_dq)(jnp.arange(self.n_joints), self.parents)
        return local_dqs

    def apply_fk(self, bone_rotations):
        """Apply FK using dual quaternion transformations."""
        def fk_step(global_dual_quats, carry):
            i, parent_idx = carry

            # Uses precomputed local_translations instead of computing on the fly
            local_dq = dq_from_rot_trans(bone_rotations[i], self.local_translations[i])

            global_dual_quat = jax.lax.cond(
                parent_idx == -1,
                lambda: local_dq,
                lambda: dq_mult(global_dual_quats[parent_idx], local_dq)  # Parent first, then child
            )

            return global_dual_quats.at[i].set(dq_normalize(global_dual_quat)), None

        IDENTITY_DQ = jnp.stack([
            jnp.array([1.0, 0.0, 0.0, 0.0]),  # identity rotation
            jnp.array([0.0, 0.0, 0.0, 0.0])  # zero translation
        ])

        init_dual_quats = jnp.broadcast_to(IDENTITY_DQ, (self.n_joints, 2, 4))

        global_dual_quats, _ = lax.scan(
            fk_step,
            init_dual_quats,
            (jnp.arange(self.n_joints), self.parents)
        )
        zero_point = jnp.zeros(3)
        return dq_transform_point_vmap(global_dual_quats, zero_point)

    def safe_dual_quaternion_interp(self, dq_initial, dq_target, t, threshold=1e-4):
        """Interpolates dual quaternions safely, using LERP when the difference is small."""

        cos_theta = jnp.clip(jnp.sum(dq_initial[0] * dq_target[0]), -1.0, 1.0) # Dot product of real parts (rotation)
        is_close = jnp.abs(1 - cos_theta) < threshold  # Condition to switch methods

        def lerp_interp():
            dq_interp = (1 - t) * dq_initial + t * dq_target
            return dq_normalize(dq_interp)

        def sclerp_interp():
            return dual_quaternion_sclerp(dq_initial.reshape(-1), dq_target.reshape(-1), t).reshape(2, 4)

        return lax.cond(is_close, lerp_interp, sclerp_interp)


    def apply_fk_sclerp(self, initial_rotations, target_rotations, t, interp_method='sclerp', beta=0.01):
        """
        Compute forward kinematics by interpolating between two poses using ScLERP.

        Parameters:
          initial_rotations: array of quaternions for the initial pose (one per joint)
          target_rotations: array of quaternions for the target pose (one per joint)
          t: interpolation parameter (0 means initial, 1 means target)

        Returns:
          global_positions: interpolated global positions of each joint.
        """

        def fk_step(global_dual_quats, carry):
            i, parent_idx = carry

            # Use precomputed local translations instead of computing them each time
            dq_initial = dq_from_rot_trans(initial_rotations[i], self.local_translations[i])
            dq_target = dq_from_rot_trans(target_rotations[i], self.local_translations[i])

            dq_interp = jax.lax.cond(
                jnp.isclose(t, 0.0),
                lambda: dq_initial,
                lambda: jax.lax.cond(
                    jnp.isclose(t, 1.0),
                    lambda: dq_target,
                    lambda: self.safe_dual_quaternion_interp(dq_initial, dq_target, t)
                )
            )

            dq_interp = dq_normalize(dq_interp)

            # Compute global transform: for root, just use dq_interp, otherwise multiply parent's global dq.
            global_dq = jax.lax.cond(
                parent_idx == -1,
                lambda: dq_interp,
                lambda: dq_mult(global_dual_quats[parent_idx], dq_interp)
            )
            # Store the normalized global dual quaternion.
            return global_dual_quats.at[i].set(dq_normalize(global_dq)), None

        IDENTITY_DQ = jnp.stack([
            jnp.array([1.0, 0.0, 0.0, 0.0]),  # identity rotation
            jnp.array([0.0, 0.0, 0.0, 0.0])  # zero translation
        ])

        init_dual_quats = jnp.broadcast_to(IDENTITY_DQ, (self.n_joints, 2, 4))

        global_dual_quats, _ = lax.scan(
            fk_step,
            init_dual_quats,
            (jnp.arange(self.n_joints), self.parents)
        )

        return dq_transform_point_vmap(global_dual_quats, jnp.zeros(3)), global_dual_quats


    def create_random_samples(self, key=jr.key(0)):
        bone_key, tissue_key = jr.split(key)

        # Generate keys for each init_local_axis
        # num_axes = self.init_local_axis.shape[0]  # Assuming this is the correct shape
        # num_axes = self.n_bones
        bone_keys = jax.random.split(bone_key, self.n_bones)
        tissue_keys = jax.random.split(tissue_key, self.n_bones)


        bone_segments = self.rest_positions[self.bone_edges]
        v0, v1 = bone_segments[:, 0], bone_segments[:, 1]

        bone_samples = jax.vmap(generate_random_samples_around_bone,  (0, 0, 0, 0, None, None, None, 0))(v0, v1, self.xyz_local_axes,
                                                                                                         self.axis_radii,
                                                                                                         0.1,
                                                                                                         0.9,
                                                                                                         self.n_rigid_samples_per_bone, bone_keys)


        return bone_samples

    def create_samples(self):
        bone_segments = self.rest_positions[self.bone_edges]
        v0, v1 = bone_segments[:, 0], bone_segments[:, 1]
        bone_samples = jax.vmap(generate_samples_around_bone,  (0, 0, 0, 0, None, None, None, None, None))(v0, v1, self.xyz_local_axes,
                                                                                                           self.axis_radii,
                                                                                                           0.1,
                                                                                                           0.9,
                                                                                                           self.NUM_MAJOR_AXIS_IPS,
                                                                                                           self.NUM_RADIAL_IPS,
                                                                                                           self.NUM_ANGULAR_SAMPLES)


        return bone_samples

    def rotate_vector(self, q, vec):
        p = jnp.concatenate([jnp.array([0.0]), vec])
        rotated = q_mult(q_mult(q, p), q_conj(q))
        return rotated[1:]

    def align_bone_transformation(self, u, v):
        Q = jnp.array([1.0, 0.0, 0.0, 0.0])
        U = jnp.array([0.0, u[0], u[1], u[2]])
        V = jnp.array([0.0, v[0], v[1], v[2]])
        r_quat = jnp.linalg.norm(q_mult(U, V)) * Q - q_mult(q_mult(V, Q), U)
        return r_quat / jnp.linalg.norm(r_quat)


    def transform_bone_samples_inter(self, bone_samples: jax.Array, translations: jax.Array, add_q: jax.Array,
                                     time: float):
        """Transform bone samples with interpolation and normalized quaternions."""
        bone_samples = jax.vmap(transform_points_inter, (0, 0, 0, 0, None))(
            self.v0, bone_samples, translations, add_q, time
        )

        return bone_samples.reshape(-1, 3)


    def calc_bone_sample_traj(self, bone_samples: jax.Array, translations: jax.Array, add_q: jax.Array, time: float):
        return jax.vmap(self.transform_bone_samples_inter, (None, None, None, 0))(bone_samples, translations, add_q, time)

    def transform_local_points_per_bone(self, p0: jax.Array, local_pts: jax.Array, translation: jax.Array, qt: jax.Array, time: float):
        t = time * translation
        local_ips = self.rotate_group(local_pts, qt)
        global_ips = (p0 + local_ips) #+ t
        return global_ips

    def transform_local_points_per_step(self, local_pts: jax.Array, p0: jax.Array, translation: jax.Array, qt: jax.Array, time: float):
        return jax.vmap(self.transform_local_points_per_bone, (0, 0, 0, 0, None))(p0, local_pts, translation, qt, time)

    def pick_closest_quaternion(self, quaternion, target_quaternion, eps=1e-6):
        """Ensure quaternion is closest to the target quaternion."""
        dot_product = jnp.clip(jnp.dot(quaternion, target_quaternion), -1.0, 1.0)
        flip = jnp.where(dot_product < 0.0, -1.0, 1.0)
        return flip * quaternion

    def compute_samples_over_time(self, t_values, test_rotations, local_samples_all):
        """
        Vectorized computation of samples over time.
        """
        parent_indices = self.bone_edges[:, 0]
        child_indices = self.bone_edges[:, 1]

        # @jit
        def compute_samples_core(t_values, test_rots, local_samples):
            def compute_fk_at_time(t):
                # identity_rotations = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (self.n_joints, 1))
                identity = jnp.array([1.0, 0.0, 0.0, 0.0])
                identity_rotations = jnp.broadcast_to(identity, (self.n_joints, 4))
                return self.apply_fk_sclerp(identity_rotations, test_rots, t)

            # Get positions for all timesteps
            positions_and_dqs = vmap(compute_fk_at_time)(t_values)
            global_positions = positions_and_dqs[0]  # [num_t, num_joints, 3]

            def process_timestep(positions):
                # Get bone start and end positions
                bone_starts = positions[parent_indices]  # [num_bones, 3]
                bone_ends = positions[child_indices]  # [num_bones, 3]

                # Get rest pose bone directions
                rest_directions = self.rest_positions[child_indices] - self.rest_positions[parent_indices]  # [num_bones, 3]

                # Get current bone directions
                bone_directions = bone_ends - bone_starts  # [num_bones, 3]

                # Compute rotations for all bones
                # rotations = vmap(self.align_vector_to_vector)(rest_directions, bone_directions)  # [num_bones, 4]

                rotations = vmap(self.align_bone_transformation)(rest_directions, bone_directions)

                def transform_bone_samples(args):
                    rotation, bone_start, rest_parent_pos, samples = args
                    # Convert samples to local space relative to rest parent position
                    local_samples = samples - rest_parent_pos

                    # Rotate all samples at once
                    rotated = vmap(self.rotate_vector, in_axes=(None, 0))(rotation, local_samples)

                    # # Create a batch of identical rotations matching the number of samples
                    # rotations_batch = jnp.tile(rotation, (samples.shape[0], 1))
                    # Rotate all samples at once
                    # rotated = vmap(self.rotate_vector)(rotations_batch, local_samples)

                    # Translate to final position
                    return rotated + bone_start

                # Transform all bones' samples
                return vmap(transform_bone_samples)((
                    rotations,
                    bone_starts,
                    self.rest_positions[parent_indices],
                    local_samples
                ))

            # Process all timesteps
            return global_positions, vmap(process_timestep)(global_positions)

        # Call the JIT-compiled core function
        return compute_samples_core(
            t_values,
            test_rotations,
            local_samples_all
        )

    # ScLERP
    def calc_system_train_sample_traj(self, bone_samples: jax.Array, root_translation: jax.Array, pred_quat_rots: jax.Array, time_samples: jax.Array):
        rigid_rotated_joint_positions_traj, rigid_rotated_bone_samples_traj = self.compute_samples_over_time(time_samples, pred_quat_rots, bone_samples)
        rigid_rotated_bone_samples_traj = rigid_rotated_bone_samples_traj.reshape(time_samples.shape[0], -1, 3)
        return rigid_rotated_joint_positions_traj, rigid_rotated_bone_samples_traj
