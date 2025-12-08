from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from probdiffeq import ivpsolve, ivpsolvers, taylor, stats
from jaxtyping import Float, Array

from curvature_enthusiasm.utils import safe_normalize_quaternion, pin_to_hemisphere
from .node import NODE
from .gso_mlp import GSO_MLP
from .ik_system_dq import IK_System_DQ
from .mdmm import MDMM


def _extract_trajs(
    y_traj: Float[Array, "n_steps total 3"],
    m: int,
    n: int,
) -> tuple[
    Float[Array, "n_steps m 3"],      # y_ode_traj
    Float[Array, "n_steps n 3"],      # bone_ode_traj
    Float[Array, "n_steps tissue 3"], # tissue_ode_traj
]:
    """Split [n_steps, (m+n+... ), 3] into (y, bone, tissue) pieces."""
    y_ode_traj      = y_traj[:, :m, :]
    bone_ode_traj   = y_traj[:, m:m + n, :]
    tissue_ode_traj = y_traj[:, m + n:, :]
    return y_ode_traj, bone_ode_traj, tissue_ode_traj

def _add_axis_offsets(
    points: Float[Array, "N 3"],
    offset: float = 0.01,
) -> Float[Array, "N4 3"]:
    """[N,3] -> [N, 1+3, 3] where the extra 3 are axis-aligned offsets."""
    offset = jnp.asarray(offset, dtype=points.dtype)
    E3 = jnp.eye(3, dtype=points.dtype)
    offset_points = points[:, jnp.newaxis, :] + E3 * offset   # [N,3,3]
    ts_x0 = jnp.concatenate([points[:, jnp.newaxis, :], offset_points], axis=1)  # [N,4,3]
    return ts_x0.reshape(-1, 3)  # [N*4, 3]

class ODESolution(NamedTuple):
    y_ode_traj: Float[Array, "n_steps m 3"]
    bone_ode_traj: Float[Array, "n_steps n 3"]
    tissue_ode_traj: Float[Array, "n_steps tissue 3"]
    bone_samples: Float[Array, "n 3"]
    rigid_rotated_bone_samples_traj: Float[Array, "n_steps n 3"]
    rigid_rotated_joint_positions_traj: Float[Array, "n_steps joints 3"]


class StandardVF(eqx.Module):
    ode_func: NODE

    @eqx.filter_jit
    def __call__(self, x, t):
        t_left = jnp.asarray(1.0, x.dtype) - jnp.asarray(t, x.dtype)
        return jax.vmap(self.ode_func, (0, None))(x, t_left)


class DivFreeVF(eqx.Module):
    ode_func: NODE

    @eqx.filter_jit
    def __call__(self, x, t):
        """
        Compute divergence-free velocity as curl of learned vector potential A.
        v = ∇ × A ensures ∇·v = 0.
        """
        t_left = jnp.asarray(1.0, x.dtype) - jnp.asarray(t, x.dtype)

        # Jacobian dA/dx for each point: [N, 3, 3]
        J = eqx.filter_vmap(
            eqx.filter_jacrev(self.ode_func),
            in_axes=(0, None)
        )(x, t_left)

        # curl(A) = [∂Az/∂y - ∂Ay/∂z, ∂Ax/∂z - ∂Az/∂x, ∂Ay/∂x - ∂Ax/∂y]
        curl = jnp.stack([
            J[:, 2, 1] - J[:, 1, 2],
            J[:, 0, 2] - J[:, 2, 0],
            J[:, 1, 0] - J[:, 0, 1],
        ], axis=1)
        return curl

# ALTERNATIVE CURL IMPLEMENTATION
# @eqx.filter_jit
# def predict_ode_func(self, x, t):
#     E = jnp.eye(3, dtype=x.dtype)  # replaces e0/e1/e2
#     t_left = jnp.asarray(1.0 - t, x.dtype)  # keep dtype stable to avoid recompiles
#
#     def curl_at(xi):
#         _, vjp = eqx.filter_vjp(lambda y: self.ode_func(y, t_left), xi)
#
#         # exactly three calls, now referencing rows of the identity
#         (r0,) = vjp(E[0])  # ∂F*/∂x  (row 0)
#         (r1,) = vjp(E[1])  # ∂F*/∂y  (row 1)
#         (r2,) = vjp(E[2])  # ∂F*/∂z  (row 2)
#
#         # curl(F)
#         return jnp.array([
#             r2[1] - r1[2],
#             r0[2] - r2[0],
#             r1[0] - r0[1],
#         ], dtype=xi.dtype)
#
#     return eqx.filter_vmap(curl_at)(x)

class CE_MODEL(eqx.Module):
    ik_system: IK_System_DQ
    conformal_func: GSO_MLP
    end_traj_bone_mdmm: MDMM
    traj_bone_mdmm: MDMM
    root_translation: Float[Array, "3"]
    raw_pred_quat_rots: Float[Array, "B 4"]
    velocity: eqx.Module

    n_ode_timesteps: int = eqx.field(static=True)
    n_tissue_samples: int = eqx.field(static=True)

    def __init__(self,
                 ik_system,
                 ode_func,
                 conformal_func,
                 n_tissue_samples=1000,
                 damping_factor=1.0,
                 n_ode_timesteps=5,
                 use_div_free: bool = False,
                 key=jr.PRNGKey(0),
                 dtype=None):

        k_noise, _ = jr.split(key)

        self.ik_system = ik_system
        self.conformal_func = conformal_func
        self.n_tissue_samples = n_tissue_samples
        self.n_ode_timesteps = n_ode_timesteps
        self.root_translation = jnp.array([0.0, 0.0, 0.0])
        raw_pred_quat_rots = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (ik_system.n_bones, 1))
        # add some noise to the initial quaternions
        self.raw_pred_quat_rots = raw_pred_quat_rots + jr.normal(k_noise, raw_pred_quat_rots.shape)
        self.end_traj_bone_mdmm = MDMM(lm_init=jnp.zeros((ik_system.n_bones,)), damping=damping_factor, target_values=1e-4, scale=1.0)
        self.traj_bone_mdmm = MDMM(lm_init=jnp.zeros((ik_system.n_bones,)), damping=damping_factor, target_values=1e-4, scale=1.0)

        self.velocity = DivFreeVF(ode_func) if use_div_free else StandardVF(ode_func)

    @property
    def pred_quat_rots(self):
        """Get predicted quaternion rotations, normalized and hemisphere-pinned."""
        quats = safe_normalize_quaternion(self.raw_pred_quat_rots)
        return pin_to_hemisphere(quats, hemisphere='positive')


    @eqx.filter_jit
    def predict_ode_func(self, x, t):
        return self.velocity(x, t)

    @eqx.filter_checkpoint
    def state_update(self, state, t):
        return self.predict_ode_func(state, jnp.asarray(t, state.dtype))

    def create_rigid_rotated_bone_info(self, n_ode_timesteps, key):

        bone_samples = self.ik_system.create_random_samples(key)

        # wobble the angle a bit
        # pred_quat_rots = self.pred_quat_rots + 1.0 * jr.normal(jr.PRNGKey(0), self.pred_quat_rots.shape)
        # pred_quat_rots = pred_quat_rots / jnp.linalg.norm(pred_quat_rots, axis=-1, keepdims=True)

        t_samples = jnp.linspace(0.0, 1.0, n_ode_timesteps)
        rigid_rotated_joint_positions_traj, rigid_rotated_bone_samples_traj = self.ik_system.calc_system_train_sample_traj(bone_samples,
                                                                                                                    self.root_translation,
                                                                                                                    self.pred_quat_rots,
                                                                                                                    t_samples)

        return bone_samples, rigid_rotated_bone_samples_traj, rigid_rotated_joint_positions_traj


    # NOTE ACAP QUATERNIONS ARE W = [0,0,0,1], RIGID TRNANFORMATION QUATERNIONS ARE W = [1,0,0,0]
    def calculate_Q(self, y_pred_traj, tissue_sample_traj):

        time_segments = jnp.linspace(0.0, 1.0, self.n_ode_timesteps)[1:]
        # n_timesteps, n_points, _ = y_pred_traj.shape

        # Prepare point cloud input
        point_cloud_shape = y_pred_traj.shape
        y_point_cloud = y_pred_traj.reshape(-1, 3)

        # Prepare tissue sample input
        y_tissue = tissue_sample_traj[:, ::4, :].reshape(-1, 3)

        # Combine both inputs
        combined_y = jnp.concatenate([y_point_cloud, y_tissue], axis=0)

        # Create time column for point cloud input
        time_column_point_cloud = jnp.repeat(time_segments, point_cloud_shape[1])

        samples_per_time = tissue_sample_traj.shape[1] // 4
        # Create time column for tissue sample input
        time_column_tissue = jnp.repeat(time_segments, samples_per_time)

        # Combine time columns
        time_column = jnp.concatenate([time_column_point_cloud, time_column_tissue])

        # Ensure time_column has the correct shape
        time_column = time_column.reshape(-1, 1)

        # Combine spatial coordinates and time
        combined_input = jnp.concatenate([combined_y, time_column], axis=1)

        # Call conformal_func once
        combined_Q = self.conformal_func(combined_input)

        # Split the output
        split_index = point_cloud_shape[0] * point_cloud_shape[1]
        Q_point_cloud = combined_Q[:split_index].reshape(point_cloud_shape[0], point_cloud_shape[1], 4)
        Q_tissue = combined_Q[split_index:].reshape(self.n_ode_timesteps - 1, tissue_sample_traj.shape[1] // 4, 4)

        return Q_point_cloud, Q_tissue


    def build_state(self, x0, tissue_samples, key, offset=0.01):
        """Returns: state, m, n, bone_samples, rigid_rotated_*_traj."""
        # Rigid bone info
        bone_samples, rigid_rotated_bone_samples_traj, rigid_rotated_joint_positions_traj = \
            self.create_rigid_rotated_bone_info(self.n_ode_timesteps, key)
        bone_samples = bone_samples.reshape(-1, 3)

        # Expand tissue samples with axis offsets
        ts_x0 = _add_axis_offsets(tissue_samples, offset=offset)

        # Pack state
        m = x0.shape[0]
        n = bone_samples.shape[0]
        state = jnp.concatenate([x0, bone_samples, ts_x0], axis=0)

        return (state, m, n, bone_samples,
                rigid_rotated_bone_samples_traj,
                rigid_rotated_joint_positions_traj)

    def _setup_solver(self, x0_states, time_grid, ode_output_scale):
        """
        Common solver initialization logic.

        Args:
            x0_states: Initial state vector
            time_grid: Time points for integration
            ode_output_scale: Output scale parameter for solver

        Returns:
            Tuple of (init, solver, ssm) for use with ivpsolve
        """
        n_derivatives = 1

        tcoeffs = taylor.odejet_padded_scan(
            lambda y: self.state_update(y, time_grid[0]),
            (x0_states,),
            num=n_derivatives
        )

        init, ibm, ssm = ivpsolvers.prior_wiener_integrated(
            tcoeffs,
            output_scale=ode_output_scale,
            ssm_fact="isotropic"
        )

        ts0 = ivpsolvers.correction_ts0(self.state_update, ssm=ssm)
        strategy = ivpsolvers.strategy_smoother(ssm=ssm)
        solver = ivpsolvers.solver_mle(strategy, prior=ibm, correction=ts0, ssm=ssm)

        return init, solver, ssm

    def _create_ode_solution(
            self,
            y_traj: Float[Array, "n_steps total 3"],
            m: int,
            n: int,
            bone_samples: Float[Array, "n 3"],
            rigid_rotated_bone_samples_traj: Float[Array, "n_steps n 3"],
            rigid_rotated_joint_positions_traj: Float[Array, "n_steps joints 3"],
    ) -> ODESolution:
        """
        Helper to construct ODESolution from trajectory.

        Args:
            y_traj: Combined trajectory [n_steps, total_points, 3]
            m: Number of point cloud points
            n: Number of bone sample points
            bone_samples: Initial bone samples
            rigid_rotated_bone_samples_traj: Rigid bone trajectory
            rigid_rotated_joint_positions_traj: Rigid joint trajectory

        Returns:
            ODESolution with all trajectories
        """
        y_ode_traj, bone_ode_traj, tissue_ode_traj = _extract_trajs(y_traj, m, n)

        return ODESolution(
            y_ode_traj=y_ode_traj,
            bone_ode_traj=bone_ode_traj,
            tissue_ode_traj=tissue_ode_traj,
            bone_samples=bone_samples,
            rigid_rotated_bone_samples_traj=rigid_rotated_bone_samples_traj,
            rigid_rotated_joint_positions_traj=rigid_rotated_joint_positions_traj,
        )

    def solve_fixed(
            self,
            x0: Float[Array, "m 3"],
            tissue_samples: Float[Array, "n_tissue 3"],
            ode_output_scale: float,
            key: jax.random.PRNGKey
    ) -> ODESolution:
        """
        Solve ODE on fixed time grid.

        Args:
            x0: Initial point cloud positions [m, 3]
            tissue_samples: Tissue sample points [n_tissue, 3]
            ode_output_scale: Output scale for ODE solver
            key: Random key for bone sampling

        Returns:
            ODESolution with trajectories and rigid transformations
        """
        # Build the concatenated state
        (x0_states, m, n, bone_samples,
         rigid_rotated_bone_samples_traj,
         rigid_rotated_joint_positions_traj) = self.build_state(x0, tissue_samples, key)

        time_grid = jnp.linspace(0.0, 1.0, self.n_ode_timesteps)

        # Setup solver
        init, solver, ssm = self._setup_solver(x0_states, time_grid, ode_output_scale)

        # Solve on fixed grid
        sol = ivpsolve.solve_fixed_grid(
            init,
            grid=time_grid,
            solver=solver,
            ssm=ssm
        )

        # Extract and return solution
        return self._create_ode_solution(
            sol.u[0], m, n, bone_samples,
            rigid_rotated_bone_samples_traj,
            rigid_rotated_joint_positions_traj
        )

    def _sample_tissue(
            self,
            tetra_centres: Float[Array, "n_centres 3"],
            n_samples: int,
            key: jax.random.PRNGKey
    ) -> Float[Array, "n_samples 3"]:
        """
        Sample tissue points from tetrahedral centers without replacement.

        Args:
            tetra_centres: Available tetrahedral centers [n_centres, 3]
            n_samples: Number of samples to draw
            key: Random key for sampling

        Returns:
            Sampled tissue points [n_samples, 3]
        """
        indices = jr.permutation(key, tetra_centres.shape[0])[:n_samples]
        return tetra_centres[indices]

    def calc_ode_trajectory(
            self,
            x0: Float[Array, "m 3"],
            n_trajectory_steps: int,
            tetra_centres: Float[Array, "n_centres 3"],
            ode_output_scale: float,
            random_key: jax.random.PRNGKey,
            n_tissue_samples: int = 2000
    ) -> ODESolution:
        """
        Calculate ODE trajectory with custom interpolation to arbitrary time steps.

        This method first solves on the standard grid (self.n_ode_timesteps),
        then interpolates to n_trajectory_steps for smooth animation.

        Args:
            x0: Initial point cloud positions [m, 3]
            n_trajectory_steps: Number of time steps for interpolated output
            tetra_centres: Tetrahedral centers for tissue sampling [n_centres, 3]
            ode_output_scale: Output scale for ODE solver
            random_key: Random key for sampling

        Returns:
            ODESolution with interpolated trajectories at n_trajectory_steps
        """
        # Sample tissue points
        node_key, tissue_key = jr.split(random_key, 2)

        tissue_samples = self._sample_tissue(tetra_centres, n_tissue_samples, tissue_key)

        # Build the concatenated state
        (x0_states, m, n, bone_samples,
         rigid_rotated_bone_samples_traj,
         rigid_rotated_joint_positions_traj) = self.build_state(x0, tissue_samples, node_key)

        time_grid = jnp.linspace(0.0, 1.0, self.n_ode_timesteps)

        # Setup solver (uses self.n_ode_timesteps for initial solve)
        init, solver, ssm = self._setup_solver(x0_states,time_grid, ode_output_scale)

        # Solve on standard grid
        sol = ivpsolve.solve_fixed_grid(
            init,
            grid=time_grid,
            solver=solver,
            ssm=ssm
        )

        # Interpolate to requested time steps
        ts = jnp.linspace(0.0 + 1e-6, 1.0 - 1e-6, n_trajectory_steps)
        u, _ = stats.offgrid_marginals_searchsorted(ts=ts, solution=sol, solver=solver)

        # Extract and return interpolated solution
        return self._create_ode_solution(
            u[0], m, n, bone_samples,
            rigid_rotated_bone_samples_traj,
            rigid_rotated_joint_positions_traj
        )


    def __call__(self, source_points, source_keypoints, tetra_centres, ode_output_scale, random_key, extra_tissue_samples=None
                 ) -> tuple[
                    ODESolution,
                    Float[Array, "n_steps_minus_1 m 4"],
                    Float[Array, "n_steps_minus_1 tissue_over_4 4"],
                ]:
        """
        Forward pass through the model.

        Args:
            input_points: Initial point cloud [m, 3]
            source_keypoints: Optional source keypoints [m, 3]
            tetra_centres: Tetrahedral centers for sampling [n_centres, 3]
            ode_output_scale: Output scale for ODE solver
            random_key: Random key for sampling
            extra_tissue_samples: Optional additional tissue samples

        Returns:
            Tuple of (ode_solution, Q_point_cloud, Q_tissue)
        """
        node_key, tissue_samples_key = jr.split(random_key, 2)

        n_source = source_points.shape[0]
        tissue_samples = self._sample_tissue(tetra_centres, self.n_tissue_samples, tissue_samples_key)

        if extra_tissue_samples is not None:
            tissue_samples = jnp.concatenate([tissue_samples, extra_tissue_samples], axis=0)

        # if source_keypoints is not None:
        #     input_points = jnp.concatenate([source_points, source_keypoints], axis=0)
        # else:
        #     input_points = source_points

        if source_keypoints is None:
            source_keypoints = jnp.zeros((0, source_points.shape[-1]))

        # Concatenate with source_points first
        input_points = jnp.concatenate([source_points, source_keypoints], axis=0)

        ode_sol = self.solve_fixed(input_points, tissue_samples, ode_output_scale, node_key)

        # extract source points from y_ode_traj
        y_ode_input_points_traj = ode_sol.y_ode_traj[:, :n_source]

        Q_point_cloud, Q_tissue = self.calculate_Q(y_ode_input_points_traj[1:], ode_sol.tissue_ode_traj[1:])

        return ode_sol, Q_point_cloud, Q_tissue




