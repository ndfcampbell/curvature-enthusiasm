from typing import Callable, Dict, List, Optional, Tuple, Any
import equinox as eqx
import jax
import jax.numpy as jnp

class TrainingData(eqx.Module):
    """Training payload that gets updated during compression resampling."""
    points: jax.Array  # (batch, points, 3)
    faces: jax.Array  # (batch, faces, 3)
    weights: Optional[jax.Array] # (batch, points) - for compression
    source_keypoints: Optional[jax.Array] = eqx.field(static=True) # (kp, 3)
    target_keypoints: Optional[jax.Array] = eqx.field(static=True) # (kp, 3)


class PhaseConfig(eqx.Module):
    """Configuration for a single training phase."""
    name: str = eqx.field(static=True)
    total_iters: int = eqx.field(static=True)
    kernel_params: List[Dict[str, jnp.ndarray]] = eqx.field(static=True)
    transition_steps: List[int] = eqx.field(static=True) # Relative to phase start
    loss_grad_fn: Callable = eqx.field(static=True)


    def current_kernel_idx(self, step_in_phase: int) -> int:
        idx = 0
        for transition in self.transition_steps:
            if step_in_phase >= transition:
                idx += 1
            else:
                break
        return min(idx, len(self.kernel_params) - 1)

class TrainingSchedule(eqx.Module):
    """Complete training schedule with phase boundaries."""
    phases: List[PhaseConfig] = eqx.field(static=True)
    total_iters: int = eqx.field(static=True)
    ode_output_scaling: jax.Array = eqx.field(static=True)# (total_iters + 1,)

    @property
    def phase_boundaries(self) -> List[int]:
        boundaries = [0]
        for phase in self.phases:
            boundaries.append(boundaries[-1] + phase.total_iters)
        return boundaries

    def get_phase_and_step(self, global_step: int) -> Tuple[int, int]:
        boundaries = self.phase_boundaries
        for i in range(len(boundaries) - 1):
            if boundaries[i] <= global_step < boundaries[i + 1]:
                return i, global_step - boundaries[i]
        return len(self.phases) - 1, global_step - boundaries[-2]

class NCCompressedInfo(eqx.Module):
    """NC structures for NC compression."""
    has_boundary: bool = eqx.field(static=True)
    template_nc_struct: Dict = eqx.field(static=True)
    target_nc_struct: Dict = eqx.field(static=True)
    nc_source_centres: jax.Array = eqx.field(static=True)