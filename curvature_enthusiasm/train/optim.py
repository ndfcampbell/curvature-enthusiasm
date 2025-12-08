from typing import Tuple
import numpy as np
import optax

from .adabelief_cautious import adabelief_cautious

def create_lr_schedule(config) -> optax.Schedule:
    """Build 3-phase LR schedule with per-phase warmup."""
    schedules, boundaries = [], []
    cumulative = 0

    def phase_schedule(total: int, peak: float, end: float):
        if total <= 0:
            return None
        warmup = min(int(getattr(config.optimization, "warmup_steps", 0)), total)
        if abs(peak - end) < 1e-9:
            if warmup >= total:
                return optax.linear_schedule(0.0, peak, total)
            return optax.join_schedules(
                [optax.linear_schedule(0.0, peak, warmup), optax.constant_schedule(peak)],
                boundaries=[warmup],
                )
        return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak,
        warmup_steps=warmup,
        decay_steps=total,
        end_value=end,
        )

    var_iters = list(getattr(config.varifold, "iters_per_lengthscale", [])) or []
    nc_iters = list(getattr(config.normal_cycles, "iters_per_lengthscale", [])) or []
    var_total = int(np.sum(var_iters)) if var_iters else 0
    nc_total = int(np.sum(nc_iters)) if nc_iters else 0
    ft_total = int(getattr(config.fine_tuning, "ft_num_iters", 0)) if getattr(config.fine_tuning, "use_fine_tuning", False) else 0


    for total, peak, end in [
        (var_total, config.optimization.peak_lr, config.optimization.peak_lr),
        (nc_total, config.optimization.peak_lr, config.optimization.end_lr),
        (ft_total, config.optimization.ft_lr, config.optimization.ft_lr),
        ]:
            sched = phase_schedule(total, peak, end)
            if sched is not None:
                if schedules:
                    boundaries.append(cumulative)
                schedules.append(sched)
                cumulative += total


    if not schedules:
        return optax.constant_schedule(0.0)
    if len(schedules) == 1:
        return schedules[0]
    return optax.join_schedules(schedules, boundaries)

def build_optimizer(config) -> Tuple[optax.GradientTransformation, optax.Schedule]:
    lr_schedule = create_lr_schedule(config)
    optimizer = optax.chain(
    # optax.zero_nans(),
    optax.clip_by_global_norm(0.5),
    adabelief_cautious(lr_schedule),
    )
    return optimizer, lr_schedule