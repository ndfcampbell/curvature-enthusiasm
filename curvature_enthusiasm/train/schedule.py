from typing import List
import numpy as np
import jax.numpy as jnp
import equinox as eqx


from .data_structures import TrainingSchedule, PhaseConfig
from .kernel_params import generate_kernel_params # assumes same path as before

# def create_training_schedule(config, varifold_cost_fn, nc_cost_fn) -> TrainingSchedule:
#     """Build the complete training schedule with phases.
#     Creates gradient functions once and stores them in the phases.
#     """
#     # Parse iteration counts
#     var_iters = list(getattr(config.varifold, "iters_per_lengthscale", [])) or []
#     nc_iters = list(getattr(config.normal_cycles, "iters_per_lengthscale", [])) or []
#     use_ft = bool(getattr(config.fine_tuning, "use_fine_tuning", False))
#     ft_iters = int(getattr(config.fine_tuning, "ft_num_iters", 0)) if use_ft else 0
#
#
#     var_total = int(np.sum(var_iters)) if var_iters else 0
#     nc_total = int(np.sum(nc_iters)) if nc_iters else 0
#     total_iters = var_total + nc_total + ft_iters
#
#
#     # Generate kernel parameters
#     var_kernels = generate_kernel_params(config.varifold.sigmas, config.varifold.sigma_sphs)
#     nc_kernels = generate_kernel_params(config.normal_cycles.sigmas)
#
#
#     # Create gradient functions once
#     var_grad_fn = eqx.filter_value_and_grad(varifold_cost_fn)
#     nc_grad_fn = eqx.filter_value_and_grad(nc_cost_fn)
#
#
#     # Build phases with relative transition steps
#     phases: List[PhaseConfig] = []
#
#     if var_total > 0:
#         var_transitions = np.cumsum(var_iters[:-1]).tolist() if len(var_iters) > 1 else []
#         phases.append(PhaseConfig(
#         name="varifold",
#         total_iters=var_total,
#         kernel_params=var_kernels,
#         transition_steps=var_transitions,
#         loss_grad_fn=var_grad_fn,
#         ))
#
#
#     if nc_total > 0:
#         nc_transitions = np.cumsum(nc_iters[:-1]).tolist() if len(nc_iters) > 1 else []
#         phases.append(PhaseConfig(
#         name="normal_cycles",
#         total_iters=nc_total,
#         kernel_params=nc_kernels,
#         transition_steps=nc_transitions,
#         loss_grad_fn=nc_grad_fn,
#         ))
#
#
#     if ft_iters > 0:
#         phases.append(PhaseConfig(
#         name="fine_tuning",
#         total_iters=ft_iters,
#         kernel_params=[nc_kernels[-1]] if len(nc_kernels) > 0 else [],
#         transition_steps=[],
#         loss_grad_fn=nc_grad_fn,
#         ))
#
#
#     ode_output_scaling = jnp.log10(jnp.linspace(20.0, 1.0, total_iters + 1)) + jnp.array(0.01)
#
#
#     # Logging
#     print("Training schedule (iterations):")
#     print(f" Initialization (varifold): {var_total} iters (per-scale: {var_iters})")
#     print(f" Main training (normal cycles): {nc_total} iters (per-scale: {nc_iters})")
#     print(f" Fine-tuning: {ft_iters} iters")
#     print(f" Total: {total_iters} iters")
#
#
#     return TrainingSchedule(phases=phases, total_iters=total_iters, ode_output_scaling=ode_output_scaling)

def create_training_schedule(config, varifold_cost_fn, nc_cost_fn) -> TrainingSchedule:
    """Build the complete training schedule with phases.
    Creates gradient functions once and stores them in the phases.
    """
    # Parse iteration counts
    var_iters = list(getattr(config.varifold, "iters_per_lengthscale", [])) or []
    nc_iters = list(getattr(config.normal_cycles, "iters_per_lengthscale", [])) or []
    use_ft = bool(getattr(config.fine_tuning, "use_fine_tuning", False))
    ft_iters = int(getattr(config.fine_tuning, "ft_num_iters", 0)) if use_ft else 0

    var_total = int(np.sum(var_iters)) if var_iters else 0
    nc_total = int(np.sum(nc_iters)) if nc_iters else 0
    total_iters = var_total + nc_total + ft_iters

    # Generate kernel parameters
    var_kernels = generate_kernel_params(config.varifold.sigmas, config.varifold.sigma_sphs)
    nc_kernels = generate_kernel_params(config.normal_cycles.sigmas)

    # Create gradient functions once
    var_grad_fn = eqx.filter_value_and_grad(varifold_cost_fn)
    nc_grad_fn = eqx.filter_value_and_grad(nc_cost_fn)

    # Build phases with relative transition steps
    phases: List[PhaseConfig] = []

    if var_total > 0:
        var_transitions = np.cumsum(var_iters[:-1]).tolist() if len(var_iters) > 1 else []
        phases.append(PhaseConfig(
            name="varifold",
            total_iters=var_total,
            kernel_params=var_kernels,
            transition_steps=var_transitions,
            loss_grad_fn=var_grad_fn,
        ))

    # Only add normal_cycles phase if nc_total > 0
    if nc_total > 0:
        nc_transitions = np.cumsum(nc_iters[:-1]).tolist() if len(nc_iters) > 1 else []
        phases.append(PhaseConfig(
            name="normal_cycles",
            total_iters=nc_total,
            kernel_params=nc_kernels,
            transition_steps=nc_transitions,
            loss_grad_fn=nc_grad_fn,
        ))

    if ft_iters > 0:
        # Fine-tuning uses the loss from the previous phase
        # If we had NC phase, use NC loss and kernels; otherwise use varifold
        if nc_total > 0 and len(nc_kernels) > 0:
            ft_kernel_params = [nc_kernels[-1]]
            ft_loss_grad_fn = nc_grad_fn
            ft_loss_name = "normal_cycles"
        elif len(var_kernels) > 0:
            ft_kernel_params = [var_kernels[-1]]
            ft_loss_grad_fn = var_grad_fn
            ft_loss_name = "varifold"
        else:
            ft_kernel_params = []
            ft_loss_grad_fn = var_grad_fn
            ft_loss_name = "varifold"

        phases.append(PhaseConfig(
            name="fine_tuning",
            total_iters=ft_iters,
            kernel_params=ft_kernel_params,
            transition_steps=[],
            loss_grad_fn=ft_loss_grad_fn,
        ))

    ode_output_scaling = jnp.log10(jnp.linspace(20.0, 1.0, total_iters + 1)) + jnp.array(0.01)

    # Logging
    print("Training schedule (iterations):")
    print(f" Initialization (varifold): {var_total} iters (per-scale: {var_iters})")
    print(f" Main training (normal cycles): {nc_total} iters (per-scale: {nc_iters})")
    if ft_iters > 0:
        print(f" Fine-tuning: {ft_iters} iters (using {ft_loss_name} loss)")
    print(f" Total: {total_iters} iters")

    return TrainingSchedule(phases=phases, total_iters=total_iters, ode_output_scaling=ode_output_scaling)