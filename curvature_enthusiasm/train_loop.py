import os
import time
from typing import Dict, List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import tqdm
import igl

from .metrics import evaluate_model, compute_geodesic_distmat

from .create_model import create_model
from .create_loss_funcs import (
    setup_losses,
    create_cost_functions,
    create_varifold_loss,
    create_normal_cycle_loss
)
from .compress_source_targets import (
    prepare_initial_training_data,
    init_nc_compressed_info,
    update_varifold_compression,
    update_nc_compression,
)

from .train import (build_optimizer,
    create_training_schedule,
    make_step_fn,
    create_lambda_weights,
    format_kernel_params
)


def run_training(problem_data: Dict, training_data_config: Dict):
    """Main training orchestration."""
    config = training_data_config["config"]

    # Optional geodesic distances (for specific datasets)
    dist_x = None
    if training_data_config.get("dataset_name") in ["FAUST_r", "SCAPE_r"]:
        print("Computing geodesic distances...")
        dist_x = compute_geodesic_distmat(
            problem_data["v_source"],
            problem_data["f_source"],
        )

    # Create model
    print("Creating model...")
    model = create_model(
        ik_template_skeleton=problem_data["ik_template_skeleton"],
        bone_sampling_config=config.bone_sampling,
        node_config=config.node,
        ode_random_key=training_data_config["ode_random_key"],
        quat_random_key=training_data_config["quat_random_key"],
        raw_pred_quat_key=training_data_config["raw_pred_quat_key"],
        nn_dtype=training_data_config["nn_dtype"],
        var_dtype=training_data_config["var_dtype"],
    )

    # Setup losses
    print("Setting up losses...")
    losses = setup_losses(problem_data, config)
    varifold_cost_fn, nc_cost_fn = create_cost_functions(
        losses,
        problem_data["tetra_centres"],
        problem_data["ik_template_skeleton"],
    )

    # Build optimizer and schedule
    print("Building optimizer...")
    optimizer, lr_schedule = build_optimizer(config)
    schedule = create_training_schedule(config, varifold_cost_fn, nc_cost_fn)

    print("Preparing training data...")
    training_data = prepare_initial_training_data(problem_data, config, training_data_config["var_dtype"])

    # Check if we'll ever use NC compression
    nc_iters = list(getattr(config.normal_cycles, "iters_per_lengthscale", [])) or []
    nc_total = int(np.sum(nc_iters)) if nc_iters else 0
    will_use_nc = nc_total > 0

    # Check fine-tuning compression setting
    use_ft = bool(getattr(config.fine_tuning, "use_fine_tuning", False))
    ft_compress_source = bool(getattr(config.fine_tuning, "compress_source", True)) if use_ft else True

    # varifold_sigma = config.varifold.sigmas[-1]
    # varifold_sigma_sph = config.varifold.sigma_sphs[-1]
    ft_compress_target = bool(getattr(config.fine_tuning, "compress_target", True)) if use_ft else True

    # Initialize compression state only if compression is enabled
    if config.compression.compress_source:
        # Only initialize NC compression if we'll actually use it
        if will_use_nc:
            nc_compressed_info = init_nc_compressed_info(problem_data)
        else:
            nc_compressed_info = None

        max_buffer = int(training_data.points.shape[1])
        n_centers = training_data.weights.shape[1]
        print(f" Initial varifold compression: {max_buffer} vertices → {n_centers} centers")
    else:
        nc_compressed_info = None
        print(f" Using full mesh: {training_data.points.shape[1]} vertices")

    # Static data
    n_joints = int(problem_data["ik_template_skeleton"]["joints_positions"].shape[0] - 1)
    bone_ids = jnp.arange(problem_data["ik_template_skeleton"]["bone_edges"].shape[0])
    lambda_w = create_lambda_weights(config.constraints, training_data_config["var_dtype"])

    # Numpy arrays for I/O
    v_src_np = np.asarray(problem_data["v_source"])
    f_src_np = np.asarray(problem_data["f_source"])

    # Evaluation data
    eval_data = (
        problem_data["v_source"][None, :],
        problem_data["f_source"],
        problem_data["v_target"][None, :],
    )

    # Initialize optimizer state
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Flatten model and opt_state for JIT stability
    flat_model, treedef_model = jax.tree_util.tree_flatten(model)
    flat_opt_state, treedef_opt = jax.tree_util.tree_flatten(opt_state)

    # Create step function with captured treedefs
    step_fn = make_step_fn(optimizer, model, opt_state)

    # PRNG key handling
    rng_key = training_data_config["predict_key"]
    if isinstance(rng_key, (list, tuple)):
        rng_key = jr.PRNGKey(rng_key[0])

    # Training loop
    objective_history: List[Tuple[int, float]] = []

    print("\nStarting training...")
    print("=" * 80)

    prev_phase_name = None

    with tqdm.trange(schedule.total_iters, desc="Training", dynamic_ncols=True) as pbar:
        for global_step in pbar:
            # Determine current phase and kernel
            phase_idx, step_in_phase = schedule.get_phase_and_step(global_step)
            current_phase = schedule.phases[phase_idx]

            kernel_idx = current_phase.current_kernel_idx(step_in_phase)
            kernel_params = current_phase.kernel_params[kernel_idx]
            loss_grad_fn = current_phase.loss_grad_fn

            # Only check for NC phase entry if we're actually using NC
            is_nc_phase_entry = (
                    will_use_nc
                    and config.compression.compress_source
                    and prev_phase_name == "varifold"
                    and current_phase.name == "normal_cycles"
                    and step_in_phase == 0
            )

            if is_nc_phase_entry:
                eval_model = jax.tree_util.tree_unflatten(treedef_model, flat_model)
                rng_key, raw_key = jr.split(rng_key)
                step_key = jr.fold_in(raw_key, int(global_step))
                step_key, pred_key = jr.split(step_key)

                metrics = evaluate_model(
                    eval_model,
                    eval_data,
                    problem_data["tetra_centres"],
                    problem_data["corr_x"],
                    problem_data["corr_y"],
                    dist_x,
                    schedule.ode_output_scaling[global_step],
                    pred_key,
                )

                deformed = metrics["y_pred"]

                centres, weights = update_nc_compression(
                    deformed, problem_data["f_source"], nc_compressed_info, config
                )

                centres = centres.astype(training_data.points.dtype)
                weights = weights.astype(training_data.weights.dtype)

                training_data = eqx.tree_at(
                    lambda d: (d.points, d.weights),
                    training_data,
                    (centres, weights),
                )
                print(f" Switched to NC compression: {centres.shape[1]} centres")

            # Handle phase transitions
            if current_phase.name != prev_phase_name:
                print(f"\n{'=' * 80}")
                print(f"→ Starting phase: {current_phase.name.upper()}")
                print(f"{'=' * 80}")

                if current_phase.name == "fine_tuning":
                    lambda_w = create_lambda_weights(
                        config.fine_tuning, training_data_config["var_dtype"]
                    )
                    print("Updated lambda weights for fine-tuning")

                    # Switch to appropriate mesh representation for fine-tuning
                    if not ft_compress_source:
                        # Update to full mesh while maintaining structure
                        full_points = problem_data["v_source"][None, :]
                        full_faces = problem_data["f_source"][None, :]

                        training_data = eqx.tree_at(
                            lambda d: (d.points, d.faces, d.weights),
                            training_data,
                            (full_points, full_faces, None),
                        )
                        print(f" Switched to full mesh for fine-tuning: {training_data.points.shape[1]} vertices")

                        # Recreate varifold loss with uncompressed target if needed
                        if not ft_compress_target and config.compression.compress_target:

                            losses['normal_cycle_loss'] = create_normal_cycle_loss(problem_data['f_source'],
                                                                                   problem_data['v_target'],
                                                                                   problem_data['f_target'],
                                                                                   problem_data['has_boundary'],
                                                                                   compress_target=False,
                                                                                   smallest_nc_sigma=None,
                                                                                   compressed_nc_target_size=None)

                            # Recreate the varifold cost function with the new loss
                            _, nc_cost_fn = create_cost_functions(
                                losses,
                                problem_data["tetra_centres"],
                                problem_data["ik_template_skeleton"],
                            )
                            # Update the gradient function for the current phase
                            loss_grad_fn = eqx.filter_value_and_grad(nc_cost_fn)

                prev_phase_name = current_phase.name

            if step_in_phase in current_phase.transition_steps:
                print(f"\n → Kernel scale transition: {kernel_idx}")

            # Training step
            rng_key, raw_key = jr.split(rng_key)
            step_key = jr.fold_in(raw_key, int(global_step))

            ode_scale = schedule.ode_output_scaling[global_step]
            loss_kwargs = {"bone_ids": jnp.arange(problem_data["ik_template_skeleton"]["bone_edges"].shape[0]),
                           "n_joints": int(problem_data["ik_template_skeleton"]["joints_positions"].shape[0] - 1)}

            loss, flat_model, flat_opt_state = step_fn(
                flat_model,
                flat_opt_state,
                training_data,
                kernel_params,
                ode_scale,
                lambda_w,
                step_key,
                loss_grad_fn,
                loss_kwargs,
            )

            loss_val = float(loss)
            objective_history.append((global_step, loss_val))
            kernel_str = format_kernel_params(kernel_params, current_phase.name)
            pbar.set_postfix_str(f"loss={loss_val:.3e} | {current_phase.name}[{kernel_str}]")

            # Periodic evaluation and checkpointing
            should_eval = (
                    (global_step > 0 and global_step % config.optimization.eval_per_epochs == 0)
                    or global_step == schedule.total_iters - 1
            )

            if should_eval:
                eval_model = jax.tree_util.tree_unflatten(treedef_model, flat_model)

                rng_key, eval_key = jr.split(rng_key)
                metrics = evaluate_model(
                    eval_model,
                    eval_data,
                    problem_data["tetra_centres"],
                    problem_data["corr_x"],
                    problem_data["corr_y"],
                    dist_x,
                    ode_scale,
                    eval_key,
                )

                os.makedirs("tmp", exist_ok=True)
                deformed_source = np.asarray(metrics["y_pred"])
                igl.writeOBJ(f"tmp/ode_pred_{global_step:04d}.obj", deformed_source, f_src_np)

                msg = (
                    f"\n{'=' * 80}\n"
                    f"Evaluation at step {global_step}:\n"
                    f"MSE: {float(metrics['mse']):.3e}\t"
                    f"Chamfer: {float(metrics['chamfer']):.3e}\t"
                    f"Score: {float(metrics['score']):.1f}%\t"
                )

                if metrics.get("geo_err") is not None:
                    msg += f" Geo Error: {float(metrics['geo_err']):.2e}\n"
                msg += f" End Samples Diff: {float(metrics['end_samples_diff_cost']):.3e}"
                msg += f" Bone Traj: {float(metrics['bone_traj_cost']):.3e}\n"
                msg += f"{'=' * 80}"
                print(msg)

                os.makedirs("checkpoints", exist_ok=True)
                checkpoint_path = f"checkpoints/step_{global_step:04d}.eqx"
                eqx.tree_serialise_leaves(checkpoint_path, eval_model)

                # Only update compression if we're using it in the current phase
                should_compress = (
                        (current_phase.name != "fine_tuning" and config.compression.compress_source)
                        or (current_phase.name == "fine_tuning" and ft_compress_source)
                )

                if should_compress:
                    # Only use NC compression if we're in NC phase and it's initialized
                    if current_phase.name == "normal_cycles" and nc_compressed_info is not None:
                        centres, weights = update_nc_compression(
                            jnp.asarray(deformed_source), problem_data["f_source"], nc_compressed_info, config
                        )
                        centres = centres.astype(training_data.points.dtype)
                        weights = weights.astype(training_data.weights.dtype)

                        training_data = eqx.tree_at(
                            lambda d: (d.points, d.weights),
                            training_data,
                            (centres, weights),
                        )
                    elif current_phase.name in ["varifold", "fine_tuning"]:
                        # Use varifold compression for varifold phase and fine-tuning (only when ft_compress=True)
                        verts, faces, weights = update_varifold_compression(
                            jnp.asarray(deformed_source), v_src_np, f_src_np, max_buffer, config
                        )
                        training_data = eqx.tree_at(
                            lambda d: (d.points, d.faces, d.weights),
                            training_data,
                            (verts, faces, weights),
                        )

    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)

    final_model = jax.tree_util.tree_unflatten(treedef_model, flat_model)

    print(f"\nSaving final model to: {training_data_config['model_fn']}")
    eqx.tree_serialise_leaves(training_data_config["model_fn"], final_model)
    time.sleep(1)

    print("\nRunning final evaluation...")
    rng_key, final_key = jr.split(rng_key)
    final_metrics = evaluate_model(
        final_model,
        eval_data,
        problem_data["tetra_centres"],
        problem_data["corr_x"],
        problem_data["corr_y"],
        dist_x,
        schedule.ode_output_scaling[-1],
        final_key,
    )

    igl.writeOBJ(
        "tmp/ode_pred_final.obj",
        np.asarray(final_metrics["y_pred"]),
        f_src_np,
    )

    print("\nObjective trajectory (sampled every 25 steps):")
    for step, obj in objective_history[::25]:
        print(f" Step {step:5d}: {obj:.3e}")

    msg = (
        f"\n{'=' * 80}\n"
        f"FINAL RESULTS:\n"
        f"MSE: {final_metrics['mse']:.3e}\t"
        f"Chamfer: {final_metrics['chamfer']:.3e}\t"
        f"Score: {final_metrics['score']:.1f}%\t"
    )
    if final_metrics.get("geo_err") is not None:
        msg += f" Geo Error: {final_metrics['geo_err']:.2e}\n"
    msg += f" Bone Traj: {final_metrics['bone_traj_cost']:.3e}\n"
    msg += f"{'=' * 80}"
    print(msg)

    return final_model