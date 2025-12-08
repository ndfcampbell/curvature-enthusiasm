from typing import Any, Callable, Dict, List, Tuple
import equinox as eqx
import jax
import jax.numpy as jnp
import optax

def make_step_fn(
    optimizer: optax.GradientTransformation,
    model_init,
    opt_state_init,
):
    """Create JIT-compiled training step with captured treedefs."""
    _, treedef_model = jax.tree_util.tree_flatten(model_init)
    _, treedef_opt = jax.tree_util.tree_flatten(opt_state_init)


    @eqx.filter_jit
    def step_flat(
        flat_model: List[jax.Array],
        flat_opt_state: List[Any],
        data,
        kernel_params: Dict[str, jnp.ndarray],
        ode_scale: jax.Array,
        lambda_w: Dict[str, jax.Array],
        key: jax.Array,
        loss_grad_fn: Callable,
        loss_kwargs: Dict[str, Any],
    ) -> Tuple[jax.Array, List[jax.Array], List[Any]]:

        model = jax.tree_util.tree_unflatten(treedef_model, flat_model)
        opt_state = jax.tree_util.tree_unflatten(treedef_opt, flat_opt_state)

        loss, grads = loss_grad_fn(
        model, data, ode_scale, kernel_params, lambda_w, key, **loss_kwargs
        )
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        return (
            loss,
            jax.tree_util.tree_leaves(model),
            jax.tree_util.tree_leaves(opt_state),
        )

    return step_flat