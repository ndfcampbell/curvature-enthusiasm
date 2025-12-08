from typing import Callable
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array

@jax.custom_vjp
def reverse_gradient(x):
    return x

def _reverse_gradient_fwd(x):
    # Returns primal output and residuals to be used in backward pass by f_bwd.
    return x, []

def _reverse_gradient_bwd(res, grad):
    # res should be empty
    return (-1.0 * grad,)

reverse_gradient.defvjp(_reverse_gradient_fwd, _reverse_gradient_bwd)


class MDMM(eqx.Module):
    raw_lm_weights: Float[Array, "..."]  # shape depends on lm_init
    damping: float
    scale: float
    reduction: Callable
    target_value: Float[Array, ""] = eqx.field(static=True)

    def __init__(
            self,
            lm_init: Float[Array, "..."],
            damping: float = 1.0,
            scale: float = 1.0,
            target_values: float | Array = 1e-3,
            reduction: Callable = jnp.mean,
    ):
        self.raw_lm_weights = lm_init
        self.damping = damping
        self.scale = scale
        self.reduction = reduction
        self.target_value = jnp.array(target_values)

    @property
    def lambda_weights(self) -> Float[Array, "..."]:
        return reverse_gradient(self.raw_lm_weights)

    def __call__(
            self,
            idxs: Array,  # indices can be int or bool arrays
            fn_value: Float[Array, "..."],
    ) -> Float[Array, "..."]:
        inf = jnp.clip(fn_value, 0.0, self.target_value) - fn_value
        l_term = self.lambda_weights * inf
        damp_term = self.damping * inf ** 2 / 2
        return self.scale * (l_term + damp_term)