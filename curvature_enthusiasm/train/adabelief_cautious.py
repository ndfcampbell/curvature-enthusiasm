from typing import Any, Callable
from typing import NamedTuple, Optional, Union

import chex
import jax
import jax.numpy as jnp
from optax import tree_utils as otu
from optax._src import base
from optax._src import numerics
from optax._src.base import GradientTransformation, ScalarOrSchedule, Params
from optax._src.combine import chain
from optax._src.transform import scale_by_learning_rate, add_decayed_weights


class ScaleByBeliefState(NamedTuple):
  """State for the rescaling by AdaBelief algorithm."""

  count: chex.Array  # shape=(), dtype=jnp.int32.
  mu: base.Updates
  nu: base.Updates


def scale_by_belief_cautious(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-16,
    eps_root: float = 1e-16,
    *,
    nesterov: bool = False,
) -> base.GradientTransformation:
  """Rescale updates according to the AdaBelief algorithm.

  See :func:`optax.adabelief` for more details.

  Args:
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of variance of grads.
    eps: Term added to the denominator to improve numerical stability.
    eps_root: Term added to the second moment of the prediction error to improve
      numerical stability. If backpropagating gradients through the gradient
      transformation (e.g. for meta-learning), this must be non-zero.
    nesterov: Whether to use Nesterov momentum.

  Returns:
    A :class:`optax.GradientTransformation` object.
  """

  def init_fn(params):
    mu = otu.tree_zeros_like(params)  # First moment
    s = otu.tree_zeros_like(params)  # Second Central moment
    return ScaleByBeliefState(count=jnp.zeros([], jnp.int32), mu=mu, nu=s)

  def update_fn(updates, state, params=None):
    del params
    mu = otu.tree_update_moment(updates, state.mu, b1, 1)
    prediction_error = otu.tree_sub(updates, mu)
    nu = otu.tree_update_moment_per_elem_norm(prediction_error, state.nu, b2, 2)
    nu = jax.tree.map(lambda v: v + eps_root, nu)
    count_inc = numerics.safe_increment(state.count)
    if nesterov:
      mu_hat = jax.tree.map(
          lambda m, g: b1 * m + (1 - b1) * g,
          otu.tree_bias_correction(
              mu, b1, numerics.safe_increment(count_inc)),
          otu.tree_bias_correction(updates, b1, count_inc))
    else:
      mu_hat = otu.tree_bias_correction(mu, b1, count_inc)
    nu_hat = otu.tree_bias_correction(nu, b2, count_inc)
    u_t = jax.tree.map(
        lambda m, v: None if m is None else m / (jnp.sqrt(v) + eps),
        mu_hat,
        nu_hat,
        is_leaf=lambda x: x is None,
    )

    def apply_mask(u, g):
        if u is None or g is None:
            return None
        # Compute alignment mask
        mask = (u * g) > 0
        mask = mask.astype(g.dtype)
        mask /= jnp.clip(jnp.mean(mask), min=1e-3)
        return u * mask

    updates = jax.tree.map(apply_mask, u_t, updates, is_leaf=lambda x: x is None)

    return updates, ScaleByBeliefState(count=count_inc, mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)

def adabelief_cautious(learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-16,
    eps_root: float = 1e-16,
    weight_decay: float = 1e-4,
    mask: Optional[Union[Any, Callable[[Params], Any]]] = None) -> GradientTransformation:

    return chain(
        scale_by_belief_cautious(b1=b1, b2=b2, eps=eps, eps_root=eps_root),
        add_decayed_weights(weight_decay, mask),
        scale_by_learning_rate(learning_rate),
    )