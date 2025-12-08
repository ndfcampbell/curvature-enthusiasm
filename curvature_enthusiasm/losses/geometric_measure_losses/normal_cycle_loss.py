from typing import Optional,Dict, Callable

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Int


from torch2jax import torch2jax_with_vjp
from .backends.torch import calc_normal_cycle_loss_torch
from .brdy_funcs_jax import calc_parts_and_weights, calc_parts_and_weights_brdy

class Keops_Normal_Cycles_Loss(eqx.Module):
    template_faces: Int[Array, "F 3"] = eqx.field(static=True)
    template_struct: Dict[str, Array] = eqx.field(static=True)
    target_centres: Float[Array, "M D"] = eqx.field(static=True)
    target_weights: Float[Array, "M F"] = eqx.field(static=True)

    extract: Callable = eqx.field(static=True)  # just the function
    dtype: jnp.dtype

    def __init__(
        self,
        template_faces,
        template_struct,
        target_centres,
        target_weights,
        *,
        has_boundary: bool = False,
        dtype=jnp.float32,
    ):
        self.dtype = dtype
        self.template_faces = template_faces.astype(jnp.int32)
        self.template_struct = template_struct
        self.target_centres = target_centres.astype(dtype)
        self.target_weights = target_weights.astype(dtype)

        # choose once
        self.extract = calc_parts_and_weights_brdy if has_boundary else calc_parts_and_weights


    def __call__(
        self,
        source_points: Float[Array, "V D"],
        gamma: Float[Array, ""],
        *,
        # If provided, these weights are used as-is and frozen (no grads).
        # If None, we compute weights from (v1,v2,v3) and allow grads to flow to x_centers via normals/weights.
        template_weights: Optional[Array] = None,        # (N,1) float, optional
        freeze_kernel: bool = True,               # if True, no grads w.r.t kernel_params
        use_torch_vjp: bool = True,
        depth: int = 2,
    )-> Float[Array, "1"] :


        if template_weights is None:
            # computed weights; allow gradients (do not stop_gradient)
            x_centers, x_weights_auto = self.extract(
                source_points, self.template_faces, self.template_struct
            )
            x_w = x_weights_auto
            freeze_x_weights = False
        else:
            # external weights must be frozen
            x_centers = source_points
            x_w = jax.lax.stop_gradient(jnp.asarray(template_weights, self.dtype))
            freeze_x_weights = True

        nondiff = [2, 3]  # targets are always non-diff
        if freeze_x_weights:
            nondiff.append(1)  # freeze x_weights if provided externally
        if freeze_kernel:
            nondiff.append(4)  # freeze kernel params if requested

        # sort nondiff_argnums
        nondiff_argnums = tuple(sorted(nondiff))



        xy_cost = torch2jax_with_vjp(
            calc_normal_cycle_loss_torch,
            jax.ShapeDtypeStruct(x_centers.shape, x_centers.dtype),
            jax.ShapeDtypeStruct(x_w.shape, x_w.dtype),
            jax.ShapeDtypeStruct(self.target_centres.shape, self.target_centres.dtype),
            jax.ShapeDtypeStruct(self.target_weights.shape, self.target_weights.dtype),
            jax.ShapeDtypeStruct(gamma.shape, gamma.dtype),
            output_shapes=jax.ShapeDtypeStruct((1,), x_centers.dtype),
            depth=depth,
            use_torch_vjp=use_torch_vjp,
            nondiff_argnums=nondiff_argnums,
            # nondiff_argnums=(2, 3, 4)
        )(x_centers,
          x_w,
          self.target_centres,
          self.target_weights,
          gamma)

        return xy_cost

