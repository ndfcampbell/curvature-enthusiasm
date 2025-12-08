from typing import Optional, Union, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
from torch2jax import dtype_t2j, torch2jax_with_vjp

from curvature_enthusiasm.utils import default_floating_dtype
from .backends.torch import calc_varifold_loss_torch, calc_varifold_loss_torch_old_style



# def jp_safe_norm(
#         x: jax.Array,
#         axis: Optional[Union[Tuple[int, ...], int]] = None,
#         keepdims: bool = False,
#         eps: float = 1e-12
# ) -> jax.Array:
#     """
#     Calculates a norm that's safe for gradients at x=0.
#
#     Uses the approach: sqrt(sum(x^2) + eps) which has well-defined gradients.
#     """
#     squared_norm = jnp.sum(x * x, axis=axis, keepdims=keepdims)
#     return jnp.sqrt(squared_norm + eps)


def tri_extr(
        pts: Float[jax.Array, "... V 3"],
        simplices: Int[jax.Array, "... F 3"],
) -> Tuple[
    Float[jax.Array, "... F 3"],
    Float[jax.Array, "... F 3"],
    Float[jax.Array, "... F 3"]
]:
    """
    Extract triangle vertices from mesh.

    Args:
        pts: Vertex positions, shape [V, 3] or [B, V, 3]
        simplices: Triangle vertex indices, shape [F, 3] or [B, F, 3]

    Returns:
        Tuple of (v1, v2, v3) where each has shape [F, 3] or [B, F, 3]

    Examples:
        >>> pts = jnp.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        >>> faces = jnp.array([[0, 1, 2]])
        >>> v1, v2, v3 = tri_extr(pts, faces)
        >>> v1.shape
        (1, 3)
    """
    if pts.ndim == 2:  # Unbatched: [V, 3], [F, 3]
        return (
            pts[simplices[:, 0]],
            pts[simplices[:, 1]],
            pts[simplices[:, 2]]
        )

    # Batched: [B, V, 3], [B, F, 3]
    B, F = simplices.shape[:2]

    # Create batch indices [B, 1, 1] for broadcasting
    batch_idx = jnp.arange(B)[:, None, None]

    v1 = pts[batch_idx, simplices[:, :, 0]]  # [B, F, 3]
    v2 = pts[batch_idx, simplices[:, :, 1]]
    v3 = pts[batch_idx, simplices[:, :, 2]]

    return v1, v2, v3


# def extract_jax(
#         verts: Tuple[
#             Float[jax.Array, "... F 3"],
#             Float[jax.Array, "... F 3"],
#             Float[jax.Array, "... F 3"]
#         ]
# ) -> Tuple[
#     Float[jax.Array, "... F 3"],  # centres
#     Float[jax.Array, "... F 3"],  # unit_normals
#     Float[jax.Array, "... F 1"]  # areas
# ]:
#     """
#     Compute triangle properties from vertices.
#
#     Args:
#         verts: Tuple of (v1, v2, v3) vertex positions, each shape [..., F, 3]
#
#     Returns:
#         centres: Triangle centroids, shape [..., F, 3]
#         unit_normals: Unit normal vectors, shape [..., F, 3]
#         areas: Triangle areas, shape [..., F, 1]
#
#     Examples:
#         >>> v1 = jnp.array([[0., 0., 0.]])
#         >>> v2 = jnp.array([[1., 0., 0.]])
#         >>> v3 = jnp.array([[0., 1., 0.]])
#         >>> c, n, a = extract_jax((v1, v2, v3))
#         >>> float(a[0, 0])  # area = 0.5 for unit right triangle
#         0.5
#     """
#     v1, v2, v3 = verts
#
#     # Centroid
#     centres = (v1 + v2 + v3) / 3.0
#
#     # Cross product for normal and area
#     u = v2 - v1
#     v = v3 - v1
#     cross = jnp.cross(u, v, axis=-1)  # [..., F, 3]
#
#     # Area is half the magnitude of cross product
#     # Use safe norm to avoid gradient issues at zero
#     cross_norm = jp_safe_norm(cross, axis=-1)  # [..., F]
#     areas = 0.5 * cross_norm
#
#     # Normalize (handle degenerate triangles)
#     eps = jnp.finfo(cross.dtype).eps
#     unit_normals = cross / jnp.maximum(cross_norm[..., None], eps)
#
#     return centres, unit_normals, areas[..., None]

# def extract_jax(
#         verts: Tuple[
#             Float[jax.Array, "... F 3"],
#             Float[jax.Array, "... F 3"],
#             Float[jax.Array, "... F 3"]
#         ]
# ) -> Tuple[
#     Float[jax.Array, "... F 3"],  # centres
#     Float[jax.Array, "... F 3"],  # unit_normals
#     Float[jax.Array, "... F 1"]  # areas
# ]:
#     v1, v2, v3 = verts
#
#     # Centroid
#     centres = (v1 + v2 + v3) / 3.0
#
#     # Cross product for normal and area
#     u = v2 - v1
#     v = v3 - v1
#     cross = jnp.cross(u, v, axis=-1)  # [..., F, 3]
#
#     # Use safe norm with small epsilon
#     eps = jnp.sqrt(jnp.finfo(cross.dtype).eps)
#     cross_norm = jp_safe_norm(cross, axis=-1, keepdims=True)  # [..., F, 1]
#     areas = 0.5 * cross_norm
#
#     # Normalize - division by (norm + eps) is safe
#     unit_normals = cross / cross_norm
#
#     return centres, unit_normals, areas



def extract_jax(verts: Tuple[jax.Array, jax.Array, jax.Array]):
    """Compute triangle centres, unit normals, and areas.

    - Degenerate triangles have exactly zero area.
    - Gradients are smooth and finite (no NaNs).
    - Uses dtype-aware smoothing and stable normalization.
    """
    v1, v2, v3 = verts

    # Centroids
    v_centres = (v1 + v2 + v3) / 3.0

    # Edge vectors and cross product
    u = v2 - v1
    v = v3 - v1
    cross = jnp.cross(u, v, axis=-1)  # [..., F, 3]

    # Squared norm of cross product
    sq_norm = jnp.sum(cross * cross, axis=-1, keepdims=True)  # [..., F, 1]

    # Smoothing epsilon for stable gradients
    # 1e-12 gives a minimum norm of ~1e-6 for float32, large enough to avoid instability
    eps_smooth = 1e-12
    safe_sq_norm = sq_norm + eps_smooth
    safe_norm = jnp.sqrt(safe_sq_norm)  # never zero

    # Base areas (smooth, differentiable everywhere)
    base_areas = 0.5 * safe_norm

    # Degeneracy threshold (controls what counts as "zero area")
    # eps_degenerate = 1e-12
    # is_degenerate = sq_norm < eps_degenerate
    #
    # # Final areas: exactly 0 for degenerate triangles
    # areas = jnp.where(is_degenerate, 0.0, base_areas)
    #
    # # Unit normals: safe normalization + fallback
    # inv_norm = jax.lax.rsqrt(safe_sq_norm)   # stable 1 / sqrt(x)
    # unit_normals_raw = cross * inv_norm      # normalized cross product
    #
    # fallback_normal = jnp.array([0.0, 0.0, 1.0], dtype=cross.dtype)
    # unit_normals = jnp.where(is_degenerate, fallback_normal, unit_normals_raw)

    inv_norm = jax.lax.rsqrt(safe_sq_norm)
    unit_normals = cross * inv_norm
    areas = base_areas

    return v_centres, unit_normals, areas

# def extract_jax(verts):
#     v1, v2, v3 = verts
#     v_centres = (v1 + v2 + v3) / 3.0
#
#     u = v2 - v1
#     v = v3 - v1
#     cross = jnp.cross(u, v, axis=-1)  # [..., F, 3]
#
#     sq_norm = jnp.sum(cross * cross, axis=-1, keepdims=True)  # [..., F, 1]
#
#     # Smooth, gradient-safe norm
#     eps_area = 1e-12
#     cross_norm = jnp.sqrt(sq_norm + eps_area)
#
#     # Areas: small but non-zero for degenerate triangles
#     areas = 0.5 * cross_norm
#
#     # Clamp normalization to avoid huge scaling of normals
#     eps_normal = 1e-6
#     safe_norm = jnp.maximum(cross_norm, eps_normal)
#     unit_normals = cross / safe_norm
#
#     return v_centres, unit_normals, areas

def extract_varifold_properties(v_target, f_target):
    """Extract properties needed for varifold loss."""
    tri_v1, tri_v2, tri_v3 = tri_extr(v_target, f_target)
    y_centres, y_normals, y_weights = extract_jax([tri_v1, tri_v2, tri_v3])

    return {
        'w_centres': y_centres,
        'w_normals':y_normals,
        'w_weights': y_weights
    }


def get_triangle_vertices(vertices, faces):
    v1 = vertices[:, faces[:, 0]]  # (n,k,3)
    v2 = vertices[:, faces[:, 1]]
    v3 = vertices[:, faces[:, 2]]
    return v1, v2, v3


class Keops_Varifold_Loss(eqx.Module):
    """
    Unified Varifold loss:
      - target_* are static (fixed reference)
      - can accept external x_weights (fixed, no grads) or compute internally (grads allowed)
      - can freeze or learn kernel_params
    """
    target_centers: Array = eqx.field(static=True)  # (M,3)
    target_normals: Array = eqx.field(static=True)  # (M,3)
    target_weights: Array = eqx.field(static=True)  # (M,1)
    dtype: jnp.dtype

    def __init__(self, const_vari_props, dtype=None):
        self.dtype = default_floating_dtype() if dtype is None else dtype
        self.target_centers = jnp.asarray(const_vari_props["w_centres"], self.dtype)
        self.target_normals = jnp.asarray(const_vari_props["w_normals"], self.dtype)
        self.target_weights = jnp.asarray(const_vari_props["w_weights"], self.dtype)

    def __call__(
        self,
        template_vertices: Array,                 # (V,3)
        template_faces: Array,                    # (F,3) int
        kernel_params: Array,                     # (2,)
        *,
        # If provided, these weights are used as-is and frozen (no grads).
        # If None, we compute weights from (v1,v2,v3) and allow grads to flow to x_centers via normals/weights.
        template_weights: Optional[Array] = None,        # (N,1) float, optional
        freeze_kernel: bool = True,               # if True, no grads w.r.t kernel_params
        use_torch_vjp: bool = True,
        depth: int = 2,
    ) -> Array:                                   # (1,1)
        # 1) Face barycenters / normals / (optional) weights
        v1, v2, v3 = tri_extr(template_vertices, template_faces)
        x_centers, x_normals, x_weights_auto = extract_jax([v1, v2, v3])   # shapes: (N,3), (N,3), (N,1)

        # Choose which weights to pass to Torch:
        if template_weights is None:
            # computed weights; allow gradients (do not stop_gradient)
            x_w = x_weights_auto
            freeze_x_weights = False
        else:
            # external weights must be frozen
            x_w = jax.lax.stop_gradient(jnp.asarray(template_weights, self.dtype))
            freeze_x_weights = True

        # 2) Build nondiff_argnums based on what is truly static
        # Torch argument order to wrapped function:
        # (0) x_centers, (1) x_normals, (2) x_weights, (3) target_centers,
        # (4) target_normals, (5) target_weights, (6) kernel_params
        nondiff = [3, 4, 5]                      # targets are always non-diff
        if freeze_x_weights:
            nondiff.append(2)                    # freeze x_weights if provided externally
        if freeze_kernel:
            nondiff.append(6)                    # freeze kernel params if requested
        nondiff_argnums = tuple(nondiff)

        # 3) Call Torch/KeOps with a VJP; let grads flow w.r.t. args not in nondiff_argnums
        varifold_cost = torch2jax_with_vjp(
            calc_varifold_loss_torch,
            # Input specs
            jax.ShapeDtypeStruct(x_centers.shape,     dtype_t2j(x_centers.dtype)),
            jax.ShapeDtypeStruct(x_normals.shape,     dtype_t2j(x_normals.dtype)),
            jax.ShapeDtypeStruct(x_w.shape,           dtype_t2j(x_w.dtype)),
            jax.ShapeDtypeStruct(self.target_centers.shape, dtype_t2j(self.target_centers.dtype)),
            jax.ShapeDtypeStruct(self.target_normals.shape, dtype_t2j(self.target_normals.dtype)),
            jax.ShapeDtypeStruct(self.target_weights.shape, dtype_t2j(self.target_weights.dtype)),
            jax.ShapeDtypeStruct(kernel_params.shape, dtype_t2j(kernel_params.dtype)),
            # Output spec
            output_shapes=jax.ShapeDtypeStruct((1, 1), dtype_t2j(x_centers.dtype)),
            depth=depth,
            use_torch_vjp=use_torch_vjp,
            nondiff_argnums=nondiff_argnums,
        )(
            x_centers,
            x_normals,
            x_w,
            self.target_centers,
            self.target_normals,
            self.target_weights,
            kernel_params,
        )

        return varifold_cost

    # def __call__(
    #     self,
    #     template_vertices: Array,                 # (V,3)
    #     template_faces: Array,                    # (F,3) int
    #     kernel_params: Array,                     # (2,)
    #     *,
    #     # If provided, these weights are used as-is and frozen (no grads).
    #     # If None, we compute weights from (v1,v2,v3) and allow grads to flow to x_centers via normals/weights.
    #     template_weights: Optional[Array] = None,        # (N,1) float, optional
    #     freeze_kernel: bool = True,               # if True, no grads w.r.t kernel_params
    #     use_torch_vjp: bool = True,
    #     depth: int = 2,
    # ) -> Array:                                   # (1,1)
    #
    #     # 3) Call Torch/KeOps with a VJP; let grads flow w.r.t. args not in nondiff_argnums
    #     varifold_cost = torch2jax_with_vjp(
    #         calc_varifold_loss_torch_old_style,
    #         # Input specs
    #         jax.ShapeDtypeStruct(template_vertices.shape,     dtype_t2j(template_vertices.dtype)),
    #         jax.ShapeDtypeStruct(template_faces.shape,     dtype_t2j(template_faces.dtype)),
    #         jax.ShapeDtypeStruct(self.target_centers.shape, dtype_t2j(self.target_centers.dtype)),
    #         jax.ShapeDtypeStruct(self.target_normals.shape, dtype_t2j(self.target_normals.dtype)),
    #         jax.ShapeDtypeStruct(self.target_weights.shape, dtype_t2j(self.target_weights.dtype)),
    #         jax.ShapeDtypeStruct(kernel_params.shape, dtype_t2j(kernel_params.dtype)),
    #         # Output spec
    #         output_shapes=jax.ShapeDtypeStruct((1, 1), dtype_t2j(template_vertices.dtype)),
    #         depth=depth,
    #         use_torch_vjp=use_torch_vjp,
    #         nondiff_argnums=(1,2,3,4,5),
    #     )(
    #         template_vertices,
    #         template_faces,
    #         self.target_centers,
    #         self.target_normals,
    #         self.target_weights,
    #         kernel_params,
    #     )
    #
    #     return varifold_cost


