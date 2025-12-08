import math
from typing import Optional, Tuple

import torch
import numpy as np
import igl

from ..keops.compression_funcs_keops import Var_met_reduce, Var_met_reduce_batched  # expects same signatures
from .loss_funcs_torch import calc_varifold_loss_torch

from pykeops.torch import KernelSolve

# ------------------------- utils -------------------------

def _eye(n: int, like: torch.Tensor) -> torch.Tensor:
    return torch.eye(n, device=like.device, dtype=like.dtype)

def _chol_solve(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    L = torch.linalg.cholesky(A)
    return torch.cholesky_solve(B, L)

def _chol_inverse(A: torch.Tensor) -> torch.Tensor:
    L = torch.linalg.cholesky(A)
    return torch.cholesky_inverse(L)

def shuffle_indices(n: int, *, seed: Optional[int] = None, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    # Avoid global manual_seed; use a local generator if seed provided.
    if generator is None and seed is not None:
        generator = torch.Generator(device='cpu').manual_seed(seed)
    return torch.randperm(n, generator=generator)


# ------------------------- core ops -------------------------


# @torch.no_grad()
# def calc_approx_ls_batched(
#     X_blocks: torch.Tensor,           # [B, m, d]
#     lamb: float | torch.Tensor,
#     kernel_params: torch.Tensor,      # [2]
# ) -> torch.Tensor:                    # [B, m]
#     B, m, d = X_blocks.shape
#     assert d % 2 == 0, "feature dim must be even (split into halves)."
#
#     device = X_blocks.device
#     dtype = X_blocks.dtype
#
#     # Split into position + direction halves
#     x, xn = torch.split(X_blocks, d // 2, dim=2)
#     x  = x.contiguous()
#     xn = xn.contiguous()
#
#     # Batched identity: [B, m, m] – make it contiguous too
#     b = torch.eye(m, device=device, dtype=dtype).unsqueeze(0).expand(B, m, m).contiguous()
#
#     gamma     = kernel_params[0]
#     gamma_sph = kernel_params[1]
#
#     # KeOps batched kernel: [B, m, m]
#     K_S = Var_met_reduce_batched(x, x, xn, xn, b, gamma, gamma_sph)
#
#     # Regularisation
#     if not torch.is_tensor(lamb):
#         lamb = torch.tensor(lamb, device=device, dtype=dtype)
#     lamb = lamb.view(1, 1, 1)
#
#     eye = torch.eye(m, device=device, dtype=dtype).view(1, m, m)
#     A = K_S + lamb * eye  # [B, m, m]
#
#     # Batched Cholesky solve
#     L = torch.linalg.cholesky(A)  # [B, m, m]
#
#     I = torch.eye(m, device=device, dtype=dtype).view(1, m, m).expand_as(A)
#     Ainv = torch.cholesky_solve(I, L)  # [B, m, m]
#
#     # diag(K_S A^{-1}) per block
#     ls = (K_S * Ainv.transpose(-1, -2)).sum(dim=2)  # [B, m]
#
#     return ls


# @torch.no_grad()
# def calc_varifold_RLS(
#     X: torch.Tensor,
#     lambda_: float,
#     sample_size: int,
#     kernel_params: torch.Tensor,
#     *,
#     seed: Optional[int] = None,
#     blocks_per_batch: int = 128,   # tune this to trade speed vs memory
# ) -> torch.Tensor:
#     N = X.size(0)
#     if N == 0:
#         return torch.zeros(0, device=X.device, dtype=X.dtype)
#
#     m = max(1, min(sample_size, N))
#     device = X.device
#
#     # Shuffle indices and X once
#     perm = shuffle_indices(N, seed=seed).to(device=device)
#     X_shuffled = X[perm]
#
#     n_full = N // m
#     remainder = N - n_full * m
#
#     scores_permuted = torch.empty(N, device=device, dtype=X.dtype)
#
#     # Process full blocks in batches
#     if n_full > 0:
#         B = blocks_per_batch
#         block_start = 0
#         while block_start < n_full:
#             block_end = min(block_start + B, n_full)
#             num_blocks = block_end - block_start
#
#             start_idx = block_start * m
#             end_idx = block_end * m
#
#             X_blocks = X_shuffled[start_idx:end_idx].reshape(num_blocks, m, -1)  # [B, m, d]
#             ls_blocks = calc_approx_ls_batched(X_blocks, lambda_, kernel_params) # [B, m]
#
#             scores_permuted[start_idx:end_idx] = ls_blocks.reshape(-1)
#
#             block_start = block_end
#
#     # Remainder block (smaller than m; just use scalar version)
#     if remainder > 0:
#         start = n_full * m
#         block = X_shuffled[start:]
#         scores_permuted[start:] = calc_approx_ls(block, lambda_, kernel_params)
#
#     # Unshuffle back to original order
#     scores = torch.empty_like(scores_permuted)
#     scores[perm] = scores_permuted
#     return scores

@torch.no_grad()
def calc_varifold_RLS(
        X: torch.Tensor,
        lambda_: float,
        sample_size: int,
        kernel_params: torch.Tensor,
        *,
        seed: Optional[int] = None,
        blocks_per_batch: int = 128,
) -> torch.Tensor:
    N = X.size(0)
    if N == 0:
        return torch.zeros(0, device=X.device, dtype=X.dtype)

    m = max(1, min(sample_size, N))
    device = X.device

    # 1. Create Permutation (Indices only)
    perm = shuffle_indices(N, seed=seed).to(device=device)

    # 2. Prepare Output
    # We can write directly into the final tensor using scatter logic
    # or filling a permuted buffer. Filling a permuted buffer then scattering
    # is usually faster for contiguous writes in the loop.
    scores_permuted = torch.empty(N, device=device, dtype=X.dtype)

    n_full = N // m
    remainder = N - n_full * m

    # 3. Process Full Blocks
    if n_full > 0:
        block_start = 0
        while block_start < n_full:
            block_end = min(block_start + blocks_per_batch, n_full)
            num_blocks = block_end - block_start

            # A. Calculate indices for this large batch
            batch_start_idx = block_start * m
            batch_end_idx = block_end * m

            # B. Get the permuted indices for this batch
            batch_perm_indices = perm[batch_start_idx:batch_end_idx]

            # C. Slice X using indices (No huge X_shuffled copy)
            # Reshape to [B, m, d]
            X_batch = X[batch_perm_indices].reshape(num_blocks, m, -1)

            # D. Compute
            ls_blocks = calc_approx_ls_batched(X_batch, lambda_, kernel_params)

            # E. Store in permuted order (contiguous write)
            scores_permuted[batch_start_idx:batch_end_idx] = ls_blocks.reshape(-1)

            block_start = block_end

    # 4. Process Remainder
    if remainder > 0:
        start_idx = n_full * m
        remainder_indices = perm[start_idx:]
        X_remainder = X[remainder_indices]

        # Note: Ensure calc_approx_ls handles size < m appropriately
        scores_permuted[start_idx:] = calc_approx_ls(X_remainder, lambda_, kernel_params)

    # 5. Unshuffle
    # Create final container
    scores = torch.empty_like(scores_permuted)
    # Invert the permutation mapping
    scores[perm] = scores_permuted

    return scores

@torch.no_grad()
def calc_approx_ls(
    X: torch.Tensor,
    lamb: float | torch.Tensor,
    kernel_params: torch.Tensor,
) -> torch.Tensor:
    assert X.dim() == 2, "X must be [m, d]"
    d = X.size(1)
    assert d % 2 == 0, f"feature dim {d} must be even (split into halves)."
    m = X.size(0)

    # split features into position + direction halves
    x, xn = torch.split(X, d // 2, dim=1)
    x = x.contiguous()
    xn = xn.contiguous()

    b = _eye(m, like=X)
    K_S = Var_met_reduce(x, x, xn, xn, b, kernel_params[0], kernel_params[1])

    if not torch.is_tensor(lamb):
        lamb = torch.tensor(lamb, device=X.device, dtype=X.dtype)
    A = K_S + lamb * _eye(m, like=K_S)

    Ainv = _chol_inverse(A)
    ls = (K_S * Ainv).sum(dim=1)
    return ls

calc_approx_ls_batched = torch.vmap(calc_approx_ls, in_dims=(0, None, None))

# def calc_varifold_RLS(
#     X: torch.Tensor,
#     lambda_: float,
#     sample_size: int,
#     kernel_params: torch.Tensor,
#     *,
#     seed: Optional[int] = None,
# ) -> torch.Tensor:
#     N = X.size(0)
#     if N == 0:
#         return torch.zeros(0, device=X.device, dtype=X.dtype)
#
#     m = max(1, min(sample_size, N))
#     perm = shuffle_indices(N, seed=seed).to(device=X.device)
#
#     X_shuffled = X[perm]
#
#     n_full = N // m
#     remainder = N - n_full * m
#
#     scores_permuted = torch.empty(N, device=X.device, dtype=X.dtype)
#
#     # Full blocks
#     for bi in range(n_full):
#         start = bi * m
#         end = start + m
#         block = X_shuffled[start:end]
#         scores_permuted[start:end] = calc_approx_ls(block, lambda_, kernel_params)
#
#     # Remainder
#     if remainder > 0:
#         start = n_full * m
#         block = X_shuffled[start:]
#         scores_permuted[start:] = calc_approx_ls(block, lambda_, kernel_params)
#
#     # Unshuffle back to original order
#     scores = torch.empty_like(scores_permuted)
#     scores[perm] = scores_permuted
#     return scores


# def calc_varifold_compressed_weights(
#         compressed_size: int,
#         input_samples: torch.Tensor,
#         v_areas: torch.Tensor,
#         kernel_params: torch.Tensor,
#         idx: torch.Tensor,
#         *,
#         jitter: float = 1e-10,
# ) -> torch.Tensor:
#     d = input_samples.size(1)
#     assert d % 2 == 0, "feature dim must be even."
#
#     ctrl = input_samples.index_select(0, idx).contiguous()
#     x, u = torch.split(ctrl, d // 2, dim=1)
#     x = x.contiguous()
#     u = u.contiguous()
#
#     y, v = torch.split(input_samples, d // 2, dim=1)
#     y = y.contiguous()
#     v = v.contiguous()
#
#     b = _eye(compressed_size, like=ctrl)
#     K_c = Var_met_reduce(x, x, u, u, b, kernel_params[0], kernel_params[1])
#     vals = Var_met_reduce(x, y, u, v, v_areas, kernel_params[0], kernel_params[1])
#
#     A = K_c + (jitter * K_c.diagonal().mean().clamp_min(1.0)) * _eye(compressed_size, like=K_c)
#     w = _chol_solve(A, vals.unsqueeze(1))
#     return w


    # KernelSolve definition
_VARIFOLD_SOLVER = KernelSolve(
    "Exp(-SqDist(x, y) * g) * Exp((u | v) * g_sph) * w",
    [
        "x=Vi(3)",
        "y=Vj(3)",
        "u=Vi(3)",
        "v=Vj(3)",
        "w=Vj(1)",
        "g=Pm(1)",
        "g_sph=Pm(1)",
    ],
    "w",
    axis=1,
    use_double_acc=True,
    sum_scheme="block_sum",
    enable_chunks=True,
    use_fast_math=True,
)

def calc_varifold_compressed_weights(
        compressed_size: int,
        input_samples: torch.Tensor,
        v_areas: torch.Tensor,
        kernel_params: torch.Tensor,
        idx: torch.Tensor,
        *,
        jitter: float = 1e-10,
        eps: float = 1e-4,
) -> torch.Tensor:
    d = input_samples.size(1)
    assert d % 2 == 0, "feature dim must be even."

    # Make sure everything is on the same device
    device = input_samples.device

    ctrl = input_samples.index_select(0, idx).contiguous().to(device)
    assert compressed_size == ctrl.shape[0]

    x, u = torch.split(ctrl, d // 2, dim=1)
    x = x.contiguous()
    u = u.contiguous()

    y, v = torch.split(input_samples.to(device), d // 2, dim=1)
    y = y.contiguous()
    v = v.contiguous()

    v_areas = v_areas.to(device)
    kernel_params = kernel_params.to(device)

    gamma = kernel_params[0].reshape(1)
    gamma_sph = kernel_params[1].reshape(1)

    # RHS: vals = K(x, y) @ v_areas  (same as dense version)
    vals = Var_met_reduce(x, y, u, v, v_areas, gamma, gamma_sph)
    b = vals.unsqueeze(1)  # [C, 1]

    # Match dense regularisation scaling analytically:
    # K_c[i,i] = exp( (||u_i||^2) * gamma_sph )
    norm_sq = (u * u).sum(dim=1)
    diag_vals = (norm_sq * gamma_sph).exp()
    diag_mean = diag_vals.mean().clamp_min(1.0)
    alpha = float(jitter * diag_mean)

    w = _VARIFOLD_SOLVER(
        x, x,
        u, u,
        b,
        gamma,
        gamma_sph,
        alpha=alpha,
        eps=eps,
        device_id=x.device.index if x.is_cuda else -1,
    )

    return w

def compute_compression_params(
    input_samples: torch.Tensor,          # [N, d]
    input_sample_areas: torch.Tensor,     # [N]
    kernel_params: torch.Tensor,          # [2]
    lambda_coeff: float,
    sample_size: int,
    compressed_size: int,
    *,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        idx: [P] sampled indices
        beta: [P, 1] weights
    """
    # approximate leverage scores
    scores = calc_varifold_RLS(input_samples, lambda_coeff, sample_size, kernel_params, seed=seed)

    # normalise and guard against zeros
    probs = (scores / (scores.sum() + 1e-12)).clamp_min(1e-12)
    probs = probs / probs.sum()

    # sample without replacement
    N = input_samples.shape[0]
    S = min(compressed_size, N)
    idx = torch.multinomial(probs, num_samples=S, replacement=False)

    beta = calc_varifold_compressed_weights(compressed_size, input_samples, input_sample_areas, kernel_params, idx)
    return idx, beta

# COMPRESSION SHOULD BE UNDERTAKEN USING 64-BIT FLOATS
def compress_mesh_torch(
    v: np.ndarray, f: np.ndarray,
    compressed_size: int,
    kernel_params: torch.Tensor,         # [2] on desired device/dtype
    lambda_coeff: float,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float64,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Builds varifold samples from mesh and computes compression indices+weights.

    Returns:
        idxs [p], weights [p,1], input_samples [F,6], input_sample_areas [F]
    """
    device = torch.device(device)

    # geometry features via libigl (CPU numpy)
    face_norms = igl.per_face_normals(v, f, np.ones(3))
    face_areas = 0.5 * igl.doublearea(v, f)
    v_centres = igl.barycenter(v, f)

    # to torch
    v_centres_t = torch.from_numpy(v_centres).to(device=device, dtype=dtype)   # [F,3]
    v_norms_t   = torch.from_numpy(face_norms).to(device=device, dtype=dtype)  # [F,3]
    input_samples = torch.cat((v_centres_t, v_norms_t), dim=1)                 # [F,6]
    input_sample_areas = torch.from_numpy(face_areas).to(device=device, dtype=dtype).flatten()  # [F]

    N = input_samples.size(0)
    if N == 0:
        raise ValueError("Mesh has no faces; cannot compress.")

    # heuristic block size: sqrt(N), clamped
    sample_size = max(1, min(int(math.isqrt(N)), N))

    with torch.no_grad():
        idxs, weights = compute_compression_params(
            input_samples, input_sample_areas, kernel_params,
            lambda_coeff=lambda_coeff, sample_size=sample_size, compressed_size=compressed_size, seed=seed
        )
    return idxs, weights, input_samples, input_sample_areas


# COMPRESSION SHOULD BE UNDERTAKEN USING 64-BIT FLOATS
def calc_varifold_compression_torch(
        v_np: np.ndarray,
        f_np: np.ndarray,
        sigma: float,
        sigma_sph: float,
        compressed_size: int,
        lambda_coeff: float = 1.0,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float64,
        check_quality: bool = False,
        quality_threshold: float = 1e-2,
        seed: Optional[int] = None,
        return_numpy: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = torch.device(device)
    gamma = 1.0 / (2.0 * sigma * sigma)
    gamma_sph = 1.0 / (2.0 * sigma_sph * sigma_sph)
    kernel_params = torch.tensor([gamma, gamma_sph], device=device, dtype=dtype)

    idxs, weights, input_samples, input_sample_areas = compress_mesh_torch(
        v_np, f_np, compressed_size, kernel_params, lambda_coeff, device=device, dtype=dtype, seed=seed
    )

    if check_quality:
        # Keep uncompressed only for quality check
        x_centres = input_samples[:, :3].contiguous()
        x_normals = input_samples[:, 3:].contiguous()
        x_areas = input_sample_areas

        # For quality check, we need the compressed samples
        w_centres_check = input_samples[idxs, :3].contiguous()
        w_normals_check = input_samples[idxs, 3:].contiguous()
        w_weights_check = weights.flatten()

        dist = calc_varifold_loss_torch(
            x_centres, x_normals, x_areas,
            w_centres_check, w_normals_check, w_weights_check,
            kernel_params
        )
        print(f"Varifold Dual Dist (Compressed vs Uncompressed): {dist.item():.3e}")
        if dist > quality_threshold:
            print("WARNING: compression quality is low; downstream fitting may degrade.")

        # Free immediately after use
        del x_centres, x_normals, x_areas, dist
        del w_centres_check, w_normals_check, w_weights_check

    if return_numpy:
        # Direct indexing -> numpy, avoiding intermediate contiguous tensors
        result = (
            idxs.detach().cpu().numpy(),
            input_samples[idxs, :3].detach().cpu().numpy(),
            input_samples[idxs, 3:].detach().cpu().numpy(),
            weights.flatten().detach().cpu().numpy(),
        )

        # Clean up
        del idxs, weights, input_samples, input_sample_areas, kernel_params
        torch.cuda.empty_cache()
        return result

    else:
        # Only create contiguous tensors if returning torch tensors
        w_centres = input_samples[idxs, :3].contiguous()
        w_normals = input_samples[idxs, 3:].contiguous()
        w_weights = weights.flatten()

        # Free large intermediate tensors
        del input_samples, input_sample_areas, weights

        return idxs, w_centres, w_normals, w_weights

