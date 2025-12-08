from typing import Tuple, Literal
from jaxtyping import Float, Int, Bool

import torch
import numpy as np
from ..keops.compression_funcs_keops import GK #, N_compressed
from .loss_funcs_torch import calc_normal_cycle_loss_torch

import torch
from pykeops.torch import KernelSolve, LazyTensor
from typing import Optional

# def rbf_kernel(
#     X: torch.Tensor,           # (..., N, k)  e.g. [2000,3] or [B,1000,3]
#     Y: torch.Tensor,           # (..., M, k)  e.g. [M,3]   or [B,1000,3]
#     gamma,                     # float or 0-dim tensor
#     use_abs: bool = False,     # True to mimic your kern_met_abs behaviour
# ) -> torch.Tensor:
#     """
#     Returns K = exp(-gamma * D), where D = ||X - Y||^2 (optionally |...|).
#     Shapes: X(...,N,k), Y(...,M,k) -> K(...,N,M). Leading dims must broadcast.
#     """
#     # Make gamma a scalar tensor on the right device/dtype (no casts later).
#     gamma = torch.as_tensor(gamma, device=X.device, dtype=X.dtype)
#
#     # Pairwise squared distances with broadcasting.
#     # XX: (..., N, 1), YY: (..., 1, M), cross: (..., N, M)
#     XX = (X * X).sum(dim=-1, keepdim=True)
#     YY = (Y * Y).sum(dim=-1, keepdim=True).transpose(-2, -1)
#     cross = -2 * (X @ Y.transpose(-2, -1))
#     D = cross + XX + YY
#
#     if use_abs:  # for the NC variant you had
#         D = D.abs()
#
#     return torch.exp(-gamma * D)
#
#
# --- stable solve (SPD) ---
# def _solve_spd(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
#     # A must be symmetric positive-definite. Uses Cholesky solve.
#     L = torch.linalg.cholesky(A)
#     return torch.cholesky_solve(B, L)
#
# # orthogonal projection weights
# def calc_nc_compressed_weights(
#     idx: torch.Tensor,
#     GEN: torch.Tensor,          # [N, 3]
#     alph: torch.Tensor,         # [N, ...]
#     gamma: torch.Tensor,        # scalar tensor
#     device: str | torch.device = "cuda",
#     dtype: torch.dtype = torch.float64,
#     jitter: float = 1e-3,
# ) -> torch.Tensor:
#     device = torch.device(device)
#     GEN = GEN.to(device=device, dtype=dtype).contiguous()
#     alph = alph.to(device=device, dtype=dtype).contiguous()
#     gamma = torch.as_tensor(gamma, device=device, dtype=dtype)
#
#     ctrl = GEN.index_select(0, idx.to(device)).contiguous()
#     P = ctrl.shape[0]
#
#     Kc = rbf_kernel(ctrl, ctrl, gamma)                         # [P, P]
#     vals = GK(ctrl, GEN, alph, gamma)                          # [P, ...] - now all inputs contiguous
#
#     A = Kc + jitter * torch.eye(P, device=device, dtype=dtype)
#     beta = _solve_spd(A, vals.to(dtype))
#     return beta

#
# # RLS scores (DAC), torch-only
# def calc_nc_RLS(
#     X: torch.Tensor,              # [N, k]
#     lambda_: float,
#     sample_size: int,
#     gamma: torch.Tensor,          # scalar tensor
#     device: str | torch.device = "cuda",
#     dtype: torch.dtype = torch.float64,
# ) -> torch.Tensor:
#     device = torch.device(device)
#     X = X.to(device=device, dtype=dtype)
#     gamma = torch.as_tensor(gamma, device=device, dtype=dtype)
#
#     N = X.shape[0]
#     if N == 0:
#         return torch.zeros(0, device=device, dtype=dtype)
#
#     S = max(1, min(sample_size, N))
#     perm = torch.randperm(N, device=device)
#     blocks = perm.split(S)
#
#     scores = torch.zeros(N, device=device, dtype=dtype)
#
#     # full-size blocks (except last)
#     for b in blocks[:-1]:
#         Xb = X[b].unsqueeze(0)                         # [1, S, k]
#         Var = rbf_kernel(Xb, Xb, gamma, use_abs=True)[0]   # [S, S]
#         A = Var + lambda_ * torch.eye(S, device=device, dtype=dtype)
#         # We need elementwise Var * A^{-1} then row-sum; use cholesky_inverse:
#         Ainv = torch.cholesky_inverse(torch.linalg.cholesky(A))
#         scores[b] = (Var * Ainv).sum(dim=1)            # [S]
#
#     # last (possibly smaller)
#     b = blocks[-1]
#     Xb = X[b].unsqueeze(0)                             # [1, S_last, k]
#     Var = rbf_kernel(Xb, Xb, gamma, use_abs=True)[0]   # [S_last, S_last]
#     S_last = Var.shape[0]
#     A = Var + lambda_ * torch.eye(S_last, device=device, dtype=dtype)
#     Ainv = torch.cholesky_inverse(torch.linalg.cholesky(A))
#     scores[b] = (Var * Ainv).sum(dim=1)
#
#     return scores



# 1. Define the KeOps Solver for Gaussian RBF
# This compiles a specific CUDA kernel to solve (K + alpha*I)w = b
# x, y: position coordinates [N, 3]
# b: target signal (RHS) [N, 1]
# g: gamma (width) [1]
# 1. Define the Solver
# Matches: (K_cc + jitter * I) * w = RHS
_NC_SOLVER = KernelSolve(
    "Exp(-SqDist(x, y) * g) * b",
    [
        "x=Vi(3)",    # Control points (rows)
        "y=Vj(3)",    # Control points (cols)
        "b=Vj(15)",   # <--- FIXED: Signal is 15-dim (Normal Cycle)
        "g=Pm(1)",    # Gamma
    ],
    "b",              # Solving for 'b'
    axis=1,
    use_double_acc=True,
    sum_scheme="block_sum",
    enable_chunks=True,
    use_fast_math=True,
)


def calc_nc_compressed_weights(
        idx: torch.Tensor,
        GEN: torch.Tensor,  # [N, 3]
        alph: torch.Tensor,  # [N, 1]
        gamma: torch.Tensor,  # scalar
        *,
        jitter: float = 1e-3,
        eps: float = 1e-4,
) -> torch.Tensor:
    device = GEN.device
    GEN = GEN.contiguous()

    if alph.dim() == 1:
        alph = alph.unsqueeze(1)
    alph = alph.contiguous()

    gamma = torch.as_tensor(gamma, device=device, dtype=GEN.dtype).reshape(1)

    # ctrl: [M, 3]
    ctrl = GEN.index_select(0, idx.to(device)).contiguous()

    # --- RHS Calculation (Same as GK) ---
    # x_i creates virtual axis 0 (M rows)
    # y_j creates virtual axis 1 (N cols)
    x_i = LazyTensor(ctrl[:, None, :])  # [M, 1, 3]
    y_j = LazyTensor(GEN[None, :, :])  # [1, N, 3]

    D2 = x_i.sqdist(y_j)  # [M, N] symbolic matrix
    K_xv = (-D2 * gamma).exp()  # [M, N] symbolic matrix

    # Equivalent to GK's (K * bb).sum(0)
    # Matrix Mult sums over the inner dimension N
    vals = K_xv @ alph  # [M, 15]

    # --- LHS Calculation (Weights) ---
    w = _NC_SOLVER(
        ctrl, ctrl,
        vals,
        gamma,
        alpha=jitter,
        eps=eps
    )

    return w

def NC_met_reduce(x, y, b, gamma):
    """
    KeOps implementation of Gaussian RBF Kernel matrix-multiply.
    Computes (Exp(-gamma * ||x - y||^2)) @ b
    """
    # x, y are [M, D]
    # b is [M, M] (Identity matrix in this specific context)

    x_i = LazyTensor(x[:, None, :])  # [M, 1, D]
    y_j = LazyTensor(y[None, :, :])  # [1, M, D]

    # Squared Euclidean Distance: ||x - y||^2
    D2 = x_i.sqdist(y_j)

    # Gaussian Kernel: exp(-gamma * D2)
    K = (-D2 * gamma).exp()

    # Matrix Multiply: K @ b
    return K @ b

def _eye(n: int, like: torch.Tensor) -> torch.Tensor:
    return torch.eye(n, device=like.device, dtype=like.dtype)


def _chol_inverse(A: torch.Tensor) -> torch.Tensor:
    L = torch.linalg.cholesky(A)
    return torch.cholesky_solve(_eye(A.size(-1), A), L)


@torch.no_grad()
def calc_approx_ls_nc(
        X: torch.Tensor,
        lamb: float | torch.Tensor,
        gamma: float | torch.Tensor,
) -> torch.Tensor:
    """
    Single block RLS calculation for Normal Cycle (Gaussian RBF).
    """
    m = X.size(0)

    # Ensure gamma is a tensor for KeOps compatibility inside vmap
    if not torch.is_tensor(gamma):
        gamma = torch.tensor(gamma, device=X.device, dtype=X.dtype)
    if not torch.is_tensor(lamb):
        lamb = torch.tensor(lamb, device=X.device, dtype=X.dtype)

    # 1. Construct Identity for reduction
    b = _eye(m, like=X)

    # 2. Compute Kernel Matrix using KeOps
    # Note: For small block sizes (m < 2000), KeOps is fast.
    # For very large m, this efficiently constructs the matrix.
    K = NC_met_reduce(X, X, b, gamma)

    # 3. Regularize
    A = K + lamb * b

    # 4. Solve and Compute Leverage Scores
    # score_i = [K * (K + lambda I)^-1]_ii
    # We compute row sums of elementwise product
    Ainv = _chol_inverse(A)
    ls = (K * Ainv).sum(dim=1)

    return ls


# Create the vectorized version
# in_dims: (X is batched, lamb is scalar/shared, gamma is scalar/shared)
calc_approx_ls_nc_batched = torch.vmap(calc_approx_ls_nc, in_dims=(0, None, None))

def shuffle_indices(n: int, seed: Optional[int] = None) -> torch.Tensor:
    if seed is not None:
        g = torch.Generator().manual_seed(seed)
        return torch.randperm(n, generator=g)
    return torch.randperm(n)


def calc_nc_RLS(
        X: torch.Tensor,
        lambda_: float,
        sample_size: int,
        gamma: float | torch.Tensor,
        *,
        seed: Optional[int] = None,
        blocks_per_batch: int = 128,  # Adjust based on GPU VRAM
) -> torch.Tensor:
    N = X.size(0)
    if N == 0:
        return torch.zeros(0, device=X.device, dtype=X.dtype)

    m = max(1, min(sample_size, N))
    device = X.device

    # 1. Create Permutation (Indices only - saves memory)
    perm = shuffle_indices(N, seed=seed).to(device=device)

    scores_permuted = torch.empty(N, device=device, dtype=X.dtype)

    n_full = N // m
    remainder = N - n_full * m

    # 2. Process Full Blocks (Batched)
    if n_full > 0:
        block_start = 0
        while block_start < n_full:
            block_end = min(block_start + blocks_per_batch, n_full)
            num_blocks = block_end - block_start

            # Batch indices
            start_idx = block_start * m
            end_idx = block_end * m

            # Slice X on the fly (Efficient)
            batch_indices = perm[start_idx:end_idx]
            X_batch = X[batch_indices].reshape(num_blocks, m, -1)

            # Compute via vmap
            ls_blocks = calc_approx_ls_nc_batched(X_batch, lambda_, gamma)

            # Store results
            scores_permuted[start_idx:end_idx] = ls_blocks.reshape(-1)

            block_start = block_end

    # 3. Process Remainder (Scalar fall-back)
    if remainder > 0:
        start_idx = n_full * m
        remainder_indices = perm[start_idx:]
        X_remainder = X[remainder_indices]

        # Call the un-batched single block solver
        scores_permuted[start_idx:] = calc_approx_ls_nc(X_remainder, lambda_, gamma)

    # 4. Unshuffle
    scores = torch.empty_like(scores_permuted)
    scores[perm] = scores_permuted

    return scores


def calc_normal_cycle_compression_torch(
    gen_centres: Float[np.ndarray, "N 3"],
    gen_weights: Float[np.ndarray, "N 15"],
    sig_coeff: float = 0.5,
    compressed_size: int = 1000,
    sampler: Literal["uni", "DAC"] = "DAC",
    device: str = "cuda",
    dtype: torch.dtype = torch.float64,
    return_numpy: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[np.ndarray, np.ndarray]:
    """
    Compress Normal Cycles centres/weights by sampling and reweighting.
    """
    device_t = torch.device(device)
    gen_centres_t = torch.as_tensor(gen_centres, device=device_t, dtype=dtype).contiguous()
    gen_weights_t = torch.as_tensor(gen_weights, device=device_t, dtype=dtype).contiguous()

    N = gen_centres_t.shape[0]
    if N == 0:
        empty = (gen_centres_t, gen_weights_t.new_zeros((0,)))
        if return_numpy:
            return empty[0].cpu().numpy(), empty[1].cpu().numpy()
        return empty

    S = min(compressed_size, N)
    sigma = torch.as_tensor(sig_coeff, device=device_t, dtype=dtype)
    gamma = 1.0 / (2.0 * sigma * sigma)   # scalar tensor

    if sampler == "uni":
        idxs = torch.randperm(N, device=device_t)[:S]
    elif sampler == "DAC":
        scores = calc_nc_RLS(gen_centres_t, lambda_=0.1, sample_size=min(1000, N), gamma=gamma)
        probs = scores / (scores.sum() + 1e-12).clamp_min(1e-12)

        idxs = torch.multinomial(probs, num_samples=S, replacement=False)
    else:
        raise ValueError(f"Unknown sampler: {sampler!r} (use 'uni' or 'DAC').")

    comp_centres = gen_centres_t.index_select(0, idxs).contiguous()  # ← ADD
    comp_weights = calc_nc_compressed_weights(idxs, gen_centres_t, gen_weights_t, gamma)

    res = calc_normal_cycle_loss_torch(
        comp_centres.contiguous(),
        comp_weights.contiguous(),
        gen_centres_t.contiguous(),
        gen_weights_t.contiguous(),
        gamma
    )
    print(f"NC Dual Dist (Compressed vs Uncompressed): {res.item():.3e}")

    # THIS IS EDGE IDXs, EDGE CENTRES, EDGE WEIGHTS
    # WHEN DEALING WITH BOUNDARY, IT HAS SPECIAL CASES SO EDGES != STRAIGHT EDGES OF TRIANGULAR MESH
    if return_numpy:
        # Convert to numpy first
        result = (
            idxs.detach().cpu().numpy(),
            comp_centres.detach().cpu().numpy(),
            comp_weights.detach().cpu().numpy()
        )

        # Explicitly delete all PyTorch tensors
        del idxs, comp_centres, comp_weights
        del gen_centres_t, gen_weights_t
        del sigma, gamma, res

        # Clear CUDA cache
        torch.cuda.empty_cache()

        return result

    return idxs, comp_centres, comp_weights
