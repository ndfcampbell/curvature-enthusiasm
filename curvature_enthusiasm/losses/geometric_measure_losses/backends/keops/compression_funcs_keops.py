import torch
from pykeops.torch import LazyTensor
from jaxtyping import Array, Float, Int


def GK(x, y, b, gamma):
    xx = LazyTensor(x[None ,: ,:])
    yy = LazyTensor(y[: ,None ,:])
    bb = LazyTensor(b[: ,None ,:])
    D2 = xx.sqdist(yy)
    K = (-D2 * gamma).exp()
    return (K * bb).sum(0)


def Var_met_reduce(x, y, u, v, b, gamma, gamma_sph):
    # x, y, b = Vi(0, 3), Vj(1, 3), Vj(2, 3)
    x = LazyTensor(x[:, None, :])
    y = LazyTensor(y[None, :, :])

    u = LazyTensor(u[:, None, :])
    v = LazyTensor(v[None, :, :])

    D2 = x.sqdist(y)
    ss = ((u * v)).sum()

    res = (ss * gamma_sph).exp()

    # res = (-(2 - 2 * ss) * gamma_sph).exp()  # 'gaussian kernel'
    # res = ss**2 # Binet kernel
    K = (-D2 * gamma).exp()

    return (res * K) @ b  # res*K



def Var_met_reduce_batched(x, y, u, v, b, gamma, gamma_sph):
    """
    x, y, u, v: [B, m, d]
    b: [B, m, k]
    gamma, gamma_sph: scalar or 0-d tensor
    returns: [B, m, k]
    """
    # Ensure dtype/device consistency
    device = x.device
    dtype = x.dtype

    gamma = torch.as_tensor(gamma, device=device, dtype=dtype)
    gamma_sph = torch.as_tensor(gamma_sph, device=device, dtype=dtype)

    # KeOps LazyTensors with batch dimension
    # shapes: x_i: [B, m, 1, d], y_j: [B, 1, m, d]
    x_i = LazyTensor(x[:, :, None, :])
    y_j = LazyTensor(y[:, None, :, :])

    u_i = LazyTensor(u[:, :, None, :])
    v_j = LazyTensor(v[:, None, :, :])

    # Squared distances and inner products
    D2 = ((x_i - y_j) ** 2).sum(-1)     # [B, m, m]
    ss = (u_i * v_j).sum(-1)           # [B, m, m]

    res = (ss * gamma_sph).exp()       # [B, m, m]
    K = (-D2 * gamma).exp()            # [B, m, m]

    # (res * K) @ b; KeOps handles matmul with dense b : (B, m, m) x (B, m, k)
    return (res * K) @ b        # [B, m, k]