from .loss_funcs_torch import calc_varifold_loss_torch, calc_normal_cycle_loss_torch, calc_varifold_loss_torch_old_style
from .varifold_compression_funcs_torch import calc_varifold_compression_torch
from .normal_cycle_compression_funcs_torch import calc_normal_cycle_compression_torch

__all__ = [
    "calc_varifold_loss_torch",
    "calc_varifold_loss_torch_old_style"
    "calc_normal_cycle_loss_torch",
    "calc_varifold_compression_torch",
    "calc_normal_cycle_compression_torch",
]