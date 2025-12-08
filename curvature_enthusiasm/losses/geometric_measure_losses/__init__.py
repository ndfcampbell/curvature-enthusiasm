from .backends.torch import (
    calc_varifold_loss_torch,
    calc_normal_cycle_loss_torch,
    calc_varifold_compression_torch,
    calc_normal_cycle_compression_torch
)

from .varifold_loss import (
    Keops_Varifold_Loss,
    extract_varifold_properties
)

from .normal_cycle_loss import (
    Keops_Normal_Cycles_Loss,
)

from .brdy_funcs_jax import (
    calc_parts_and_weights,
    calc_parts_and_weights_brdy,
    pre_comp_nc_vec_con,
)


__all__ = [
    "calc_varifold_loss_torch",
    "calc_normal_cycle_loss_torch",
    "calc_varifold_compression_torch",
    "calc_normal_cycle_compression_torch",
    "Keops_Varifold_Loss",
    "Keops_Normal_Cycles_Loss",
    "extract_varifold_properties",
    "calc_parts_and_weights",
    "calc_parts_and_weights_brdy",
    "pre_comp_nc_vec_con",
]