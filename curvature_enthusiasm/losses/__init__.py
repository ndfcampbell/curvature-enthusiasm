"""Loss functions for shape deformation."""

# Varifold loss components
from .geometric_measure_losses import (
    Keops_Varifold_Loss,
    Keops_Normal_Cycles_Loss,
    extract_varifold_properties,
    calc_parts_and_weights,
    calc_parts_and_weights_brdy,
    pre_comp_nc_vec_con,
)

# Conformal energy components
from .as_conformal_as_possible import ACAP_Surface_Energy, ACAP_Sample_Energy

# Define public API
__all__ = [
    # Varifold
    "Keops_Varifold_Loss",
    "extract_varifold_properties",

    # Normal cycles
    "Keops_Normal_Cycles_Loss",
    "calc_parts_and_weights",
    "calc_parts_and_weights_brdy",
    "pre_comp_nc_vec_con",

    # ACAP
    "ACAP_Surface_Energy",
    "ACAP_Sample_Energy",
]