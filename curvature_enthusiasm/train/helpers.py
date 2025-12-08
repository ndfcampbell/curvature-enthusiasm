from typing import Dict
import numpy as np
import jax
import jax.numpy as jnp


def create_lambda_weights(config_section, dtype) -> Dict[str, jax.Array]:
    return {
        "sample_weights": jnp.asarray(getattr(config_section, "bone_sample_ep", 0.0), dtype=dtype),
        "sample_traj_weight": jnp.asarray(getattr(config_section, "bone_sample_traj", 0.0), dtype=dtype),
        "pc_acap_coeff": jnp.asarray(getattr(config_section, "surface_acap", 0.0), dtype=dtype),
        "tissue_acap_coeff": jnp.asarray(getattr(config_section, "tissue_acap", 0.0), dtype=dtype),
        "key_points_ep": jnp.asarray(getattr(config_section, "key_points_ep", 0.0), dtype=dtype),
    }


def format_kernel_params(kernel_params, phase_name: str) -> str:
    params_np = np.asarray(kernel_params).flatten()
    if params_np.size == 0:
        return "no_params"
    if phase_name == "varifold":
        if params_np.size >= 2:
            gamma = float(params_np[0])
            gamma_sph = float(params_np[1])
            return f"γ={gamma:.1f},γ_sph={gamma_sph:.1f}"
        gamma = float(params_np[0])
        return f"γ={gamma:.1f}"
    else:
        gamma = float(params_np[0])
        return f"γ={gamma:.1f}"