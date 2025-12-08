# config_schema.py
from __future__ import annotations

from typing import Dict, List, Tuple, Callable, Any
from pathlib import Path
import yaml

import jax.nn as jnn
from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    field_validator,
    model_validator,
    PrivateAttr,           # <-- add this import
    computed_field
)

# ---------------------------
# Small activation registry
# ---------------------------

_ACTIVATIONS: Dict[str, Callable] = {
    "gelu": jnn.gelu,
    "siren": lambda x: jnn.sin(x),
    "finer": lambda x: x,      # TODO: replace with your real function
    "identity": lambda x: x,
}

def resolve_activation(name: str) -> Callable:
    key = (name or "identity").lower()
    if key not in _ACTIVATIONS:
        raise KeyError(f"Unknown activation: {name!r}. Known: {sorted(_ACTIVATIONS)}")
    return _ACTIVATIONS[key]


# ---------------------------
# Enum
# ---------------------------

# class UpDirection(str, Enum):
#     Y_UP = "y_up"
#     Z_UP = "z_up"


# ---------------------------
# Submodels
# ---------------------------

class OptimizationSettings(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    peak_lr: float = Field(5.0e-3, alias="PEAK_LR", gt=0)
    end_lr: float = Field(1.0e-4, alias="END_LR", gt=0)
    ft_lr: float = Field(1.0e-4, alias="FINE_LR", gt=0)
    warmup_steps: int = Field(50, alias="WARMUP_STEPS", ge=0)
    eval_per_epochs: int = Field(500, alias="EVAL_PER_EPOCHS", gt=0)

class NodeSettings(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    n_ode_steps: int = Field(10, alias="N_ODE_STEPS", gt=0)
    use_div_free: bool = Field(False, alias="USE_DIV_FREE")
    activation_func: str = Field("gelu", alias="ACTIVATION_FUNC")

    # not serialized; only on the Python object
    _activation_callable: Callable = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _resolve_activation_callable(self):
        self._activation_callable = resolve_activation(self.activation_func)
        return self

    @property
    def activation_fn(self) -> Callable:
        return self._activation_callable



class BoneSamplingSettings(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    n_rigid_samples_per_bone: int = Field(50, alias="N_RIGID_SAMPLES_PER_BONE", gt=0)
    n_tissue_samples_per_iter: int = Field(2000, alias="N_TISSUE_SAMPLES_PER_ITER", gt=0)
    n_tissue_samples_per_tissue: int = Field(25, alias="N_TISSUE_SAMPLES_PER_TISSUE", gt=0)
    axis_radii: Dict[int, List[float]] = Field(
        default_factory=lambda: {0: [0.1], 1: [0.1]},
        alias="AXIS_RADII",
        description="Per-bone list of radii for sampling"
    )

    @field_validator("axis_radii")
    @classmethod
    def _validate_radii(cls, v: Dict[int, List[float]]):
        for bone_id, radii in v.items():
            if not isinstance(radii, list):
                raise ValueError(f"Bone {bone_id}: radii must be a list (got {type(radii)})")
            if len(radii) == 0:
                raise ValueError(f"Bone {bone_id}: must have at least one radius value")
            for i, r in enumerate(radii):
                if r <= 0:
                    raise ValueError(f"Bone {bone_id}, radius[{i}]: must be > 0 (got {r})")
        return v


# --- VarifoldInitialisationSettings ---
class VarifoldInitialisationSettings(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    sigmas: List[float] = Field([1.0, 0.5], alias="SIGMAS")
    sigma_sphs: List[float] = Field([1.0, 0.5], alias="SIGMA_SPHS")
    iters_per_lengthscale: List[int] = Field([1000, 1000], alias="ITERS_PER_LENGTHSCALE")

    @field_validator("sigmas", "sigma_sphs")
    @classmethod
    def _positive_lists(cls, vals: List[float]):
        if any(v <= 0 for v in vals):
            raise ValueError("All sigma values must be positive")
        return vals

    @field_validator("iters_per_lengthscale")
    @classmethod
    def _positive_iters(cls, vals: List[int]):
        if any(v <= 0 for v in vals):
            raise ValueError("All VARIFOLD_INITIALISATION iter values must be positive")
        return vals

    @model_validator(mode="after")
    def _match_lengths(self):
        if len(self.sigmas) != len(self.sigma_sphs):
            raise ValueError("VARIFOLD_INITIALISATION: SIGMAS and SIGMA_SPHS must have the same length")
        if len(self.sigmas) != len(self.iters_per_lengthscale):
            raise ValueError("VARIFOLD_INITIALISATION: len(SIGMAS) must equal len(ITERS_PER_LENGTHSCALE)")
        return self



# --- NormalCycleSettings ---
class NormalCycleSettings(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    sigmas: List[float] = Field([1.0], alias="SIGMAS")
    iters_per_lengthscale: List[int] = Field([0], alias="ITERS_PER_LENGTHSCALE")

    @field_validator("sigmas")
    @classmethod
    def _positive_sigmas(cls, vals: List[float]):
        if any(v <= 0 for v in vals):
            raise ValueError("All NORMAL_CYCLES sigma values must be positive")
        return vals

    @field_validator("iters_per_lengthscale")
    @classmethod
    def _positive_iters(cls, vals: List[int]):
        if any(v < 0 for v in vals):
            raise ValueError("All NORMAL_CYCLES iter values must be positive")
        return vals

    @model_validator(mode="after")
    def _match_lengths(self):
        if len(self.sigmas) != len(self.iters_per_lengthscale):
            raise ValueError("NORMAL_CYCLES: len(SIGMAS) must equal len(ITERS_PER_LENGTHSCALE)")
        return self



class ConstraintsSettings(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    bone_sample_ep: float = Field(1.0, alias="BONE_SAMPLE_EP", ge=0)
    bone_sample_traj: float = Field(1.0e1, alias="BONE_SAMPLE_TRAJ", ge=0)
    surface_acap: float = Field(5.0e3, alias="SURFACE_ACAP", ge=0)
    tissue_acap: float = Field(1.0e1, alias="TISSUE_ACAP", ge=0)
    key_points_ep: float = Field(0.0, alias="KEY_POINTS_EP", ge=0)


class FineTuningSettings(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    use_fine_tuning: bool = Field(False, alias="USE_FINE_TUNING")
    compress_source: bool = Field(False, alias="COMPRESS_SOURCE")
    compress_target: bool = Field(False, alias="COMPRESS_TARGET")
    ft_num_iters: int = Field(2000, alias="FT_NUM_ITERS", gt=0)
    bone_sample_ep: float = Field(2.0e2, alias="BONE_SAMPLE_EP", ge=0)
    bone_sample_traj: float = Field(1.0e1, alias="BONE_SAMPLE_TRAJ", ge=0)
    surface_acap: float = Field(5.0e3, alias="SURFACE_ACAP", ge=0)
    tissue_acap: float = Field(1.0e1, alias="TISSUE_ACAP", ge=0)
    key_points_ep: float = Field(0.0, alias="KEY_POINTS_EP", ge=0)


class DatasetSettings(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    dataset_name: str = Field("DFAUST", alias="DATASET_NAME", min_length=1)
    remesh: bool = Field(False, alias="REMESH")


class CompressionSettings(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    compress_source: bool = Field(False, alias="COMPRESS_SOURCE")
    compress_target: bool = Field(False, alias="COMPRESS_TARGET")
    compressed_var_source_size: int = Field(0, alias="COMPRESSED_VARIFOLD_SOURCE_SIZE", gt=0)
    compressed_var_target_size: int = Field(0, alias="COMPRESSED_VARIFOLD_TARGET_SIZE", gt=0)
    compressed_nc_source_size: int = Field(0, alias="COMPRESSED_NORMAL_CYCLE_SOURCE_SIZE", gt=0)
    compressed_nc_target_size: int = Field(0, alias="COMPRESSED_NORMAL_CYCLE_TARGET_SIZE", gt=0)


# --- ResultsSettings (add RESULTS_DIR since your YAML has it) ---
class ResultsSettings(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    load_from_file: bool = Field(False, alias="LOAD_FROM_FILE")
    results_dir: str = Field("results/", alias="RESULTS_DIR")
    eval_training_data: bool = Field(True, alias="EVAL_TRAINING_DATA")
    percentage_eval_seq: float = Field(10.0, alias="PERCENTAGE_EVAL_SEQ", ge=0, le=100)



# class VisualizationSettings(BaseModel):
#     # use_enum_values: True ensures YAML can be "y_up" | "z_up"
#     model_config = ConfigDict(populate_by_name=True, use_enum_values=True)
#
#     show_plots: bool = Field(False, alias="SHOW_PLOTS")
#     up_dir: UpDirection = Field(UpDirection.Y_UP, alias="UP_DIR")


# ---------------------------
# Top-level Config
# ---------------------------

class Config(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    dataset: DatasetSettings = Field(default_factory=DatasetSettings, alias="DATASET")
    optimization: OptimizationSettings = Field(default_factory=OptimizationSettings, alias="OPT")
    node: NodeSettings = Field(default_factory=NodeSettings, alias="NODE")
    bone_sampling: BoneSamplingSettings = Field(default_factory=BoneSamplingSettings, alias="BONE_SAMPLING")
    varifold: VarifoldInitialisationSettings = Field(default_factory=VarifoldInitialisationSettings, alias="VARIFOLD_INITIALISATION")
    normal_cycles: NormalCycleSettings = Field(default_factory=NormalCycleSettings, alias="NORMAL_CYCLES")
    constraints: ConstraintsSettings = Field(default_factory=ConstraintsSettings, alias="CONSTRAINTS")
    fine_tuning: FineTuningSettings = Field(default_factory=FineTuningSettings, alias="FINE_TUNING")

    compression: CompressionSettings = Field(default_factory=CompressionSettings, alias="COMPRESSION")
    results: ResultsSettings = Field(default_factory=ResultsSettings, alias="RESULTS")
    # visualization: VisualizationSettings = Field(default_factory=VisualizationSettings, alias="VISUALISATION")

    @model_validator(mode="after")
    def _enforce_invariants(self):
        # LR ordering
        if self.optimization.end_lr > self.optimization.peak_lr:
            raise ValueError("OPT.END_LR must be <= OPT.PEAK_LR.")
        return self

    # Convenience: dump a resolved, YAML-shaped dict
    def to_resolved_dict(self) -> Dict[str, Any]:
        return self.model_dump(by_alias=True)


# ---------------------------
# Loader helpers
# ---------------------------

def _preprocess_legacy_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle backward-compat for legacy/typo keys before validation.
    e.g., RESULTS.EVAL_TRAINGING_DATA -> RESULTS.EVAL_TRAINING_DATA
    """
    if not isinstance(d, dict):
        return d
    results = d.get("RESULTS")
    if isinstance(results, dict):
        if "EVAL_TRAINGING_DATA" in results and "EVAL_TRAINING_DATA" not in results:
            results["EVAL_TRAINING_DATA"] = results.pop("EVAL_TRAINGING_DATA")
    return d

def load_config(path: str | Path) -> Config:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    data = yaml.safe_load(p.read_text()) or {}
    data = _preprocess_legacy_keys(data)
    return Config.model_validate(data)

def save_resolved_config(cfg: Config, out_path: str | Path) -> None:
    """Save the fully-resolved, alias-keyed config next to your run for reproducibility."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # dump YAML with aliases
    resolved = cfg.to_resolved_dict()
    out.write_text(yaml.safe_dump(resolved, sort_keys=False))
