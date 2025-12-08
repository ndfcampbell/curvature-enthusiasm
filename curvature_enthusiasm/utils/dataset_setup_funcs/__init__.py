"""Dataset-specific skeleton setup functions."""

from .setup_faust_smpl_skeleton import (
    setup_faust_smpl_skeleton,
    setup_smpl_skeleton_no_hands,
)
from .setup_mano_skeleton import (
    setup_mano_skeleton,
    # load_mano_model,
)
from .setup_skel_model import (
    load_skel_model,
    setup_skel_skeleton,
)
from .setup_smal_skeleton import setup_smal_skeleton
from .setup_smpl_skeleton import (
    setup_smpl_skeleton,
    load_smpl_model,
    determine_gender_dfaust,
)

__all__ = [
    # FAUST/SMPL
    "setup_faust_smpl_skeleton",
    "setup_smpl_skeleton_no_hands",
    # MANO
    "setup_mano_skeleton",
    # "load_mano_model",
    # SKEL
    "load_skel_model",
    "setup_skel_skeleton",
    # SMAL
    "setup_smal_skeleton",
    # SMPL
    "setup_smpl_skeleton",
    "load_smpl_model",
    "determine_gender_dfaust",
]