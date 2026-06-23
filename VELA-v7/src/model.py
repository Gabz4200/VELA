# VELA-v7/src/model.py
# Facade module - re-exports for backward compatibility
# All implementation is in models/vlm.py and models/vla.py

from .models import VLA, VLM
from .models.vlm import (
    CHUNK_LEN,
    HAS_CPP_EXT,
    HEAD_SIZE,
    RWKV,
    Block,
    MHCBlock,
    RWKV_CMix_x070,
    RWKV_Tmix_x070,
    Sinkhorn_Knopp,
    WindBackstepping,
    wind_backstepping_ref_backward,
    wind_backstepping_ref_forward,
)

# Backward-compatible alias
VELA = VLM

__all__ = [
    "VLM",
    "VLA",
    "VELA",  # alias
    "RWKV",
    "Block",
    "MHCBlock",
    "RWKV_Tmix_x070",
    "RWKV_CMix_x070",
    "WindBackstepping",
    "wind_backstepping_ref_forward",
    "wind_backstepping_ref_backward",
    "Sinkhorn_Knopp",
    "HAS_CPP_EXT",
    "HEAD_SIZE",
    "CHUNK_LEN",
]
