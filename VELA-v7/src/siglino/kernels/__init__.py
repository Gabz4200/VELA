"""SigLino attention kernels.

CPU path: compiled SDPA  (``cpu_attn.cpu_sdpa``)
CUDA path: flex_attention (``cuda_attn.cuda_flex_attn``)
"""

from .cpu_attn import cpu_sdpa

__all__ = ["cpu_sdpa"]

# cuda_flex_attn is imported lazily in attention.py so that CPU-only
# environments don't fail at import time.
