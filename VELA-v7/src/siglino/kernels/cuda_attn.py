"""CUDA attention kernel for SigLino: thin wrapper over flex_attention.

Provides a unified call signature matching the CPU kernel so ``Attention.forward``
can swap backends with a single dispatch call.
"""

import functools

import torch
from torch.nn.attention.flex_attention import AuxRequest, flex_attention


def cuda_flex_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask=None,        # BlockMask | None
    compile: bool = True,   # whether to use the compiled flex_attention kernel
    return_lse: bool = False,
):
    """CUDA attention: flex_attention with optional sink-attention LSE output.

    Args:
        q, k, v: (B, H, S, D) — must already be on CUDA.
        block_mask: ``BlockMask`` or ``None``.
        compile: use the pre-compiled flex_attention kernel.
        return_lse: also return log-sum-exp for sink attention.

    Returns:
        output tensor, or (output, lse) when ``return_lse=True``.
    """
    _flex = _compiled_flex() if compile else flex_attention
    if return_lse:
        return _flex(q, k, v, block_mask=block_mask, return_aux=AuxRequest(lse=True))
    return _flex(q, k, v, block_mask=block_mask)


@functools.lru_cache(maxsize=1)
def _compiled_flex():
    return torch.compile(flex_attention, mode="max-autotune-no-cudagraphs")
