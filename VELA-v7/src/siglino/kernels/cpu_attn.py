"""CPU-optimised scaled dot-product attention kernel for SigLino.

Uses ``torch.compile(mode="reduce-overhead")`` applied only to the SDPA
function — not to the whole model — so we avoid the Inductor lowering path
that tries to trace ``flex_attention`` (unsupported on CPU).
"""

import functools

import torch
import torch.nn.functional as F


@functools.lru_cache(maxsize=1)
def _get_compiled_cpu_sdpa():
    """Lazily compile the SDPA kernel once and cache it."""

    def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

    return torch.compile(_sdpa, mode="reduce-overhead", fullgraph=True)


def cpu_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """CPU attention: torch.compile'd SDPA (oneDNN/MKL-DNN back-end).

    Args:
        q: (B, H, S, D)
        k: (B, H, S, D)
        v: (B, H, S, D)
        attn_mask: optional dense bool or additive mask

    Returns:
        output: (B, H, S, D)
    """
    return _get_compiled_cpu_sdpa()(q, k, v, attn_mask)
