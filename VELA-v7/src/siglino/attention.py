# Attention module for SigLino
# GPU path : flex_attention (Triton kernel, with sink-attention via LSE aux)
# CPU path : compiled SDPA   (oneDNN/MKL-DNN back-end, no flex required)

import einops as E
import torch
import torch.nn.functional as F
from torch import nn

from .kernels.cpu_attn import cpu_sdpa
from .rope import apply_3d_rotary_emb

# Lazy flex_attention imports — only available when CUDA is compiled in
try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask

    from .kernels.cuda_attn import cuda_flex_attn

    _FLEX_AVAILABLE = True
except ImportError:
    _FLEX_AVAILABLE = False
    BlockMask = None  # type: ignore[assignment,misc]
    create_block_mask = None  # type: ignore[assignment]
    cuda_flex_attn = None  # type: ignore[assignment]


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads to match query heads."""
    if n_rep == 1:
        return x
    return torch.repeat_interleave(x, repeats=n_rep, dim=2)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        head_dim: int | None = None,
        use_qk_norm: bool = False,
        enable_3d_rope: bool = False,
        use_flex_attn: bool = True,
        use_sink_attn: bool = True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = head_dim or dim // n_heads
        self.q_dim = self.n_heads * self.head_dim
        self.kv_dim = self.n_kv_heads * self.head_dim

        self.wq = nn.Linear(dim, self.q_dim, bias=False)
        self.wk = nn.Linear(dim, self.kv_dim, bias=False)
        self.wv = nn.Linear(dim, self.kv_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        self.use_qk_norm = use_qk_norm
        self.enable_3d_rope = enable_3d_rope
        # use_flex_attn is only honoured when flex_attention is actually available
        self.use_flex_attn = use_flex_attn and _FLEX_AVAILABLE

        # Sink attention requires flex_attention (needs lse aux output)
        self.sink_attn = use_sink_attn and self.use_flex_attn
        if self.sink_attn:
            self.sinks = nn.Parameter(torch.empty(n_heads))


    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
        nn.init.zeros_(self.wo.weight)
        if self.sink_attn:
            nn.init.trunc_normal_(self.sinks, mean=0.0, std=init_std)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        freqs_cis_2d: torch.Tensor | None = None,
        pos_thw: torch.Tensor | None = None,
        attention_masks=None,  # BlockMask | torch.Tensor | None
        compile: bool = True,
    ) -> torch.Tensor:
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        if self.use_qk_norm:
            xq = F.rms_norm(xq, (xq.size(-1),))
            xk = F.rms_norm(xk, (xk.size(-1),))

        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xq, xk = apply_3d_rotary_emb(xq, xk, freqs_cis, freqs_cis_2d, pos_thw)

        xq = xq.transpose(1, 2)  # (B, H, S, D)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        if self.use_flex_attn and _FLEX_AVAILABLE and cuda_flex_attn is not None and xq.is_cuda:
            if self.sink_attn:
                output, aux = cuda_flex_attn(
                    xq,
                    xk,
                    xv,
                    block_mask=attention_masks,
                    compile=compile,
                    return_lse=True,
                )
                sinks_BHL = E.rearrange(self.sinks, "h -> 1 h 1")
                sink_scale = torch.sigmoid(aux.lse - sinks_BHL)
                output = (output * sink_scale.unsqueeze(-1)).to(output.dtype)
            else:
                output = cuda_flex_attn(
                    xq,
                    xk,
                    xv,
                    block_mask=attention_masks,
                    compile=compile,
                    return_lse=False,
                )
        else:
            # CPU path: compiled SDPA kernel (oneDNN/MKL-DNN, no flex)
            attn_mask = None
            if attention_masks is not None and isinstance(attention_masks, torch.Tensor):
                attn_mask = attention_masks
            output = cpu_sdpa(xq, xk, xv, attn_mask=attn_mask)

        output = E.rearrange(output, "b h s d -> b s (h d)").contiguous()
        return self.wo(output)


def create_attention_mask(
    mask_mod,
    B: int | None,
    H: int | None,
    Q_LEN: int,
    KV_LEN: int,
    BLOCK_SIZE: tuple[int, int] = (64, 64),
):
    """Create a BlockMask for flex_attention. Returns None if flex is unavailable."""
    if not _FLEX_AVAILABLE or create_block_mask is None:
        return None
    return create_block_mask(
        mask_mod,
        B=B,
        H=H,
        Q_LEN=Q_LEN,
        KV_LEN=KV_LEN,
        BLOCK_SIZE=BLOCK_SIZE,
    )
