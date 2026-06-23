"""VLA (Vision-Language-Action) model with NitroGen-style Flow Matching."""

import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Beta
from torch.nn import functional as F

from ..dataset import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from .vlm import CHUNK_LEN, VLM, L2Wrap, block_attn_res

try:
    import deepspeed
except ImportError:
    deepspeed = None


class SinusoidalTimestepEncoder(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        half_dim = embedding_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim) / half_dim)
        self.register_buffer("freqs", freqs)

    def forward(self, timesteps: Tensor) -> Tensor:
        args = timesteps.unsqueeze(-1) * self.freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimestepMLP(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(embedding_dim * 4, embedding_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class AdaLayerNorm(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(hidden_size, hidden_size * 2)
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        scale, shift = self.linear(self.silu(temb)).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """NitroGen-style block: self-attn (AdaNorm) + cross-attn on residual + FFN."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.norm1 = AdaLayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, x: Tensor, cond: Tensor, temb: Tensor) -> Tensor:
        h = self.norm1(x, temb)
        x = x + self.self_attn(h, h, h)[0]
        x = x + self.cross_attn(self.norm2(x), cond, cond)[0]
        x = x + self.ffn(self.norm3(x))
        return x


class FlowMatchingHead(nn.Module):
    """NitroGen-style Flow Matching DiT with Attention Residual conditioning."""

    def __init__(
        self,
        hidden_size: int,
        action_dim: int,
        action_horizon: int = 16,
        num_layers: int = 4,
        num_heads: int = 8,
    ):
        super().__init__()
        self.action_horizon = action_horizon
        self.action_dim = action_dim

        self.action_embed = nn.Linear(action_dim, hidden_size)
        self.t_embed = nn.Sequential(
            SinusoidalTimestepEncoder(hidden_size),
            TimestepMLP(hidden_size),
        )

        self.cond_proj = nn.Linear(hidden_size, 1, bias=False)
        self.cond_norm = nn.LayerNorm(hidden_size)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(num_layers)
        ])

        self.norm_out = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.proj_out = nn.Linear(hidden_size, hidden_size * 2)
        self.velocity = nn.Linear(hidden_size, action_dim)

        self.beta_dist = Beta(1.5, 1.0)

    def forward(self, noise: Tensor, V_blocks: Tensor, partial_block: Tensor, t: Tensor | None = None) -> Tensor:
        B = noise.shape[0]
        device = noise.device

        if t is None:
            t = self.beta_dist.sample([B]).to(device)
            t = (1 - t) * 0.999
        elif t.dim() == 0:
            t = t.unsqueeze(0).expand(B)

        x = self.action_embed(noise)
        temb = self.t_embed(t)

        cond = block_attn_res(V_blocks, partial_block, self.cond_proj, self.cond_norm)
        cond = cond[:, -self.action_horizon:, :]

        for block in self.blocks:
            x = block(x, cond, temb)

        shift, scale = self.proj_out(F.silu(temb)).chunk(2, dim=-1)
        x = self.norm_out(x) * (1 + shift.unsqueeze(1)) + scale.unsqueeze(1)
        return self.velocity(x)

    @torch.inference_mode()
    def sample(self, V_blocks: Tensor, partial_block: Tensor, num_steps: int = 2) -> Tensor:
        B = partial_block.shape[0]
        device = partial_block.device
        dtype = partial_block.dtype

        x = torch.randn(B, self.action_horizon, self.action_dim, device=device, dtype=dtype)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full([B], i * dt, device=device, dtype=dtype)
            v = self.forward(x, V_blocks, partial_block, t=t)
            x = x + dt * v

        return x


def info_nce_loss(pred: Tensor, target: Tensor, negatives: Tensor, tau: float = 0.1) -> Tensor:
    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)
    neg_n = F.normalize(negatives, dim=-1)

    pos = (pred_n * target_n).sum(dim=-1, keepdim=True)
    neg = torch.matmul(pred_n, neg_n.transpose(-1, -2))

    logits = torch.cat([pos, neg], dim=-1) / tau
    labels = torch.zeros(logits.shape[:-1], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))


class VLA(VLM):
    """Vision-Language-Action model. Head 1 (InfoNCE+CE) and Head 2 (Flow Matching) are parallel.

    Both heads consume the Attention Residual stream (V_blocks) from all backbone layers.
    """

    def __init__(self, args):
        super().__init__(args)

        self.action_dim = getattr(args, "action_dim", 14)
        self.action_horizon = getattr(args, "action_horizon", 16)

        self.info_nce_weight = getattr(args, "info_nce_weight", 1.0)
        self.ce_weight = getattr(args, "ce_weight", 0.1)

        self.flow_head = FlowMatchingHead(
            hidden_size=args.n_embd,
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
        )

    @staticmethod
    def _unpad(x: Tensor, num_tokens: int) -> Tensor:
        if num_tokens > 0:
            return x[:, num_tokens:]
        return x

    def _info_nce_loss(self, residual: Tensor, input_embeds: Tensor) -> Tensor:
        pred = residual[:, :-1, :]
        target = residual[:, 1:, :]
        negatives = input_embeds[:, :-1, :]
        return info_nce_loss(pred, target, negatives)

    def _ce_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()

        image_mask = shift_labels == IMAGE_TOKEN_INDEX
        shift_labels = torch.where(image_mask, IGNORE_INDEX, shift_labels)

        valid_lengths = (shift_labels != IGNORE_INDEX).sum(1)
        valid_lengths = torch.max(valid_lengths, torch.ones_like(valid_lengths))

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        )
        return (loss.view(shift_labels.size()).sum(1) / valid_lengths).mean()

    def _flow_loss(self, V_blocks: Tensor, partial_block: Tensor, actions: Tensor) -> Tensor:
        noise = torch.randn_like(actions)
        velocity_pred = self.flow_head(noise, V_blocks, partial_block)
        return F.mse_loss(velocity_pred, actions - noise)

    def forward(self, samples):
        return super().forward(samples)

    def training_step(self, batch, batch_idx):
        x, targets = self.preparing_embedding(batch)
        B, T, D = x.shape

        num_tokens_to_pad = CHUNK_LEN - T % CHUNK_LEN if T % CHUNK_LEN != 0 else 0
        x = self.rwkv.pad_left(x, num_tokens_to_pad)
        if self.args.dropout > 0:
            x = self.rwkv.drop0(x)

        V_blocks = torch.empty(0, B, x.size(1), D, dtype=x.dtype, device=x.device)
        partial_block = x
        v_first = torch.empty_like(x)

        for block in self.rwkv.blocks:
            if self.args.grad_cp == 1 and deepspeed is not None:
                V_blocks, partial_block, v_first = deepspeed.checkpointing.checkpoint(
                    block, V_blocks, partial_block, v_first
                )
            else:
                V_blocks, partial_block, v_first = block(V_blocks, partial_block, v_first)

        residual = self.rwkv.ln_out(partial_block)
        logits = self.rwkv.head(residual)

        logits = self._unpad(logits, num_tokens_to_pad)
        residual = self._unpad(residual, num_tokens_to_pad)
        x = self._unpad(x, num_tokens_to_pad)

        info_nce = self._info_nce_loss(residual, x)
        ce_loss = self._ce_loss(logits, targets)

        flow_loss = torch.tensor(0.0, device=logits.device)
        if "actions" in batch:
            flow_loss = self._flow_loss(V_blocks, partial_block, batch["actions"])

        total_loss = self.info_nce_weight * info_nce + self.ce_weight * ce_loss + flow_loss

        self.log("vla/info_nce", info_nce, prog_bar=True)
        self.log("vla/ce", ce_loss, prog_bar=True)
        self.log("vla/flow", flow_loss, prog_bar=True)
        self.log("vla/total", total_loss)

        return L2Wrap.apply(total_loss, logits)
