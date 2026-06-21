import torch
import torch.nn as nn
import torch.nn.functional as F
import einops as E
from typing import Optional, Dict, Union, Tuple
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

# Relative imports from your local files
from .configuration_siglino import SigLinoConfig
from .attention import Attention, create_attention_mask
from .moe import MoE, FeedForward
from .rope import (
    precompute_freqs_cis,
    precompute_golden_freqs_cis,
    apply_golden_freqs_cis_to_visual_pos,
)

class Siglip2MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.activation_fn = nn.GELU(approximate="tanh")
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class Siglip2MultiheadAttentionPoolingHead(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, output_dim: int):
        super().__init__()
        self.probe = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.attention = nn.MultiheadAttention(hidden_size, num_attention_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(hidden_size, eps=1e-5)
        self.mlp = Siglip2MLP(hidden_size, 4304)
        self.num_heads = num_attention_heads

    def forward(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        if attention_mask is not None:
            # Mask expansion logic kept from your original model.py
            # Note: This uses einops and specific expansion for MHA
            def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: int | None = None):
                bsz, src_len = mask.size()
                tgt_len = tgt_len if tgt_len is not None else src_len
                expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
                inverted_mask = torch.tensor(1.0, dtype=dtype, device=mask.device) - expanded_mask
                return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

            attention_mask = E.rearrange(attention_mask, "(b s) -> b s", b=batch_size)
            target_len, source_len = probe.shape[1], hidden_state.shape[1]
            attention_mask = _expand_mask(attention_mask, hidden_state.dtype, target_len)
            attention_mask = attention_mask.repeat(1, self.num_heads, target_len, 1)
            attention_mask = attention_mask.reshape(-1, target_len, source_len)

        hidden_state = self.attention(probe, hidden_state, hidden_state, attn_mask=attention_mask)[0]
        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)
        return hidden_state[:, 0]

class Adapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, config: SigLinoConfig):
        super().__init__()
        self.dim = config.dim
        self.parameterized_norm = getattr(config, 'parameterized_norm', True)
        if self.parameterized_norm:
            self.attention_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
            self.ffn_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)

        self.attention = Attention(
            dim=config.dim,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            use_qk_norm=config.use_qk_norm,
            enable_3d_rope=config.enable_3d_rope,
            use_flex_attn=config.use_flex_attn,
            use_sink_attn=True,
        )

        # Handle MoE initialization from config dict
        moe_args = config.moe_args
        if isinstance(moe_args, dict):
            from .moe import MoEArgs
            moe_args = MoEArgs(**moe_args)

        first_n_dense = getattr(config, 'first_n_layers_dense', 0)
        use_dense = layer_id < first_n_dense
        if use_dense:
            ffn_hidden = getattr(config, 'ffn_dim', None) or config.moe_dim
            activation = getattr(config, 'activation', 'silu')
            self.feed_forward = FeedForward(config.dim, ffn_hidden, activation=activation)
            self.moe_enabled = False
        elif moe_args and moe_args.num_experts > 0:
            self.moe = MoE(moe_args, dim=config.dim, hidden_dim=config.moe_dim)
            self.moe_enabled = True
        else:
            self.feed_forward = FeedForward(config.dim, config.moe_dim)
            self.moe_enabled = False

        self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5

    def forward(self, x, freqs_cis, freqs_cis_2d=None, pos_thw=None, attention_masks=None, compile=False):
        if self.parameterized_norm:
            x_norm = self.attention_norm(x)
        else:
            x_norm = F.rms_norm(x, (x.size(-1),))
        h = x + self.attention(
            x_norm,
            freqs_cis,
            freqs_cis_2d,
            pos_thw,
            attention_masks=attention_masks,
            compile=compile,
        )
        h_norm = self.ffn_norm(h) if self.parameterized_norm else F.rms_norm(h, (h.size(-1),))
        out = h + self.moe(h_norm) if self.moe_enabled else h + self.feed_forward(h_norm)
        return out

class SigLinoPreTrainedModel(PreTrainedModel):
    config_class = SigLinoConfig
    base_model_prefix = "siglino"
    main_input_name = "pixel_values"
    _no_split_modules = ["TransformerBlock"]

    def _init_weights(self, module):
        # Weight initialization is handled by the internal init_weights call in __init__
        pass

    def _apply(self, fn):
        # Prevent casting complex RoPE buffers (freqs_cis) to real dtypes on model.to(bf16/fp16)
        complex_buffers = {}
        for name, buf in list(self.named_buffers(recurse=False)):
            if buf is not None and buf.is_complex():
                complex_buffers[name] = buf
                del self._buffers[name]

        ret = super()._apply(fn)

        for name, buf in complex_buffers.items():
            dummy = torch.tensor([0.0], device=buf.device)
            res = fn(dummy)

            if not res.is_complex():
                new_buf = buf.to(device=res.device)
            else:
                new_buf = fn(buf)

            persistent = name not in self._non_persistent_buffers_set
            self.register_buffer(name, new_buf, persistent=persistent)

        return ret


class SigLinoModel(SigLinoPreTrainedModel):
    def __init__(self, config: SigLinoConfig):
        super().__init__(config)
        self.config = config
        self.n_layers = config.n_layers
        self.patch_size = config.spatial_patch_size
        self.n_storage_tokens = config.n_storage_tokens

        # Patch embedding
        self.n_pixels_per_patch = config.temporal_patch_size * config.spatial_patch_size ** 2
        self.img_projector = nn.Linear(
            self.n_pixels_per_patch * config.channel_size,
            config.dim,
            bias=False,
        )

        self.cls_token = nn.Parameter(torch.empty(1, 1, config.dim))
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, self.n_storage_tokens, config.dim))

        # RoPE
        head_dim = config.head_dim or config.dim // config.n_heads
        d = head_dim // 2
        self.register_buffer("freqs_cis_golden", self._precompute_golden_freqs_cis(d, config))
        self.register_buffer("freqs_cis", self._precompute_freqs_cis(d, config), persistent=False)

        self.layers = nn.ModuleList([TransformerBlock(i, config) for i in range(config.n_layers)])
        self.norm = nn.RMSNorm(config.dim, eps=config.norm_eps)

        # Teacher adapters
        teachers_dict = dict(zip(config.teachers, config.teachers_dim))
        dinov3_dim = teachers_dict.get("dinov3", 1280)
        siglip2_dim = teachers_dict.get("siglip2", 1152)

        self.dinov3_adapter = Adapter(config.dim, dinov3_dim, bias=False)
        self.siglip2_adapter = Adapter(config.dim, siglip2_dim, bias=False)
        self.layer_norm_dinov3 = nn.LayerNorm(dinov3_dim)
        self.siglip2_multihead_attention_pooling_head = Siglip2MultiheadAttentionPoolingHead(
            siglip2_dim, 16, siglip2_dim
        )

        self.post_init()

    def _precompute_freqs_cis(self, head_dim: int, config: SigLinoConfig) -> torch.Tensor:
        return precompute_freqs_cis(head_dim, config.max_seq_len, config.rope_theta)

    def _precompute_golden_freqs_cis(self, head_dim: int, config: SigLinoConfig) -> torch.Tensor:
        return precompute_golden_freqs_cis(
            config.n_heads, head_dim, config.rope_min_freqs, config.rope_max_freqs
        )

    def _get_thw_pos(self, batch_size, num_patches, spatial_shapes, device):
        N = batch_size
        R = 1 + self.n_storage_tokens
        S = R + num_patches
        tpos = torch.zeros((N, S), dtype=torch.float32, device=device)
        hpos = torch.zeros((N, S), dtype=torch.float32, device=device)
        wpos = torch.zeros((N, S), dtype=torch.float32, device=device)

        for n in range(N):
            H, W = spatial_shapes[n].tolist()
            h_coords = torch.arange(H, device=device).float()
            w_coords = torch.arange(W, device=device).float()
            xlim, ylim = (W / H) ** 0.5, (H / W) ** 0.5
            h_norm = -ylim + 2 * ylim * h_coords / max(H - 1, 1)
            w_norm = -xlim + 2 * xlim * w_coords / max(W - 1, 1)
            
            # Vectorized fill for patches
            h_grid, w_grid = torch.meshgrid(h_norm, w_norm, indexing='ij')
            hpos[n, R:R+H*W] = h_grid.reshape(-1)
            wpos[n, R:R+H*W] = w_grid.reshape(-1)
            
            hpos[n, :R], wpos[n, :R] = float('nan'), float('nan')

        return torch.stack([tpos, hpos, wpos], dim=0)

    def forward(
        self,
        pixel_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        compile: bool = True,
    ) -> Union[Dict, Tuple]:
        N, L, _ = pixel_values.shape
        device = pixel_values.device
        R = 1 + self.n_storage_tokens

        if padding_mask is None:
            padding_mask = torch.ones((N, L), dtype=pixel_values.dtype, device=device)
        
        h_NLD = self.img_projector(pixel_values)
        cls_expanded = self.cls_token.expand(N, -1, -1)
        if self.n_storage_tokens > 0:
            reg_expanded = self.storage_tokens.expand(N, -1, -1)
            h_NSD = torch.cat([cls_expanded, reg_expanded, h_NLD], dim=1)
        else:
            h_NSD = torch.cat([cls_expanded, h_NLD], dim=1)

        S = h_NSD.shape[1]
        cls_reg_mask = torch.ones((N, R), dtype=padding_mask.dtype, device=device)
        full_mask = torch.cat([cls_reg_mask, padding_mask], dim=1)
        
        # FlexAttention Mask
        def mask_mod(b, h, q_idx, kv_idx):
            return full_mask.bool()[b, q_idx] & full_mask.bool()[b, kv_idx]
        
        block_mask = create_attention_mask(mask_mod, N, None, S, S)

        # RoPE
        thw_pos = self._get_thw_pos(N, L, spatial_shapes, device)
        pos_thw = E.rearrange(thw_pos, "p n s -> n s p").to(dtype=torch.float32)
        patch_mask_2d = torch.zeros((N, S), dtype=torch.bool, device=device)
        patch_mask_2d[:, R:] = padding_mask.bool()
        pos_thw[:, :, 1:] = pos_thw[:, :, 1:].masked_fill(~patch_mask_2d.unsqueeze(-1), float("nan"))
        
        freqs_cis_golden = apply_golden_freqs_cis_to_visual_pos(
            self.freqs_cis_golden.to(dtype=pos_thw.dtype), pos_thw[:, :, 1:]
        )

        all_hidden_states = () if output_hidden_states else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (h_NSD,)
            h_NSD = layer(h_NSD, self.freqs_cis, freqs_cis_2d=freqs_cis_golden, 
                          pos_thw=pos_thw, attention_masks=block_mask, compile=compile)

        h_NSD = self.norm(h_NSD)
        
        # Feature Extraction & Adapters
        cls_feats = h_NSD[:, 0]
        patch_feats = h_NSD[:, R:]
        
        student_patch_dinov3 = self.dinov3_adapter(patch_feats)
        student_patch_siglip = self.siglip2_adapter(patch_feats)
        student_cls_dinov3 = self.dinov3_adapter(cls_feats)

        h_sig = self.siglip2_adapter(h_NSD)
        siglip_attn_mask = full_mask.reshape(-1)
        student_summary_siglip = self.siglip2_multihead_attention_pooling_head(h_sig, siglip_attn_mask)

        output = {
            "last_hidden_state": h_NSD,
            "patch_features": {
                "dinov3": student_patch_dinov3,
                "siglip2": student_patch_siglip,
                "siglino": patch_feats,
            },
            "summary_features": {
                "dinov3": student_cls_dinov3,
                "siglip2": student_summary_siglip,
                "siglino": cls_feats,
            },
            "hidden_states": all_hidden_states,
        }

        if not return_dict:
            return tuple(v for v in output.values() if v is not None)
        return output