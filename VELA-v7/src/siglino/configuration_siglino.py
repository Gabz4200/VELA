from typing import Dict, Optional, Tuple

from transformers import PretrainedConfig


class SigLinoConfig(PretrainedConfig):
    """
    Configuration class to store the configuration of an `SigLinoModel`.
    """

    model_type = "siglino"

    def __init__(
        self,
        dim: int = 768,
        n_layers: int = 18,
        n_heads: int = 12,
        head_dim: Optional[int] = 128,
        n_kv_heads: Optional[int] = 4,
        # MoE configuration
        moe_dim: int = 768,
        moe_args: Optional[Dict] = None,
        # Dense FFN configuration
        first_n_layers_dense: int = 0,
        ffn_dim: Optional[int] = None,
        activation: str = "silu",
        # Vision settings
        channel_size: int = 3,
        spatial_patch_size: int = 16,
        temporal_patch_size: int = 1,
        # RoPE settings
        enable_3d_rope: bool = True,
        rope_theta: float = 100000.0,
        rope_min_freqs: float = 1.0,
        rope_max_freqs: float = 20.0,
        max_seq_len: int = 8192,
        # Normalization
        norm_eps: float = 1e-5,
        use_qk_norm: bool = True,
        use_tok_norm: bool = True,
        parameterized_norm: bool = True,
        # Distillation settings
        n_storage_tokens: int = 4,
        teachers: Tuple[str, ...] = ("siglip2", "dinov3"),
        teachers_dim: Tuple[int, ...] = (1152, 1024),
        # FlexAttention
        use_flex_attn: bool = True,
        **kwargs,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads

        self.moe_dim = moe_dim
        # Default MoEArgs matching your configs.py
        self.moe_args = (
            moe_args
            if moe_args is not None
            else {
                "num_experts": 16,
                "num_shared_experts": 1,
                "top_k": 3,
                "score_before_experts": False,
                "route_norm": True,
                "route_scale": 0.8633,
                "activation": "relu2",
                "score_func": "sigmoid",
            }
        )

        self.first_n_layers_dense = first_n_layers_dense
        self.ffn_dim = ffn_dim
        self.activation = activation

        self.channel_size = channel_size
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size

        self.enable_3d_rope = enable_3d_rope
        self.rope_theta = rope_theta
        self.rope_min_freqs = rope_min_freqs
        self.rope_max_freqs = rope_max_freqs
        self.max_seq_len = max_seq_len

        self.norm_eps = norm_eps
        self.use_qk_norm = use_qk_norm
        self.use_tok_norm = use_tok_norm
        self.parameterized_norm = parameterized_norm

        self.n_storage_tokens = n_storage_tokens
        self.teachers = teachers
        self.teachers_dim = teachers_dim

        self.use_flex_attn = use_flex_attn

        super().__init__(**kwargs)
