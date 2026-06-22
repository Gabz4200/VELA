# Utilities for Falcon Vision
# Model loading and image preprocessing without tokenizer dependency

from typing import Union

import torch

from .configs import siglino_configs
from .image_processor import SigLinoImageProcessor
from .model import SigLino


def _quantize_model_if_needed(
    model: SigLino, quantize: bool, device: Union[str, torch.device]
) -> SigLino:
    if not quantize:
        return model
    dev_str = str(device)
    try:
        from torchao.quantization import quantize_

        if "cuda" in dev_str:
            from torchao.quantization import Int4WeightOnlyConfig

            print("Quantizing model to CUDA int4 with torchao...")
            quantize_(
                model,
                Int4WeightOnlyConfig(
                    group_size=32,
                    int4_packing_format="tile_packed_to_4d",
                    int4_choose_qparams_algorithm="hqq",
                ),
            )  # type: ignore[arg-type]
            print("Model quantized to CUDA int4 successfully.")
        else:
            from torchao.quantization import Int8WeightOnlyConfig

            print("Quantizing model to CPU int8 with torchao...")
            quantize_(model, Int8WeightOnlyConfig())
            print("Model quantized to CPU int8 successfully.")
    except ImportError:
        print("torchao is not installed or not available. Skipping quantization.")
    return model


def load_siglino_model(
    checkpoint_path: str,
    config_name: str = "siglino-0.3B",
    device: Union[str, torch.device] = "cuda",
    dtype: torch.dtype | None = None,
    quantize: bool = False,
    **kwargs,
) -> tuple[SigLino, SigLinoImageProcessor]:
    """
    Load a SigLino model from a checkpoint.

    Args:
        checkpoint_path: Path to the model checkpoint
        config_name: Name of the model configuration
        device: Device to load the model on
        dtype: Optional dtype to cast model weights to (e.g. torch.bfloat16)

    Returns:
        Tuple of (model, image_processor)
    """
    # Get configuration
    if config_name in siglino_configs:
        args = siglino_configs[config_name]
    else:
        raise ValueError(
            f"Unknown config: {config_name}. Available: {list(siglino_configs.keys())}"
        )

    # Create model
    model = SigLino(args)

    # Standard PyTorch checkpoint
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model.load_state_dict(state_dict)

    if dtype is None:
        model = model.to(device=device)
    else:
        model = model.to(device=device, dtype=dtype)
    model.eval()

    model = _quantize_model_if_needed(model, quantize, device)

    # Create image processor
    image_processor = SigLinoImageProcessor(patch_size=args.spatial_patch_size, **kwargs)

    return model, image_processor


# def convert_torchtitan_checkpoint(
#     torchtitan_ckpt_path: str,
#     output_path: str,
#     config_name: str = "0.25B-1B-a-tall-se-24l16e-route-distillation",
# ):
#     """
#     Convert a torchtitan checkpoint to standalone format.
#
#     This handles the key mapping differences between the torchtitan
#     DistillPerceptionTransformerMultiTeacher and FalconVisionEncoder.
#     """
#     # Load torchtitan checkpoint
#     if os.path.isdir(torchtitan_ckpt_path):
#         from torch.distributed.checkpoint import load as dcp_load
#         config = omni_falcon_perception_configs[config_name]
#         config.max_seq_len = 2048
#         config.seq_len = 2304 + 5
#         config.vocab_size = 65536
#         config.eos_id = 31999
#         config.dtype = torch.bfloat16
#         config.use_grouped_mm = False
#         config.use_flex_attn = True
#         config.attn_mask_type = "distill_mask"
#         config.img_start_id = 31998
#         config.img_end_id = 31997
#         config.img_id = 31996
#         config.eager = True
#         config.n_storage_tokens = 4
#         config.img_row_sep_id = 31995
#         config.vid_start_id = 31994
#         config.vid_end_id = 31993
#         config.frame_sep_id = 31992
#         config.image_mask_token_id = 31991
#         config.image_cls_token_id = 31990
#         config.image_reg_1_token_id = 31989
#         config.image_reg_2_token_id = 31988
#         config.image_reg_3_token_id = 31987
#         config.image_reg_4_token_id = 31986
#         config.cls_weight = 0
#         config.patch_weight = 0
#         config.storage_weight = 0
#         config.pairwise_distance_weight = 0
#         config.pairwise_cosine_weight = 0
#         config.pairwise_distance_patch_weight = 0
#         config.pairwise_cosine_patch_weight = 0
#         config.high_res_distillation_weight = 0
#         config.teachers = ("siglip2", "dinov3")
#         config.teachers_dim = (1152, 1024)
#         config.optimizable_teachers = ("siglip2", "dinov3")
#         config.average_patch_loss = False
#         config.weighted_patch_loss = False
#         config.jitter_rope = False
#         config.use_phis = False
#         config.use_pixel_head = True
#
#         # Load model
#         model = DistillPerceptionTransformerMultiTeacher(config).to("cuda")
#         state_dict = model.state_dict()
#         state_dict.pop('freqs_cis', None)
#         keys = list(state_dict.keys())
#         for k in keys:
#             if "coord" in k:
#                 state_dict.pop(k, None)
#             if "size" in k:
#                 state_dict.pop(k, None)
#             if "proj_segm" in k:
#                 state_dict.pop(k, None)
#             if "itok_upsampler" in k:
#                 state_dict.pop(k, None)
#             if "rope_upsampler" in k:
#                 state_dict.pop(k, None)
#
#         dcp_load(state_dict, checkpoint_id=torchtitan_ckpt_path)
#     else:
#         state_dict = torch.load(torchtitan_ckpt_path, map_location="cpu", weights_only=False)
#         if "model" in state_dict:
#             state_dict = state_dict["model"]
#
#     # Key mapping from torchtitan to standalone
#     key_map = {
#         "tok_embeddings": None,  # Remove text embeddings
#         "output": None,  # Remove text output
#         "pixel_mlp": None,  # Remove pixel head
#         "proj_segm": None,  # Remove segmentation head
#         "itok_upsampler": None,  # Remove upsampler
#         "coord_encoder": None,  # Remove coordinate heads
#         "coord_decoder": None,
#         "size_encoder": None,
#         "size_decoder": None,
#         "phis_statistics": None,  # Remove PHIs statistics
#         "rope_upsampler": None,  # Remove RoPE upsampler
#     }
#
#     new_state_dict = {}
#     for k, v in state_dict.items():
#         # Skip keys that should be removed
#         skip = False
#         for prefix in key_map.keys():
#             if k.startswith(prefix) or k.startswith(f"model.{prefix}"):
#                 skip = True
#                 break
#         if skip:
#             continue
#
#         # Remove "model." prefix if present
#         new_key = k[6:] if k.startswith("model.") else k
#         print(new_key)
#         new_state_dict[new_key] = v
#
#     # Save converted checkpoint
#     torch.save(new_state_dict, output_path)
#     print(f"Saved converted checkpoint to {output_path}")


# ── HuggingFace Hub loader ──────────────────────────────────────────────────

# Map from HF repo-id to local config name.
_HF_REPO_TO_CONFIG: dict[str, str] = {
    "tiiuae/siglino-0.6B": "dense-0.6B",
    "tiiuae/siglino-30M": "dense-30M",
    "tiiuae/siglino-70M": "dense-70M",
}


def load_siglino_from_hub(
    repo_id: str,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype | None = None,
    config_name: str | None = None,
    quantize: bool = False,
    **processor_kwargs,
) -> tuple["SigLino", SigLinoImageProcessor]:
    """
    Load a SigLino model from the HuggingFace Hub without ``trust_remote_code``.

    Uses the locally-vendored model code in ``src/siglino/`` and fetches only
    the weight file from the Hub via ``huggingface_hub``.

    Args:
        repo_id:        HF repo, e.g. ``"tiiuae/siglino-0.6B"``.
        device:         Target device (``"cpu"`` / ``"cuda"`` / …).
        dtype:          Optional dtype cast (e.g. ``torch.bfloat16``).
        config_name:    Override the auto-detected config key.
        **processor_kwargs: Extra kwargs forwarded to ``SigLinoImageProcessor``.

    Returns:
        ``(model, image_processor)``
    """
    from huggingface_hub import hf_hub_download

    if config_name is None:
        config_name = _HF_REPO_TO_CONFIG.get(repo_id)
        if config_name is None:
            raise ValueError(
                f"Unknown HF repo '{repo_id}'. "
                f"Either add it to _HF_REPO_TO_CONFIG or pass config_name explicitly."
            )

    if config_name not in siglino_configs:
        raise ValueError(
            f"Unknown config '{config_name}'. Available: {list(siglino_configs.keys())}"
        )

    args = siglino_configs[config_name]

    # Try safetensors first, fall back to .pt
    try:
        ckpt_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
        state_dict = _load_safetensors(ckpt_path)
    except Exception:
        ckpt_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = SigLino(args)
    model.load_state_dict(state_dict, strict=True)

    if dtype is None:
        model = model.to(device=device)
    else:
        model = model.to(device=device, dtype=dtype)
    model.eval()

    model = _quantize_model_if_needed(model, quantize, device)

    image_processor = SigLinoImageProcessor(patch_size=args.spatial_patch_size, **processor_kwargs)
    return model, image_processor


def _load_safetensors(path: str) -> dict:
    """Load a safetensors file into a plain state-dict."""
    try:
        from safetensors.torch import load_file

        return load_file(path, device="cpu")
    except ImportError:
        raise ImportError(
            "Install 'safetensors' to load .safetensors checkpoints: pip install safetensors"
        )


# Feature dimension constants
FEATURE_DIM_DICT = {
    "dinov3": 1024,
    "siglip2": 1152,
    "siglino": 768,  # Model dimension
}

PATCH_SIZE = 16
