import torch

from .configs import siglino_configs
from .image_processor import SigLinoImageProcessor
from .model import SigLino


def _quantize_model_if_needed(
    model: SigLino, quantize: bool, device: str | torch.device
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
            quantize_(model, Int8WeightOnlyConfig(version=2))
            print("Model quantized to CPU int8 successfully.")
    except ImportError:
        print("torchao is not installed or not available. Skipping quantization.")
    return model


def load_siglino_model(
    checkpoint_path: str,
    config_name: str = "siglino-0.3B",
    device: str | torch.device = "cuda",
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


# Map from HF repo-id to local config name.
_HF_REPO_TO_CONFIG: dict[str, str] = {
    "tiiuae/siglino-0.6B": "dense-0.6B",
    "tiiuae/siglino-30M": "dense-30M",
    "tiiuae/siglino-70M": "dense-70M",
}


def load_siglino_from_hub(
    repo_id: str,
    device: str | torch.device = "cpu",
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
