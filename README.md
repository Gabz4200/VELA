# VELA: Visual Early-fusion Language-model for Action

<p align="center">
  <img src="./VELA-arch.png" alt="VELA Architecture" width="800">
</p>

VELA extends the RWKV language model with early visual fusion and action output, enabling a single recurrent model to perceive, reason, and act across embodied tasks.

> ⚠️ **DISCLAIMER 1:** This repository is a learning project made by a single Brazilian student that is exploring the design space of RWKV-based vision backbones. The architecture started as a port from VisualRWKV-7 to use Attention Residuals and slowly transition it to a Vision Language Action Model (VLA), but I am not restricted to that and I might explore multiple design paths.

> ⚠️ **DISCLAIMER 2:** All the ideas behind what to do for this architecture are mine, but AI is still used in this project, mainly for those distinct tasks: commit message writing and automatic commit splitting, batch code writing for repetitive chores and helper routines. Parts of this README may be written by AI too as I usually ask it to compile information from the results of tests that I do.

## Description

VELA is a visual language model built on [RWKV-7](https://github.com/BlinkDL/RWKV-LM), a recurrent neural network architecture with linear-complexity inference. Unlike conventional vision-language models that use cross-attention between a vision encoder and a language decoder, VELA fuses visual tokens directly into the RWKV embedding space before the recurrent stack processes the sequence. This early visual fusion lets the RWKV's recurrent state jointly encode vision and language context from step one.

The architecture combines a multi-scale vision backbone (SAM, DINOv2, SigLIP) with an RWKV-7 language model, and (in v7.10+) adds action output heads that extend the model from visual perception and reasoning to closed-loop control for embodied AI tasks.

**VELA-MultiImage**: The data format supports multiple images per sample with dynamic image feature insertion, enabling multi-image comprehension, video frame processing, and document understanding from image splits.

## Key Features

- **Multi-Scale Vision Backbone**: Combines SAM (1024px), DINOv2 (448px), and **SigLino** (from tiiuae/siglino) features for rich, multi-scale visual representations. SigLino is locally vendored and uses custom optimized kernels (compiled SDPA on CPU, flex_attention on CUDA).
- **Weight Quantization**: Integrates optional **torchao** weight quantization support (CUDA int4 weight-only and CPU int8 weight-only) for efficient low-memory footprint.
- **Early Visual Fusion & In-place Visual Tokens**: Visual tokens are wrapped with `<img_start>` and `<img_end>` and injected in-place in the token sequence directly at their occurrences in documents/conversations, enabling dynamic early fusion of document images and text.
- **Multi-Image Support**: The model supports any number of images per sample, with dynamic image feature insertion from single image splits, multi-images, and video frames.
- **MHC MoE Layers (Layers 0-3)**: The first 4 block layers of the recurrent stack are configured as Dense MoE layers with 4 FFN experts. Routing is computed from WKV head pre-output projections using RMSNorm, linear projections, and a 20-iteration Sinkhorn-Knopp doubly stochastic normalization loop, following the Manifold-Constrained Hyper-Connections (mHC) formulation.
- **ChatML Suffix Formatting**: Standardizes the chat format to a customizable metadata ChatML template (`<im_start>{role}:{metadata}\n{content}<im_end>\n`) with dynamic target masking that only trains on assistant responses and dynamically masks speaker headers.
- **Linear-Time Inference**: Inherits RWKV's O(n) time complexity and O(1) memory — no quadratic attention bottleneck.
- **Block Attention Residuals**: Replaces standard additive residual connections with **Block AttnRes**, which partitions layers into chunks and uses a learned, input-dependent cross-layer attention mechanism to selectively aggregate previous representations, solving the PreNorm hidden-state dilution problem.
- **Multi-Resolution Support**: Dynamic tile splitting processes images at multiple aspect ratios (1:1, 1:2, 2:1, 1:3, 3:1).
- **VLA Action Head (v7.10+)**: Two parallel heads extend the model from perception and reasoning to action prediction:
  - **Head 1 (JEPA World Model)**: InfoNCE contrastive loss on all tokens + Cross-Entropy on non-image tokens, forcing the model to learn physics, dynamics, and language structure.
  - **Head 2 (Flow Matching Motor Controller)**: NitroGen-style DiT with Attention Residual conditioning from all backbone layers, predicting 16-step action chunks via flow-matching ODE solving.
- **Distributed Training**: Built on PyTorch Lightning with DeepSpeed ZeRO for multi-GPU training across model scales.
- **CUDA-Optimized WKV Kernel**: Custom WindBackstepping CUDA kernel for efficient RWKV-7 recurrence on GPU.

## Project Structure

```
VELA/
├── README.md                    # This file
├── AGENTS.md                    # Agent/LLM guidelines for the codebase
├── pyproject.toml               # Project configuration
├── LICENSE                      # Apache 2.0
├── VELA-arch.png                # Architecture diagram
├── rwkv_emoji.png               # Logo
├── VELA-v7/                     # VELA models based on RWKV-7
│   ├── src/
│   │   ├── models/
│   │   │   ├── __init__.py      # Exports VLM, VLA
│   │   │   ├── vlm.py           # VLM base model (RWKV, Block, MHCBlock, VLM)
│   │   │   └── vla.py           # VLA model (InfoNCE+CE Head 1 + Flow Matching Head 2)
│   │   ├── model.py             # Facade — re-exports from models/ for backward compat
│   │   ├── dataset.py           # Multi-modal dataset and tokenization
│   │   ├── trainer.py           # Training loop and LR schedule callbacks
│   │   ├── config.py            # Vision tower checkpoint paths
│   │   └── utils.py             # Utility functions
│   │   └── siglino/              # Locally-vendored Falcon Vision (SigLino)
│   │       └── kernels/          # CPU + CUDA attention kernels
│   ├── app/                     # Inference demo / serving app
│   ├── eval/                    # Benchmark evaluation tools (PCA visualization)
│   ├── tests/                   # Integration and unit tests
│   │   ├── test_with_weights.py # Weight-based regression tests
│   │   ├── test_moe_early_fusion.py  # MoE + ChatML tests
│   │   └── test_vla.py          # VLA component tests (InfoNCE, FlowMatchingHead)
│   ├── train.py                 # Training entry point
│   ├── evaluate.py              # Local evaluation entry point
│   └── tokenizer/               # RWKV tokenizer data
├── cuda/                        # CUDA kernels (wkv7)
│   ├── wkv7_cuda.cu
│   └── wkv7_op.cpp
└── download_huggingface.py      # HuggingFace model download
```

## VLA Architecture

The VLA extends VLM with two parallel heads that both consume the **Attention Residual** stream from all RWKV backbone layers:

```
┌─────────────────────────────────────────────────────┐
│                  Input Sequence                      │
│  <img> [VisionTokens] </img> <text> <act>           │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              RWKV Backbone (VLM)                     │
│  Blocks 0-3: MHC MoE (4 experts each)               │
│  Blocks 4+: Standard RWKV-7 blocks                  │
│  → V_blocks (saved layer states)                    │
│  → partial_block (final hidden state)               │
└────┬──────────────────────────────────┬─────────────┘
     │                                  │
     ▼                                  ▼
┌──────────────┐               ┌──────────────────┐
│  Head 1      │               │  Head 2          │
│  World Model │               │  Motor Controller│
├──────────────┤               ├──────────────────┤
│ InfoNCE      │               │ Flow Matching    │
│ (all tokens) │               │ DiT (NitroGen)   │
│ + CE         │               │                  │
│ (non-image)  │               │ 16-step action   │
│              │               │ chunk prediction │
└──────────────┘               └──────────────────┘
```

**Head 1 (JEPA-style World Model):**
- InfoNCE contrastive loss on all tokens — learns smooth continuous representations
- Cross-Entropy loss on non-image tokens only (masked on image tokens) — ensures discrete token precision
- Prevents averaging collapse in text while maintaining clean visual representations

**Head 2 (Flow Matching Motor Controller):**
- NitroGen-style DiT architecture with sinusoidal timestep encoding
- AdaLayerNorm for timestep conditioning (scale/shift modulation)
- Cross-attention on Attention Residuals from all backbone layers
- Beta-distributed timestep sampling (α=1.5, β=1.0) during training
- Euler-step ODE solving during inference (distilled to 1-2 steps)
- Triggered by `<act>` token — dormant during text generation

## Installation

Requires Python ≥ 3.11 and PyTorch.

```bash
# Clone repository
git clone https://github.com/your-org/VELA.git
cd VELA

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Version Differences (v7 consolidated)

- `multi_image_collate_fn` reverted to fixed shape `(B, N, C, H, W)` and restored image token padding by region count.
- `encode_images` returns to the `(B, N, L, D)` processing path, `compress_visual_tokens` aggregates along the N dimension, and loss computation reverts to the v7.02 style.
- `num_token_per_image` default restored to 256; script cleanup (kept `diff_stem_delete_common.py`, `rename.sh`, removed redundant data processing scripts).
- Model split into `models/vlm.py` (VLM base) and `models/vla.py` (VLA with action heads).

## PCA Feature Visualization

To visualize the patch features learned by the SigLino vision encoder compared to SigLIP2 and DINOv3, you can use the PCA visualization tool:

```bash
# Run on CPU with a HuggingFace hub model (with optional torchao int8 CPU weight-only quantization)
python VELA-v7/eval/pca_vis.py \
  --hub_repo tiiuae/siglino-30M \
  --config_name dense-30M \
  --device cpu \
  --quantize \
  --input_dir VELA-v7/dummy_data/images/textvqa/train_images \
  --output_path VELA-v7/eval/pca_out \
  --num_samples 5 \
  --max_num_patches 1024

# Run on CUDA (with optional torchao int4 weight quantization)
python VELA-v7/eval/pca_vis.py \
  --hub_repo tiiuae/siglino-0.6B \
  --config_name dense-0.6B \
  --device cuda \
  --quantize \
  --input_dir /path/to/images \
  --output_path /path/to/output \
  --num_samples 10 \
  --max_num_patches 1024
```

## References

- **RWKV-7 "Goose"**: Peng, B., Alcaide, E., et al. "RWKV-7 'Goose' with Expressive Dynamic State Evolution." _arXiv:2503.14456_ (2025).
- **VELA: Exploring RNNs for Visual Language Models**: Hou, H., et al. _arXiv:2406.13362_ (2024).
- **SAM**: Kirillov, A., et al. "Segment Anything." _ICCV 2023_.
- **DINOv2**: Oquab, M., et al. "DINOv2: Learning Robust Visual Features without Supervision." _arXiv:2304.07193_ (2023).
- **SigLIP**: Zhai, X., et al. "Sigmoid Loss for Language Image Pre-Training." _ICCV 2023_.
- **NitroGen**: MineDojo. "NitroGen: Flow Matching for MineDojo Agent Control." _GitHub: MineDojo/NitroGen_ (2024).
- **Flow Matching**: Lipman, Y., et al. "Flow Matching for Generative Modeling." _ICLR 2023_.

## License

Apache 2.0. See [LICENSE](LICENSE) for details.
