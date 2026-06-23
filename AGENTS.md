# Repository Guidelines

## Project Overview
VELA is a visual language model built on the [RWKV-7](https://github.com/BlinkDL/RWKV-LM) recurrent neural network architecture. VELA implements early visual fusion, projecting and injecting visual tokens directly into the RWKV recurrent embedding space at step one rather than relying on standard cross-attention. This allows the model to process sequences with linear time complexity ($O(N)$) and constant memory usage ($O(1)$) during inference.

For the vision encoder, VELA-v7 uses **SigLino** (distilled from DINOv3 and SigLIP2 teachers) with custom compiled kernels, supporting CPU-optimized attention (via compiled SDPA) and GPU-optimized attention (via flex_attention).

## Architecture & Data Flow
1. **Visual Token Input**: Raw images are processed by the `SigLinoImageProcessor` to construct multi-scale patches and sequence parameters (shapes, locations).
2. **Vision Encoding (SigLino)**: Visual features are extracted by a locally-vendored `SigLino` model.
   - **Attention Dispatch**: Device-aware routing in `Attention.forward` dispatches CUDA tensors to Triton-based `flex_attention` (with sink attention auxiliary scaling) and CPU tensors to a pre-compiled `scaled_dot_product_attention` loop.
3. **Visual Fusion & Projection**: Vision features are projected into VELA's embedding dimension via linear projection layer mappings and injected in-place inside the sequence (wrapped within `<img_start>` and `<img_end>` tokens) before passing to the language decoder, enabling dynamic early fusion of document images and text.
4. **RWKV Recurrent Stack**: The unified sequence of visual and language tokens passes through RWKV-7 block layers.
   - **Attention Residuals (Block AttnRes)**: The main residual path in RWKV-7 blocks. It replaces typical additive residual links with learned cross-layer attention mechanisms to combat hidden-state dilution.
   - **MHC MoE Layers (Layers 0-3)**: The first 4 layers of the recurrent stack are configured as Dense MoE blocks with 4 experts. Routing weights are computed from WKV head outputs (pre-output projection) using RMSNorm, linear projections, and a 20-iteration Sinkhorn-Knopp algorithm, following the Manifold-Constrained Hyper-Connections (mHC) formulation.
   - **WindBackstepping CUDA Kernel**: An optimized custom kernel performing RWKV-7 linear recurrence on GPU.
5. **VLA Action Heads (v7.10+)**: Two parallel heads consume the Attention Residual stream for action prediction:
   - **Head 1 (JEPA World Model)**: InfoNCE contrastive loss on all tokens + CE on non-image tokens. Predicts next embeddings in the residual stream.
   - **Head 2 (Flow Matching Motor Controller)**: NitroGen-style DiT with sinusoidal timestep encoding, AdaLayerNorm, and cross-attention on the Attention Residual stream. Predicts 16-step action chunks via ODE solving.

## Key Directories
* `VELA-v7/src/`: Core Python packages and files for the VELA model.
  - `models/`: Model definitions
    - `vlm.py`: VLM base model (RWKV, Block, MHCBlock, VLM LightningModule)
    - `vla.py`: VLA model (inherits VLM, adds FlowMatchingHead + InfoNCE loss)
  - `model.py`: Facade module — re-exports `VLM`, `VLA` and all RWKV internals for backward compatibility
  - `dataset.py`: Multi-modal conversation datasets, tokenizers, padding, and masking.
  - `trainer.py`: Training loop callbacks (LR schedule, logging, checkpointing).
  - `siglino/`: Locally-vendored Falcon Vision (SigLino) implementation.
  - `siglino/kernels/`: Platform-specific attention kernels (CPU compiled SDPA, CUDA Triton flex_attention).
* `VELA-v7/tests/`: Integration, stability, and correctness tests.
  - `test_vla.py`: Unit tests for VLA components (InfoNCE loss, FlowMatchingHead, SinusoidalTimestepEncoder).
* `VELA-v7/eval/`: Model evaluations and feature mapping.
  - `pca_vis.py`: High-resolution PCA visualization script plotting SigLino, SigLIP, and DINOv3 features side-by-side.
* `app/`: Web UI applications for local testing.

## Development Commands
All packaging, dependencies, and environments are managed via `uv`.

* **Dependency Installation**:
  ```bash
  uv sync
  ```
* **Run Test Suite**:
  ```bash
  pytest
  ```
* **Run PCA Visualization (CPU)**:
  ```bash
  python VELA-v7/eval/pca_vis.py \
    --hub_repo tiiuae/siglino-30M \
    --config_name dense-30M \
    --device cpu \
    --quantize \
    --input_dir VELA-v7/dummy_data/images/textvqa/train_images \
    --output_path VELA-v7/eval/pca_out \
    --num_samples 5
  ```
* **Run PCA Visualization (CUDA with int4 Weight-Only Quantization)**:
  ```bash
  python VELA-v7/eval/pca_vis.py \
    --hub_repo tiiuae/siglino-0.6B \
    --config_name dense-0.6B \
    --device cuda \
    --quantize \
    --input_dir /path/to/images \
    --output_path /path/to/output
  ```

## VLA-Specific Architecture Notes

### Head 1 — JEPA World Model (`VLA._info_nce_loss`, `VLA._ce_loss`)
- **InfoNCE** (weight=1.0): Contrastive loss on ALL tokens. Predicts next continuous embedding in the residual stream. Shapes latent space for smooth physics-aware representations.
- **CE** (weight=0.1): Cross-entropy on non-image tokens only. Masked on `IMAGE_TOKEN_INDEX`. Guarantees discrete token precision in text/actions.
- Both losses computed from the Attention Residual output (`V_blocks` + `partial_block`).

### Head 2 — Flow Matching Motor Controller (`FlowMatchingHead`)
- **Architecture**: SinusoidalTimestepEncoder → TimestepMLP → 4× DiTBlock (self-attn + cross-attn on residual + FFN) → AdaLayerNorm output modulation → velocity projection.
- **Timestep Sampling**: Beta(1.5, 1.0) → `t = (1 - t) * 0.999` during training.
- **Inference**: Euler-step ODE solving (distilled to 2 steps) via `FlowMatchingHead.sample()`.
- **Attention Residual Conditioning**: All backbone layers contribute via `block_attn_res`. The last `action_horizon` tokens of the aggregated representation are used as cross-attention keys/values.
- **Trigger**: The `<act>` token — the head is dormant during text generation.

### Training
- `VLA.training_step` overrides `VLM.training_step` — runs full RWKV forward pass with V_blocks capture, computes all three losses, logs each component.
- VLA.forward delegates to VLM.forward for backward compatibility.

## Code Conventions & Common Patterns
1. **Device Dispatch**: Dual-path routing is used to optimize execution on CPU and GPU. Check `xq.is_cuda` at runtime rather than static configurations to prevent Inductor from tracing `flex_attention` on CPU environments.
2. **Quantization with torchao**:
   - On CUDA: Apply `Int4WeightOnlyConfig` weight quantization to the vision tower.
   - On CPU: Fallback to `Int8WeightOnlyConfig` weight quantization, as int4 quantization is GPU-exclusive.
3. **Avoid Custom Reinventions**: Use native PyTorch equivalents over scratch implementations where possible (e.g., `nn.GELU(approximate="tanh")` and `torch.repeat_interleave`).
4. **Compiled CPU Blocks**: When compiling for CPU, target the isolated attention kernel via `torch.compile(mode="reduce-overhead")` instead of full-module compilations.
5. **No `trust_remote_code`**: Load model weights using Hub safetensors directly into local class files via `load_siglino_from_hub()` to ensure code is clean, local, and auditable.
6. **Module Architecture**: New model variants go in `models/` as separate files (e.g., `vla.py` inherits from `vlm.py`). The root `model.py` is a facade for backward compatibility.

## Important Files
* `VELA-v7/src/models/vlm.py`: VLM base model — RWKV backbone, Block, MHCBlock, training loop.
* `VELA-v7/src/models/vla.py`: VLA model — InfoNCE loss, FlowMatchingHead, VLA training_step.
* `VELA-v7/src/model.py`: Facade — re-exports `VLM`, `VLA`, and all RWKV internals.
* `VELA-v7/src/dataset.py`: Multi-modal dataset, ChatML formatting, image token processing.
* `VELA-v7/src/siglino/model.py`: Standard definitions of Falcons/SigLino transformer architecture.
* `VELA-v7/src/siglino/attention.py`: Attention routing coordinator.
* `pyproject.toml`: Project build requirements, Pytest path configurations, and direct package dependencies.

## Runtime/Tooling Preferences
* **Runtime**: Python 3.13 (declared in `.python-version`, target range `[3.11, 3.14)`).
* **Package Manager**: `uv` (strict lockfile maintenance).
* **GPU Hardware Acceleration**: Custom Triton kernels (`flex_attention`) and C++ autograd functions (`WindBackstepping`) compile dynamically at runtime.

## Testing & QA
* **Framework**: `pytest`.
* **Testing Pattern**: Runs actual regression tests loading VisualRWKV weights. Checks correctness of forward/backward operations, asserts that intermediate representations have no NaNs or Infs, verifies customized kernel builds, and tests CPU quantizations under `test_with_weights.py`. Also verifies ChatML formatting, in-place token wrapping, Sinkhorn-Knopp convergence, and bfloat16 gradient flow stability under `test_moe_early_fusion.py`.
* **VLA Tests**: `test_vla.py` tests InfoNCE loss correctness, FlowMatchingHead shape/gradient flow, SinusoidalTimestepEncoder output, and special token integration.
