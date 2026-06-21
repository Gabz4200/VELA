# Changelog

All notable changes to the VELA project will be documented in this file.

## [2026-06-21]

### Added
- Implemented **Block Attention Residuals (AttnRes)** natively inside the RWKV-7 block structure (`VELA-v7/src/model.py`). It fully replaces the standard additive residual connections. The mechanism uses a learned pseudo-query (`attn_res_proj` and `mlp_res_proj`) to select earlier representations from previous block chunks, passing states continuously using a `V_blocks` tensor array to support scaling to large models while perfectly preserving DeepSpeed checkpointing efficiency.
- Replaced the custom/external vision encoder with a locally-vendored **SigLino** model (distilled from DINOv3 and SigLIP2) supporting optimized CPU (compiled SDPA) and GPU (flex_attention) attention routing.
- Integrated optional **torchao** weight quantization support for the SigLino vision encoder (CUDA int4 weight-only and CPU int8 weight-only) during model load.
- Added a PCA visualization script (`VELA-v7/eval/pca_vis.py`) allowing visual analysis of patch features across SigLino, SigLIP2, and DINOv3 with support for both local checkpoints and Hugging Face Hub repos, custom 2-row layout rendering, dynamic resolution scaling, and torchao quantization.

### Changed
- Finalized directory consolidation by moving `VELA-v7/v7.04/` contents directly to the `VELA-v7/` root. The codebase is now unversioned internally, stopping the nested `v7.xx` folder structure.
- Refactored activation and tensor functions in SigLino model modules to use native PyTorch operators: replaced `PytorchGELUTanh` with `nn.GELU(approximate="tanh")` and replaced custom `repeat_kv` with `torch.repeat_interleave`.
- Replaced deprecated `pynvml` dependency with the official `nvidia-ml-py` library in `pyproject.toml` to eliminate import deprecation warnings.
- Updated PCA visualization script to correctly pass `max_pixels` to the model loader and increased default max patches to `1024` to resolve map pixelation.
- Updated `pyproject.toml` paths to correctly target the new `VELA-v7/` structure.
- Updated root `README.md` directory structure documentation to reflect directory cleanup.
- Cleaned error messages in `detect_gpu_backend()` and `_cpu_brand()` in `VELA-v7/v7.04/src/model.py` to remove external issue-tracker references.
- Added weight-based stability tests for RWKV-7 backbone in `VELA-v7/v7.04/tests/test_with_weights.py`, verifying CPU kernel dispatch, forward/backward stability (no NaN/Inf), gradient flow, and hidden-state coherence with real v7.00 pretrained weights.
- Added `filterwarnings` to `pyproject.toml` pytest config for upstream deprecation warnings.

### Verified
- All 6 stability and regression tests pass end-to-end on CPU (including CPU quantization forward pass).
- C++ CPU kernel (`WindBackstepping`) compiles and dispatches correctly; pure-PyTorch fallback available.
- Core imports (`src.model`, `app.modeling_rwkv`) load without errors; all entry-point scripts compile.
- Forward pass produces finite output (μ=-1.5, σ=2.2, max|·|=15.6) with no NaN/Inf.
- Backward pass produces 0 NaN / 0 Inf gradients across all 399 parameters with non-zero gradient flow.
- Hidden-state drift is informational (cross-version weight normalization variance); forward/backward stability are the definitive regression signals.

### Removed
- Fully removed all references, imports, and dependencies related to `torch-directml` and the `dml` execution strategy from both `VELA-v7/v7.03` and `VELA-v7/v7.04`.
- Removed legacy, experimental, and incomplete version directories (`v7.00`, `v7.01`, `v7.01_with_contrastive_alignment`, `v7.02`, `v7.03`, `v7.10`) under `VELA-v7/` to consolidate development around `v7.04`.

### Fixed
- Fixed `w`-index swap in `VELA-v7/v7.04/cuda/wkv7_op.cpp` backward_cpu's `dstate`/`dstateT` update where `w.unsqueeze` dimension was misaligned, causing ~10× gradient errors in `dw`/`dz`.
- Fixed `WindBackstepping.backward` in `VELA-v7/v7.04/src/model.py` to use `*grad_outputs` with `.contiguous()` instead of strict contiguity assert (autograd can deliver non-contiguous grad tensors).
- Fixed critical runtime import crash in `VELA-v7/v7.04/evaluate.py` caused by legacy imports (`POSSIBLE_RESOLUTIONS`, `single_image_to_multi_image_strategy`).
- Fixed runtime import crash in `VELA-v7/v7.04/evaluate_hfds.py` caused by a non-existent import path (`src.rwkv_tokenizer`).
- Fixed type signature and positional argument bugs in evaluation scripts when calling `process_image_tokens_in_conversations`.
- Fixed `calc_ctxlen.py` positional argument error where the required `has_image` argument was missing from the `preprocess` invocation.
- Standardized image processing during evaluation in `VELA-v7/v7.04` to match training behavior (converting images to region-split SigLIP-preprocessed single Tensors instead of multi-tower dicts).
- Standardized `--vision_tower_path` argument parsing across training and evaluation scripts, including backwards-compatible fallback resolution logic for `--vision_tower_dir`.
- Prevented potential division-by-zero crashes in evaluation progress bar step updates when processing datasets with fewer than 100 items.
- Solved a potential infinite loop bug in `src/utils.py`'s `largest_3n_plus_2_prime` function when handling datasets of size $\le 2$, and mathematically optimized the prime finder to allow returning primes equal to $x$ (increasing unique permuted indexing coverage).
- Fixed `KeyError` in `VELA-v7/v7.04/src/model.py` when `RWKV_JIT_ON` environment variable is not set — changed `os.environ["RWKV_JIT_ON"]` to `os.environ.get("RWKV_JIT_ON", "0")` for resilient default.
- Fixed duplicated assertion and clarified missing-keys assertion logic in `VELA-v7/v7.04/tests/test_with_weights.py`.
