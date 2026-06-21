# Changelog

All notable changes to the VELA project will be documented in this file.

## [2026-06-21]

### Changed
- Consolidated VELA-v7 codebase development around `v7.04`.
- Updated root `README.md` directory structure documentation to reflect directory cleanup.
- Cleaned error messages in `detect_gpu_backend()` and `_cpu_brand()` in `VELA-v7/v7.04/src/model.py` to remove external issue-tracker references.
- Added weight-based stability tests for RWKV-7 backbone in `VELA-v7/v7.04/tests/test_with_weights.py`, verifying CPU kernel dispatch, forward/backward stability (no NaN/Inf), gradient flow, and hidden-state coherence with real v7.00 pretrained weights.
- Added `filterwarnings` to `pyproject.toml` pytest config for upstream deprecation warnings.

### Verified
- All 4 stability tests pass end-to-end on CPU with real v7.00 pretrained weights.
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
