# VELA: A Visual-Enhanced RWKV

## tl;dr
VELA-MultiImage: Support multiple images in VELA.

## Important Changes:
- Data format is changed. Now it is different from the LLaVA format. So LLaVA can not be directly used in this version.
- Image feature insertion logic is changed. Now it can support any number of images.
- Therefore, this version can support multiple images from single image splits, multi-images and video frames.

## Version Differences (v7 consolidated)
- `multi_image_collate_fn` reverted to fixed shape `(B, N, C, H, W)` and restored image token padding by region count.  
- `encode_images` returns to the `(B, N, L, D)` processing path, `compress_visual_tokens` aggregates along the N dimension, and loss computation reverts to the v7.02 style.  
- `num_token_per_image` default restored to 256; script cleanup (kept `diff_stem_delete_common.py`, `rename.sh`, removed redundant data processing scripts).  
