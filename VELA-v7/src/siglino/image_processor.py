# Image preprocessing for SigLino
# Handles resizing, normalisation, and patchification for CPU and CUDA inference.

import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMAGE_MEAN = [0.5, 0.5, 0.5]
IMAGE_STD = [0.5, 0.5, 0.5]

# Pillow ≥10 moved resampling filters to Image.Resampling.*
_BICUBIC = getattr(Image, "Resampling", Image).BICUBIC


def smart_resize(
    height: int,
    width: int,
    factor: int = 16,
    min_pixels: int = 128 * 128,
    max_pixels: int = 256 * 256,
) -> tuple[int, int]:
    """Resize dimensions to be divisible by factor while respecting pixel bounds."""
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute aspect ratio must be smaller than 200")

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    if h_bar * w_bar > max_pixels:
        beta = np.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = np.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


def convert_image_to_patches(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert image tensor (H, W, C) → patches (num_patches, patch_size² × C)."""
    image_height, image_width, num_channels = image.shape
    ph = image_height // patch_size
    pw = image_width // patch_size
    patched = image.reshape(ph, patch_size, pw, patch_size, num_channels)
    patched = patched.permute(0, 2, 1, 3, 4)
    return patched.reshape(ph * pw, -1)


def pad_along_first_dim(
    array: torch.Tensor,
    target_length: int,
    pad_value: float = 0.0,
    mask_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad tensor along dim 0 and return (padded_tensor, valid_mask)."""
    current_length = array.shape[0]
    padding_length = target_length - current_length
    mask = torch.ones(target_length, dtype=mask_dtype, device=array.device)

    if padding_length > 0:
        array = F.pad(array, (0, 0, 0, padding_length), value=pad_value)
        mask[-padding_length:] = 0

    return array, mask


class SigLinoImageProcessor:
    """Preprocesses PIL images into patchified tensors for SigLino.

    Output dict keys (always ``spatial_shapes``, never ``spatial_shape``):
        pixel_values  : (N, L, patch_size² × C)  — patchified & normalised pixels
        padding_mask  : (N, L)                    — 1 = valid patch, 0 = padding
        spatial_shapes: (N, 2)                    — (n_patches_h, n_patches_w) per image
    """

    def __init__(
        self,
        patch_size: int = 16,
        min_pixels: int = 128 * 128,
        max_pixels: int = 256 * 256,
        image_mean: list[float] | None = None,
        image_std: list[float] | None = None,
        do_resize: bool = True,
        do_rescale: bool = True,
        do_normalize: bool = True,
    ):
        self.patch_size = patch_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.image_mean = image_mean or IMAGE_MEAN
        self.image_std = image_std or IMAGE_STD
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize

    def preprocess_single(self, image: "Image.Image | np.ndarray") -> tuple[np.ndarray, tuple[int, int]]:
        """Preprocess one image → (HWC float32 array, (n_patches_h, n_patches_w))."""
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))

        # Ensure HWC
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[0] == 3:  # CHW → HWC
            image = np.transpose(image, (1, 2, 0))

        height, width = image.shape[:2]

        if self.do_resize:
            resized_height, resized_width = smart_resize(
                height, width,
                factor=self.patch_size,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            pil_image = Image.fromarray(image.astype(np.uint8))
            pil_image = pil_image.resize((resized_width, resized_height), _BICUBIC)
            image = np.array(pil_image)
        else:
            resized_height, resized_width = height, width

        if self.do_rescale:
            image = image.astype(np.float32) / 255.0

        if self.do_normalize:
            mean = np.array(self.image_mean, dtype=np.float32)
            std = np.array(self.image_std, dtype=np.float32)
            image = (image - mean) / std

        spatial_shape = (resized_height // self.patch_size, resized_width // self.patch_size)
        return image, spatial_shape

    def preprocess(
        self,
        images: "list[Image.Image | np.ndarray]",
    ) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
        pixel_values, spatial_shapes = [], []
        for img in images:
            pv, ss = self.preprocess_single(img)
            pixel_values.append(pv)
            spatial_shapes.append(ss)
        return pixel_values, spatial_shapes

    def batch_images_with_mask(
        self,
        pixel_values: list[np.ndarray],
        spatial_shapes: list[tuple[int, int]],
        max_num_patches: int = 256,
        pad: bool = True,
        output_dtype: torch.dtype = torch.float32,
        mask_dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        """Batch preprocessed arrays into padded tensors with validity masks."""
        if mask_dtype is None:
            mask_dtype = output_dtype

        batched_pixels, batched_masks, batched_shapes = [], [], []

        for img, shape in zip(pixel_values, spatial_shapes):
            patches = convert_image_to_patches(
                torch.from_numpy(img).to(dtype=output_dtype), self.patch_size
            )
            if pad:
                patches, mask = pad_along_first_dim(patches, max_num_patches, mask_dtype=mask_dtype)
            else:
                mask = torch.ones(patches.shape[0], dtype=mask_dtype)

            batched_pixels.append(patches)
            batched_masks.append(mask)
            batched_shapes.append(list(shape))

        return {
            "pixel_values": torch.stack(batched_pixels),
            "padding_mask": torch.stack(batched_masks),
            "spatial_shapes": torch.tensor(batched_shapes, dtype=torch.long),
        }

    def __call__(
        self,
        images: "list[Image.Image] | Image.Image",
        max_num_patches: int = 256,
        pad: bool = True,
        output_dtype: torch.dtype = torch.float32,
        mask_dtype: torch.dtype | None = None,
        # kept for API compat with upstream siglino — not used internally
        n_storage_tokens: int = 4,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        """Process one or more images and return a batched tensor dict."""
        if isinstance(images, Image.Image):
            images = [images]
        pixel_values, spatial_shapes = self.preprocess(images)
        return self.batch_images_with_mask(
            pixel_values, spatial_shapes,
            max_num_patches=max_num_patches,
            pad=pad,
            output_dtype=output_dtype,
            mask_dtype=mask_dtype,
        )
