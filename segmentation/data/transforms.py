"""
Augmentation transforms for breast cancer mammogram segmentation.

Designed specifically for CBIS-DDSM grayscale mammogram patches.
Augmentations preserve clinical validity:
    - Horizontal flip OK (mediolateral symmetry)
    - Vertical flip NOT used (craniocaudal orientation matters)
    - Small rotations OK (anatomical variation)
    - Intensity jitter OK (acquisition variability)
    - Elastic deformation OK (tissue deformation)

All transforms operate on (image, mask) pairs to maintain spatial alignment.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import random


def get_train_transforms() -> list:
    """Get training augmentation pipeline as a list of callable transforms.

    Returns a list of (image, mask) → (image, mask) functions
    designed for the CBIS-DDSM dataset.
    """
    return [
        random_horizontal_flip,
        random_rotation,
        random_brightness_contrast,
        random_elastic_deformation,
        normalize_image,
        normalize_mask,
        to_tensor,
    ]


def get_val_transforms() -> list:
    """Get validation/test transforms (no augmentation, only normalization)."""
    return [
        normalize_image,
        normalize_mask,
        to_tensor,
    ]


# ── Atomic Transforms ────────────────────────────────────────────────

def random_horizontal_flip(image: np.ndarray, mask: np.ndarray, prob: float = 0.5) -> tuple:
    """Random horizontal flip — safe for mammograms."""
    if random.random() < prob:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    return image, mask


def random_rotation(
    image: np.ndarray,
    mask: np.ndarray,
    degrees: float = 15.0,
) -> tuple:
    """Random rotation within ±degrees. Uses scipy if available, else skips."""
    if random.random() < 0.5:
        angle = random.uniform(-degrees, degrees)
        try:
            from scipy.ndimage import rotate
            image = rotate(image, angle, reshape=False, order=1, mode="reflect")
            mask = rotate(mask, angle, reshape=False, order=0, mode="reflect")
        except ImportError:
            pass  # scipy not available; skip rotation
    return image, mask


def random_brightness_contrast(
    image: np.ndarray,
    mask: np.ndarray,
    brightness_range: tuple[float, float] = (0.85, 1.15),
    contrast_range: tuple[float, float] = (0.85, 1.15),
) -> tuple:
    """Random brightness and contrast adjustment (image only, not mask)."""
    if random.random() < 0.5:
        brightness = random.uniform(*brightness_range)
        contrast = random.uniform(*contrast_range)
        mean = image.mean()
        image = (image - mean) * contrast + mean * brightness
        image = np.clip(image, 0, 255)
    return image, mask


def random_elastic_deformation(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 50.0,
    sigma: float = 5.0,
    prob: float = 0.3,
) -> tuple:
    """Elastic deformation for medical image augmentation.

    Uses Gaussian-filtered random displacement fields. Important for
    capturing natural tissue deformation in mammograms.

    Args:
        alpha: Scaling factor for the deformation.
        sigma: Smoothing factor for the Gaussian filter.
        prob: Probability of applying.
    """
    if random.random() >= prob:
        return image, mask

    try:
        from scipy.ndimage import gaussian_filter, map_coordinates
    except ImportError:
        return image, mask  # scipy not available

    shape = image.shape
    # Create random displacement field
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha

    # Create meshgrid
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    # Apply deformation
    image = map_coordinates(image, indices, order=1, mode="reflect").reshape(shape)
    mask = map_coordinates(mask, indices, order=0, mode="reflect").reshape(shape)

    return image, mask


def normalize_image(image: np.ndarray, mask: np.ndarray) -> tuple:
    """Normalize image from uint8 [0, 255] to float32 [0.0, 1.0]."""
    image = image.astype(np.float32) / 255.0
    return image, mask


def normalize_mask(image: np.ndarray, mask: np.ndarray) -> tuple:
    """Normalize mask from uint8 {0, 255} to float32 {0.0, 1.0}."""
    mask = (mask > 127.5).astype(np.float32)
    return image, mask


def to_tensor(image: np.ndarray, mask: np.ndarray) -> tuple:
    """Convert numpy arrays to PyTorch tensors with channel dimension."""
    # Add channel dim: (H, W) → (1, H, W)
    # .copy() required because np.flip/augmentations produce negative-strided views
    image_tensor = torch.from_numpy(image.copy()).unsqueeze(0).float()
    mask_tensor = torch.from_numpy(mask.copy()).unsqueeze(0).float()
    return image_tensor, mask_tensor


# ── Compose Utility ──────────────────────────────────────────────────

class ComposeTransforms:
    """Compose a list of augmentation functions.

    Usage:
        transforms = ComposeTransforms(get_train_transforms())
        img_tensor, mask_tensor = transforms(image, mask)
    """

    def __init__(self, transforms: list):
        self._transforms = transforms

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> tuple:
        for transform in self._transforms:
            image, mask = transform(image, mask)
        return image, mask

    def __repr__(self) -> str:
        names = [t.__name__ if hasattr(t, "__name__") else type(t).__name__ for t in self._transforms]
        return f"ComposeTransforms({names})"
