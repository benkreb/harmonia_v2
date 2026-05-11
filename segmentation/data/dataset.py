"""
PyTorch Dataset for CBIS-DDSM preprocessed mammogram data.

Reads from .npy files produced by the preprocessing pipeline (steps 1-10),
applies on-the-fly augmentation, and returns (image, mask) tensor pairs.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Literal, Callable

from .transforms import ComposeTransforms, get_train_transforms, get_val_transforms


class CBISDDSMDataset(Dataset):
    """CBIS-DDSM Breast Cancer Mammogram Segmentation Dataset.

    Loads preprocessed .npy files containing 256×256 grayscale mammogram
    patches and corresponding binary tumor masks.

    Args:
        data_dir: Directory containing X_{split}.npy and masks_{split}.npy.
        split: Dataset split ("train", "val", "test").
        transform: Optional ComposeTransforms or callable for augmentation.
            If None, uses default transforms based on split.
        use_class_weights: If True, also returns the classification label (benign/malignant).
        return_index: If True, also returns the sample index (for debugging).
    """

    def __init__(
        self,
        data_dir: Path | str,
        split: Literal["train", "val", "test"] = "train",
        transform: ComposeTransforms | Callable | None = None,
        use_class_weights: bool = False,
        return_index: bool = False,
    ):
        data_dir = Path(data_dir)

        self._split = split
        self._use_class_weights = use_class_weights
        self._return_index = return_index

        # Load images and masks
        x_path = data_dir / f"X_{split}.npy"
        mask_path = data_dir / f"masks_{split}.npy"

        if not x_path.exists():
            raise FileNotFoundError(
                f"X_{split}.npy not found at {x_path}. "
                f"Run the preprocessing pipeline (steps 1-10) first."
            )
        if not mask_path.exists():
            raise FileNotFoundError(
                f"masks_{split}.npy not found at {mask_path}."
            )

        # Memory-map for efficiency with large datasets
        self._images = np.load(x_path, mmap_mode="r")
        self._masks = np.load(mask_path, mmap_mode="r")

        # Optionally load labels
        y_path = data_dir / f"y_{split}.npy"
        self._labels = np.load(y_path) if y_path.exists() else None

        # Set transforms
        if transform is None:
            transform = (
                ComposeTransforms(get_train_transforms()) if split == "train"
                else ComposeTransforms(get_val_transforms())
            )
        self._transform = transform

        # Dataset statistics
        self._n_samples = len(self._images)
        self._image_shape = self._images.shape[1:] if self._n_samples > 0 else (256, 256, 1)

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single sample.

        Returns:
            Dict with keys:
                'image': Float tensor (1, H, W) normalized to [0, 1]
                'mask': Float tensor (1, H, W) with values {0.0, 1.0}
                'label': (optional) Long tensor with 0=benign, 1=malignant
                'index': (optional) Sample index as tensor
        """
        # Load from numpy (H, W, 1) → squeeze to (H, W)
        image = self._images[idx].squeeze(-1).astype(np.float32)
        mask = self._masks[idx].squeeze(-1).astype(np.float32)

        # Apply transforms
        if self._transform is not None:
            image, mask = self._transform(image, mask)

        # Build output
        output = {
            "image": image,
            "mask": mask,
        }

        if self._use_class_weights and self._labels is not None:
            output["label"] = torch.tensor(self._labels[idx], dtype=torch.long)

        if self._return_index:
            output["index"] = torch.tensor(idx, dtype=torch.long)

        return output

    def get_raw(self, idx: int) -> tuple:
        """Get raw (untransformed) image and mask for visualization."""
        image = self._images[idx].squeeze(-1).copy()
        mask = self._masks[idx].squeeze(-1).copy()
        return image, mask

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def image_shape(self) -> tuple:
        return self._image_shape

    @property
    def split(self) -> str:
        return self._split

    def get_class_distribution(self) -> dict:
        """Get class distribution if labels are available."""
        if self._labels is not None:
            unique, counts = np.unique(self._labels, return_counts=True)
            return {int(k): int(v) for k, v in zip(unique, counts)}
        return {}

    def get_mask_coverage_stats(self) -> dict:
        """Get mask foreground coverage statistics."""
        total_pixels = self._masks.shape[1] * self._masks.shape[2]
        coverages = self._masks.sum(axis=(1, 2, 3)) / (255.0 * total_pixels)
        return {
            "mean_coverage": float(coverages.mean()),
            "median_coverage": float(np.median(coverages)),
            "min_coverage": float(coverages.min()),
            "max_coverage": float(coverages.max()),
            "n_empty_masks": int((coverages == 0).sum()),
        }

    def __repr__(self) -> str:
        return (
            f"CBISDDSMDataset(split={self._split}, "
            f"samples={self._n_samples}, "
            f"shape={self._image_shape}, "
            f"transform={'yes' if self._transform else 'no'})"
        )
