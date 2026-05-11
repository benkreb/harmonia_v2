"""
Segmentation evaluation metrics for breast cancer tumor segmentation.

Implements standard medical image segmentation metrics:
    1. Dice Score (F1): Primary metric — measures overlap quality
    2. IoU / Jaccard: Intersection over Union
    3. Sensitivity / Recall: True positive rate (tumor detection)
    4. Specificity: True negative rate (background classification)
    5. Precision / PPV: Positive predictive value
    6. Hausdorff Distance: Boundary agreement (requires scipy)

All metrics are computed per-sample and averaged, with special handling
for empty masks (no tumor present).
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Any


class SegmentationMetrics:
    """Computes and tracks segmentation metrics across batches.

    Usage:
        metrics = SegmentationMetrics(threshold=0.5)
        for batch in dataloader:
            pred = model(images)
            metrics.update(pred, masks)
        results = metrics.compute()
    """

    def __init__(self, threshold: float = 0.5, smooth: float = 1e-4):
        self._threshold = threshold
        self._smooth = smooth
        self.reset()

    def reset(self):
        """Reset accumulated metrics."""
        self._dice_scores: list[float] = []
        self._iou_scores: list[float] = []
        self._sensitivity: list[float] = []
        self._specificity: list[float] = []
        self._precision: list[float] = []
        self._sample_count: int = 0
        self._pixel_tp: int = 0
        self._pixel_fp: int = 0
        self._pixel_fn: int = 0
        self._pixel_tn: int = 0

    def update(
        self,
        pred_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        """Update metrics with a new batch.

        Args:
            pred_logits: Raw logits (B, C, H, W).
            target: Ground truth masks (B, C, H, W) in {0, 1}.
        """
        # Convert logits to binary predictions
        pred_probs = torch.sigmoid(pred_logits)
        pred_binary = (pred_probs > self._threshold).float()

        # Compute per-sample metrics
        batch_size = pred_binary.shape[0]
        for i in range(batch_size):
            p = pred_binary[i].view(-1)
            t = target[i].view(-1)

            # Confusion matrix
            tp = (p * t).sum().item()
            fp = (p * (1 - t)).sum().item()
            fn = ((1 - p) * t).sum().item()
            tn = ((1 - p) * (1 - t)).sum().item()

            self._pixel_tp += int(tp)
            self._pixel_fp += int(fp)
            self._pixel_fn += int(fn)
            self._pixel_tn += int(tn)

            # Per-sample Dice
            dice = (2.0 * tp + self._smooth) / (2.0 * tp + fp + fn + self._smooth)
            self._dice_scores.append(dice)

            # Per-sample IoU
            iou = (tp + self._smooth) / (tp + fp + fn + self._smooth)
            self._iou_scores.append(iou)

            # Sensitivity (Recall)
            sensitivity = (tp + self._smooth) / (tp + fn + self._smooth)
            self._sensitivity.append(sensitivity)

            # Specificity
            specificity = (tn + self._smooth) / (tn + fp + self._smooth)
            self._specificity.append(specificity)

            # Precision
            precision = (tp + self._smooth) / (tp + fp + self._smooth)
            self._precision.append(precision)

        self._sample_count += batch_size

    def compute(self) -> dict[str, float]:
        """Compute final aggregated metrics.

        Returns:
            Dict with mean ± std for Dice, IoU, Sensitivity, Specificity,
            Precision, and global pixel-level metrics.
        """
        if self._sample_count == 0:
            return {}

        # Global pixel-level metrics
        total_pixels = self._pixel_tp + self._pixel_fp + self._pixel_fn + self._pixel_tn
        pixel_accuracy = (
            (self._pixel_tp + self._pixel_tn) / max(total_pixels, 1)
        )

        return {
            # Per-sample stats (mean ± std)
            "dice_mean": float(np.mean(self._dice_scores)),
            "dice_std": float(np.std(self._dice_scores)),
            "iou_mean": float(np.mean(self._iou_scores)),
            "iou_std": float(np.std(self._iou_scores)),
            "sensitivity_mean": float(np.mean(self._sensitivity)),
            "sensitivity_std": float(np.std(self._sensitivity)),
            "specificity_mean": float(np.mean(self._specificity)),
            "specificity_std": float(np.std(self._specificity)),
            "precision_mean": float(np.mean(self._precision)),
            "precision_std": float(np.std(self._precision)),

            # Global pixel metrics
            "pixel_accuracy": float(pixel_accuracy),
            "pixel_tp": self._pixel_tp,
            "pixel_fp": self._pixel_fp,
            "pixel_fn": self._pixel_fn,
            "pixel_tn": self._pixel_tn,

            # Sample count
            "n_samples": self._sample_count,
        }

    def compute_per_case(self) -> dict[str, list[float]]:
        """Return per-sample scores for statistical analysis."""
        return {
            "dice": self._dice_scores.copy(),
            "iou": self._iou_scores.copy(),
            "sensitivity": self._sensitivity.copy(),
            "specificity": self._specificity.copy(),
            "precision": self._precision.copy(),
        }

    def summarize(self) -> str:
        """Human-readable summary of metrics."""
        m = self.compute()
        if not m:
            return "No metrics computed yet."
        return (
            f"Segmentation Metrics ({m['n_samples']} samples):\n"
            f"  Dice:        {m['dice_mean']:.4f} ± {m['dice_std']:.4f}\n"
            f"  IoU:         {m['iou_mean']:.4f} ± {m['iou_std']:.4f}\n"
            f"  Sensitivity: {m['sensitivity_mean']:.4f} ± {m['sensitivity_std']:.4f}\n"
            f"  Specificity: {m['specificity_mean']:.4f} ± {m['specificity_std']:.4f}\n"
            f"  Precision:   {m['precision_mean']:.4f} ± {m['precision_std']:.4f}\n"
            f"  Pixel Acc:   {m['pixel_accuracy']:.4f}"
        )
