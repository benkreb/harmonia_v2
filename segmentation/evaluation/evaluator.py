"""
Evaluation pipeline for ConnectedUnet breast cancer tumor segmentation.

Handles:
    - Test-set evaluation with full metrics
    - Per-sample Dice scores and distribution analysis
    - Best/worst case identification for error analysis
    - Saving predictions as .npy for external review
    - Overlay visualization of predictions on images
"""

from __future__ import annotations

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast
from pathlib import Path
from typing import Any

from segmentation.config import SegmentationConfig
from segmentation.training.metrics import SegmentationMetrics


class Evaluator:
    """Full evaluation engine for segmentation models.

    Runs the model on the test set, computes comprehensive metrics,
    and saves predictions + visualizations.

    Args:
        model: Trained ConnectedUnet.
        config: Full segmentation configuration.
        test_loader: Test DataLoader.
    """

    def __init__(
        self,
        model: nn.Module,
        config: SegmentationConfig,
        test_loader: DataLoader,
    ):
        self._cfg = config
        self._model = model
        self._loader = test_loader

        self._device = torch.device(
            config.training.device
            if config.training.device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model.to(self._device)
        self._model.eval()

        self._output_dir = config.logging.output_dir / "evaluation"
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        save_predictions: bool = True,
        save_visualizations: bool = False,
        max_visualizations: int = 20,
    ) -> dict[str, Any]:
        """Run full evaluation on the test set.

        Args:
            save_predictions: Save predicted masks as .npy.
            save_visualizations: Save overlay images as .png.
            max_visualizations: Maximum number of visualization samples.

        Returns:
            Dict with all evaluation metrics and per-sample data.
        """
        print(f"[Evaluator] Running evaluation on {len(self._loader.dataset)} samples...")

        metrics = SegmentationMetrics()
        per_sample_dice: list[float] = []
        all_predictions: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []
        all_images: list[np.ndarray] = []

        sample_idx = 0
        with torch.no_grad():
            for batch in self._loader:
                images = batch["image"].to(self._device, non_blocking=True)
                masks = batch["mask"].to(self._device, non_blocking=True)

                with autocast("cuda", enabled=self._cfg.training.use_amp):
                    outputs = self._model(images)

                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                metrics.update(outputs, masks)

                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                for i in range(images.size(0)):
                    p = preds[i, 0].cpu().numpy()
                    t = masks[i, 0].cpu().numpy()

                    smooth = 1e-7
                    intersection = (p * t).sum()
                    dice = (2.0 * intersection + smooth) / (p.sum() + t.sum() + smooth)
                    per_sample_dice.append(float(dice))

                    if save_predictions:
                        all_predictions.append(p)
                    if save_visualizations:
                        all_targets.append(t)
                        all_images.append(images[i, 0].cpu().numpy())

                sample_idx += images.size(0)

        computed = metrics.compute()
        per_sample = metrics.compute_per_case()

        # Add per-sample dice to results
        per_sample["dice_per_sample"] = per_sample_dice

        results = {
            "metrics": computed,
            "per_sample": per_sample,
            "n_test_samples": len(self._loader.dataset),
            "best_dice_idx": int(np.argmax(per_sample_dice)) if per_sample_dice else -1,
            "worst_dice_idx": int(np.argmin(per_sample_dice)) if per_sample_dice else -1,
            "best_dice": float(np.max(per_sample_dice)) if per_sample_dice else 0.0,
            "worst_dice": float(np.min(per_sample_dice)) if per_sample_dice else 0.0,
            "dice_std": float(np.std(per_sample_dice)) if per_sample_dice else 0.0,
            "dice_median": float(np.median(per_sample_dice)) if per_sample_dice else 0.0,
        }

        # Save predictions
        if save_predictions and all_predictions:
            preds_array = np.stack(all_predictions)
            np.save(self._output_dir / "test_predictions.npy", preds_array)
            print(f"[Evaluator] Saved {len(all_predictions)} predictions to {self._output_dir}")

        # Save visualizations
        if save_visualizations and all_images:
            self._save_visualizations(
                all_images, all_targets, all_predictions,
                per_sample_dice, max_visualizations,
            )

        # Save results JSON
        results_path = self._output_dir / "evaluation_results.json"
        json_safe = {}
        for k, v in results.items():
            if isinstance(v, (np.integer,)):
                json_safe[k] = int(v)
            elif isinstance(v, (np.floating,)):
                json_safe[k] = float(v)
            elif isinstance(v, dict):
                json_safe[k] = {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv for kk, vv in v.items()}
            elif isinstance(v, list):
                json_safe[k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v]
            else:
                json_safe[k] = v

        with open(results_path, "w") as f:
            json.dump(json_safe, f, indent=2)

        print(f"[Evaluator] Results saved to {results_path}")
        return results

    def _save_visualizations(
        self,
        images: list[np.ndarray],
        targets: list[np.ndarray],
        predictions: list[np.ndarray],
        dice_scores: list[float],
        max_samples: int,
    ):
        """Save overlay visualizations as PNG images."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[Evaluator] matplotlib not available; skipping visualizations.")
            return

        vis_dir = self._output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)

        n_vis = min(max_samples, len(images))

        # Save best and worst cases
        sorted_indices = np.argsort(dice_scores)
        indices_to_vis = list(sorted_indices[:3]) + list(sorted_indices[-3:])
        indices_to_vis = [i for i in indices_to_vis if i < len(images)]

        for i, idx in enumerate(indices_to_vis[:n_vis]):
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))

            axes[0].imshow(images[idx], cmap="gray")
            axes[0].set_title("Input Image")
            axes[0].axis("off")

            axes[1].imshow(targets[idx], cmap="gray")
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")

            axes[2].imshow(predictions[idx], cmap="gray")
            axes[2].set_title(f"Prediction (Dice: {dice_scores[idx]:.3f})")
            axes[2].axis("off")

            # Overlay
            overlay = np.zeros((*images[idx].shape, 3))
            overlay[..., 0] = targets[idx]          # Red: ground truth
            overlay[..., 2] = predictions[idx]       # Blue: prediction
            overlay[..., 1] = targets[idx] * predictions[idx]  # Green: overlap
            axes[3].imshow(images[idx], cmap="gray")
            axes[3].imshow(overlay, alpha=0.5)
            axes[3].set_title("Overlay (R=GT, B=Pred, G=Overlap)")
            axes[3].axis("off")

            plt.tight_layout()
            plt.savefig(vis_dir / f"sample_{idx:04d}.png", dpi=100, bbox_inches="tight")
            plt.close(fig)

        print(f"[Evaluator] Saved {len(indices_to_vis[:n_vis])} visualizations to {vis_dir}")

    def load_checkpoint(self, checkpoint_path: Path | str):
        """Load model weights from a checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self._device, weights_only=False)
        self._model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Evaluator] Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
              f"(Val Dice: {ckpt.get('best_val_dice', 'N/A')})")

    def summarize(self, results: dict) -> str:
        """Human-readable evaluation summary."""
        m = results.get("metrics", {})
        return (
            f"\n{'=' * 60}\n"
            f"EVALUATION RESULTS ({results.get('n_test_samples', 0)} test samples)\n"
            f"{'=' * 60}\n"
            f"  Dice:        {m.get('dice_mean', 0):.4f} ± {results.get('dice_std', 0):.4f} "
            f"(median: {results.get('dice_median', 0):.4f})\n"
            f"  IoU:         {m.get('iou_mean', 0):.4f} ± {m.get('iou_std', 0):.4f}\n"
            f"  Sensitivity: {m.get('sensitivity_mean', 0):.4f} ± {m.get('sensitivity_std', 0):.4f}\n"
            f"  Specificity: {m.get('specificity_mean', 0):.4f} ± {m.get('specificity_std', 0):.4f}\n"
            f"  Precision:   {m.get('precision_mean', 0):.4f} ± {m.get('precision_std', 0):.4f}\n"
            f"  Pixel Acc:   {m.get('pixel_accuracy', 0):.4f}\n"
            f"{'=' * 60}\n"
            f"  Best Dice:   {results.get('best_dice', 0):.4f} (sample {results.get('best_dice_idx', -1)})\n"
            f"  Worst Dice:  {results.get('worst_dice', 0):.4f} (sample {results.get('worst_dice_idx', -1)})\n"
            f"{'=' * 60}"
        )
