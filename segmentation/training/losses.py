"""
Segmentation loss functions for breast cancer tumor segmentation.

Implements:
    1. Dice Loss: Directly optimizes the Dice similarity coefficient
    2. BCE-with-Logits Loss: Pixel-wise binary cross-entropy with pos_weight
    3. Focal Loss: Addresses extreme class imbalance (tumors are ~0.4% of pixels)
    4. Tversky Loss: Asymmetric weighting of FP vs FN for small tumor recall
    5. Combined Loss: Weighted sum of Dice + BCE + Focal + Tversky
    6. Boundary Loss: Penalizes boundary errors (for small tumor precision)

The CBIS-DDSM dataset has extremely small tumor masks (~0.4% image area),
making class imbalance a CRITICAL challenge.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice Loss for binary segmentation.

    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    Dice Loss = 1 - Dice

    Args:
        smooth: Laplace smoothing factor. Keep small (1e-7) for sharp gradients.
            Large smooth values (1.0) make the loss landscape flat and hurt
            convergence on tiny targets.
        eps: Numerical stability epsilon.
    """

    def __init__(self, smooth: float = 1e-7, eps: float = 1e-7):
        super().__init__()
        self._smooth = smooth
        self._eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Dice loss.

        Args:
            pred: Predicted probabilities (B, C, H, W) in [0, 1].
            target: Ground truth masks (B, C, H, W) in {0, 1}.

        Returns:
            Scalar loss value (1 - Dice).
        """
        pred = pred.contiguous().view(pred.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)

        dice = (2.0 * intersection + self._smooth) / (union + self._smooth + self._eps)
        loss = 1.0 - dice

        return loss.mean()


class BCELoss(nn.Module):
    """Binary Cross-Entropy Loss with mandatory class weighting.

    For CBIS-DDSM where tumors cover ~0.4% of pixels, pos_weight must be
    set to compensate. A good default is the inverse frequency ratio:

        pos_weight = num_negative_pixels / num_positive_pixels

    With 0.44% tumor coverage: pos_weight ≈ 225. This forces the model to
    pay equal attention to foreground and background during training.

    Args:
        pos_weight: Weight multiplier for the positive (tumor) class.
            Set to ~50-250 for this dataset. Higher → more tumor recall.
    """

    def __init__(self, pos_weight: float = 100.0):
        super().__init__()
        self._pos_weight_val = pos_weight
        self._pw = torch.tensor([pos_weight])
        self._loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=self._pw,
            reduction="mean",
        )

    def forward(
        self,
        pred_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute BCE loss.

        Args:
            pred_logits: Raw logits (B, C, H, W) — sigmoid applied internally.
            target: Ground truth masks (B, C, H, W) in {0, 1}.

        Returns:
            Scalar loss.
        """
        return self._loss_fn(pred_logits, target)

    def to(self, device):
        if self._loss_fn.pos_weight is not None:
            self._loss_fn.pos_weight = self._loss_fn.pos_weight.to(device)
        return self

    @property
    def pos_weight(self) -> float:
        return self._pos_weight_val


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced segmentation.

    From Lin et al. (2017): "Focal Loss for Dense Object Detection."
    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    The modulating factor (1-p_t)^γ reduces loss for well-classified
    examples, focusing training on hard (misclassified) pixels.

    For CBIS-DDSM tumor segmentation, γ=2 and α=0.25 work well as
    the vast majority of pixels are background.

    Args:
        alpha: Weighting factor for the rare class (tumor).
        gamma: Focusing parameter; higher = more focus on hard examples.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self._alpha = alpha
        self._gamma = gamma

    def forward(
        self,
        pred_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Focal loss.

        Args:
            pred_logits: Raw logits (B, C, H, W).
            target: Ground truth masks (B, C, H, W) in {0, 1}.

        Returns:
            Scalar focal loss.
        """
        # BCE with logits for each pixel
        bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")

        # Probability of the target class
        p_t = torch.where(
            target == 1,
            torch.sigmoid(pred_logits),     # p for positive class
            1 - torch.sigmoid(pred_logits),  # p for negative class
        )

        # Alpha weighting
        alpha_t = torch.where(
            target == 1,
            torch.tensor(self._alpha, device=target.device),
            torch.tensor(1 - self._alpha, device=target.device),
        )

        # Focal loss
        focal = alpha_t * (1 - p_t) ** self._gamma * bce

        return focal.mean()


class TverskyLoss(nn.Module):
    """Tversky Loss for small-target segmentation.

    Generalizes Dice by allowing asymmetric weighting of false positives
    and false negatives. For tiny tumor regions, recall (catching tumors)
    is more important than precision (avoiding false alarms).

    Tversky = |X ∩ Y| / (|X ∩ Y| + α·|X \ Y| + β·|Y \ X|)

    Where:
        α = weight on false positives (predicted tumor where none exists)
        β = weight on false negatives (missed tumor)

    For tumor segmentation with 0.4% coverage:
        α = 0.3, β = 0.7  → penalize missed tumors 2.3× more than false alarms

    Args:
        alpha: Weight on false positives. Lower = less penalty for FP.
        beta: Weight on false negatives. Higher = more penalty for FN.
        smooth: Laplace smoothing.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-7):
        super().__init__()
        self._alpha = alpha
        self._beta = beta
        self._smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.contiguous().view(pred.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        intersection = (pred * target).sum(dim=1)
        fp = (pred * (1 - target)).sum(dim=1)       # False positives
        fn = ((1 - pred) * target).sum(dim=1)       # False negatives

        tversky = (intersection + self._smooth) / (
            intersection + self._alpha * fp + self._beta * fn + self._smooth
        )
        return (1 - tversky).mean()


class CombinedLoss(nn.Module):
    """Combined loss for breast cancer tumor segmentation.

    Loss = w_bce·BCE + w_dice·Dice + w_focal·Focal + w_tversky·Tversky

    The combination of these four losses addresses the core problem:
    - BCE (distribution): pixel-level accuracy with class weighting
    - Dice (region): optimizes overlap ratio directly
    - Focal (hard mining): focuses on difficult-to-classify pixels
    - Tversky (recall): asymmetric penalty prioritizing missed tumors over false alarms

    Recommended config for CBIS-DDSM (0.44% tumor coverage):
        Loss = 0.2·BCE(pos_weight=100) + 0.3·Dice + 0.3·Focal + 0.2·Tversky(α=0.3, β=0.7)
    """

    def __init__(
        self,
        bce_weight: float = 0.2,
        dice_weight: float = 0.3,
        focal_weight: float = 0.3,
        tversky_weight: float = 0.2,
        dice_smooth: float = 1e-7,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        bce_pos_weight: float = 100.0,
    ):
        super().__init__()
        self._bce_weight = bce_weight
        self._dice_weight = dice_weight
        self._focal_weight = focal_weight
        self._tversky_weight = tversky_weight

        self._bce = BCELoss(pos_weight=bce_pos_weight)
        self._dice = DiceLoss(smooth=dice_smooth)

        self._focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma) if focal_weight > 0 else None

        self._tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta) if tversky_weight > 0 else None

        self._last_bce = 0.0
        self._last_dice = 0.0
        self._last_focal = 0.0
        self._last_tversky = 0.0

    def forward(
        self,
        pred_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        pred_probs = torch.sigmoid(pred_logits)

        loss = 0.0

        bce = self._bce(pred_logits, target)
        self._last_bce = bce.item()
        loss = loss + self._bce_weight * bce

        dice = self._dice(pred_probs, target)
        self._last_dice = dice.item()
        loss = loss + self._dice_weight * dice

        if self._focal is not None:
            focal = self._focal(pred_logits, target)
            self._last_focal = focal.item()
            loss = loss + self._focal_weight * focal

        if self._tversky is not None:
            tversky = self._tversky(pred_probs, target)
            self._last_tversky = tversky.item()
            loss = loss + self._tversky_weight * tversky

        return loss

    @property
    def last_losses(self) -> dict[str, float]:
        return {
            "bce": self._last_bce,
            "dice": self._last_dice,
            "focal": self._last_focal,
            "tversky": self._last_tversky,
        }

    def to(self, device):
        self._bce = self._bce.to(device)
        return self


class DeepSupervisionLoss(nn.Module):
    """Wrapper for deep supervision: applies the same loss at multiple scales.

    If the model outputs (final_logits, [deep_logits_2, deep_logits_3, ...]),
    this computes the loss at each scale and averages with exponential
    decay weight (deeper scales get lower weight).

    Args:
        base_loss: Base loss function (e.g., CombinedLoss).
        deep_weights: Weights for each deep supervision output.
    """

    def __init__(self, base_loss: nn.Module, deep_weights: list[float] | None = None):
        super().__init__()
        self._base_loss = base_loss
        self._deep_weights = deep_weights or [0.3, 0.2]  # Default for 2 deep heads

    def forward(
        self,
        pred: torch.Tensor | tuple,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute deep supervision loss.

        Args:
            pred: Either single tensor or (final, list_of_deep) tuple.
            target: Ground truth masks.

        Returns:
            Weighted sum of losses at all scales.
        """
        if isinstance(pred, tuple):
            final_logits, deep_logits = pred
        else:
            return self._base_loss(pred, target)

        # Main loss at final output
        loss = self._base_loss(final_logits, target)

        # Deep supervision losses at intermediate scales
        for i, deep_logit in enumerate(deep_logits):
            # Resize target to match deep output
            deep_target = F.interpolate(
                target, size=deep_logit.shape[2:],
                mode="nearest",
            )
            deep_loss = self._base_loss(deep_logit, deep_target)
            weight = self._deep_weights[i] if i < len(self._deep_weights) else 0.1
            loss = loss + weight * deep_loss

        return loss
