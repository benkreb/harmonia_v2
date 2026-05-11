"""Quiet model/loss factories for FL code paths."""

from __future__ import annotations

import torch.nn as nn

from segmentation.config import SegmentationConfig
from segmentation.model import ConnectedUnet
from segmentation.training import CombinedLoss, DeepSupervisionLoss


def build_model_from_config(config: SegmentationConfig) -> nn.Module:
    return ConnectedUnet(
        efficientnet_variant=config.model.efficientnet_variant,
        in_channels=config.model.in_channels,
        num_classes=config.model.num_classes,
        decoder_channels=config.model.decoder_channels,
        use_attention_gate=config.model.use_attention_gate,
        use_deep_supervision=config.model.use_deep_supervision,
        dropout=config.model.dropout,
        pretrained=config.model.pretrained,
    )


def build_loss_from_config(config: SegmentationConfig) -> nn.Module:
    base_loss = CombinedLoss(
        bce_weight=config.loss.bce_weight,
        dice_weight=config.loss.dice_weight,
        focal_weight=config.loss.focal_weight,
        tversky_weight=config.loss.tversky_weight,
        dice_smooth=config.loss.dice_smooth,
        focal_alpha=config.loss.focal_alpha,
        focal_gamma=config.loss.focal_gamma,
        tversky_alpha=config.loss.tversky_alpha,
        tversky_beta=config.loss.tversky_beta,
        bce_pos_weight=config.loss.bce_pos_weight,
    )
    if config.model.use_deep_supervision:
        return DeepSupervisionLoss(base_loss)
    return base_loss

