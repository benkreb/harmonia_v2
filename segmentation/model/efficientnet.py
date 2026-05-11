"""
EfficientNet Implementation for Segmentation Encoder.

Implements the core building blocks from the EfficientNet paper
(Tan & Le, 2019) — MBConv with Squeeze-and-Excitation, Swish activation,
and compound scaling. Used as the encoder backbone for ConnectedUnet.

Key architectural elements:
    1. MBConv: Mobile Inverted Bottleneck Convolution
    2. SE: Squeeze-and-Excitation optimization
    3. Compound scaling: width × depth × resolution
    4. Swish activation (SiLU in PyTorch)

Reference: arXiv:1905.11946v5
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


# ── Building Blocks ──────────────────────────────────────────────────

class Swish(nn.Module):
    """Swish activation function: f(x) = x * sigmoid(x).

    From the paper: "Swish consistently outperforms ReLU on deeper models."
    PyTorch's SiLU is equivalent; this is an explicit implementation.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel-wise feature recalibration.

    Args:
        in_channels: Number of input channels.
        reduced_channels: Number of channels in the squeeze (bottleneck) layer.
            Default: in_channels // 4 (following paper Section 4).
    """

    def __init__(self, in_channels: int, reduced_channels: int | None = None):
        super().__init__()
        if reduced_channels is None:
            reduced_channels = max(1, in_channels // 4)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            Swish(),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution (MBConv) block.

    This is the core building block of EfficientNet. It consists of:
        1. 1x1 expansion convolution (pointwise)
        2. Depthwise convolution (k×k)
        3. Squeeze-and-Excitation
        4. 1x1 projection convolution (pointwise)
        5. Residual connection (if stride=1 and channels match)

    Architecture from EfficientNet paper Figure 2 and Section 3.

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        kernel_size: Depthwise convolution kernel size.
        stride: Stride for the depthwise convolution.
        expand_ratio: Channel expansion factor (typically 1 or 6).
        se_ratio: Squeeze-and-Excitation reduction ratio.
        drop_rate: Dropout rate for stochastic depth (DropPath).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: int,
        se_ratio: float = 0.25,
        drop_rate: float = 0.0,
    ):
        super().__init__()

        self._has_skip = (stride == 1) and (in_channels == out_channels)
        self._drop_rate = drop_rate

        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        self._expand_conv = (
            nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False)
            if expand_ratio != 1
            else nn.Identity()
        )
        self._bn0 = nn.BatchNorm2d(expanded_channels) if expand_ratio != 1 else nn.Identity()

        # Depthwise convolution
        self._depthwise_conv = nn.Conv2d(
            expanded_channels,
            expanded_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=expanded_channels,  # Depthwise: one filter per channel
            bias=False,
        )
        self._bn1 = nn.BatchNorm2d(expanded_channels)

        # Squeeze-and-Excitation
        se_channels = max(1, int(in_channels * se_ratio))
        self._se = SqueezeExcitation(expanded_channels, se_channels)

        # Projection (output) phase
        self._project_conv = nn.Conv2d(
            expanded_channels, out_channels, kernel_size=1, bias=False
        )
        self._bn2 = nn.BatchNorm2d(out_channels)

        # Activation
        self._swish = Swish()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Expansion
        x = self._expand_conv(inputs)
        x = self._bn0(x)
        x = self._swish(x)

        # Depthwise
        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze-and-Excitation
        x = self._se(x)

        # Projection
        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection with stochastic depth
        if self._has_skip:
            if self._drop_rate > 0.0 and self.training:
                x = _drop_path(x, self._drop_rate)
            x = x + inputs

        return x


def _drop_path(x: torch.Tensor, drop_rate: float) -> torch.Tensor:
    """Stochastic depth (DropPath) for regularization.

    Randomly drops entire residual blocks during training.
    Used in deeper EfficientNet variants.
    """
    if drop_rate == 0.0:
        return x
    keep_prob = 1.0 - drop_rate
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    return x.div(keep_prob) * random_tensor


# ── EfficientNet Encoder ─────────────────────────────────────────────

class EfficientNet(nn.Module):
    """EfficientNet encoder for segmentation backbone.

    Implements the EfficientNet architecture as described in
    Tan & Le (2019), "EfficientNet: Rethinking Model Scaling for
    Convolutional Neural Networks."

    The encoder outputs skip connection features at multiple scales
    for the ConnectedUnet decoder.

    Compound Scaling Formula (Equation 3):
        depth:    d = α^φ
        width:    w = β^φ
        resolution: r = γ^φ
        where α · β² · γ² ≈ 2

    Args:
        variant: EfficientNet scale ("b0" through "b4").
        in_channels: Input channels (1 for grayscale mammograms).
        override_width: Manual width multiplier override.
    """

    # Stage → output stride table for skip connection extraction
    # These are the stages whose outputs feed into the U-Net decoder
    _SKIP_STAGES = [1, 2, 4, 6, 7]  # MBConv stages (1-indexed for paper convention)

    def __init__(
        self,
        variant: Literal["b0", "b1", "b2", "b3", "b4"] = "b0",
        in_channels: int = 1,
        override_width: float | None = None,
    ):
        super().__init__()

        self.variant = variant
        self._in_channels = in_channels

        # Load variant config
        from segmentation.config import EFFICIENTNET_CONFIGS
        cfg = EFFICIENTNET_CONFIGS[variant]

        self._width_coef = override_width if override_width is not None else cfg["width_coefficient"]
        self._depth_coef = cfg["depth_coefficient"]
        self._dropout_rate = cfg["dropout_rate"]
        self._stem_channels = self._round_filters(cfg["stem_channels"])
        self._block_config = cfg["block_config"]

        # ── Stem ──
        # 3×3 conv with stride 2, same as paper
        self._stem = nn.Sequential(
            nn.Conv2d(
                in_channels, self._stem_channels,
                kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(self._stem_channels),
            Swish(),
        )

        # ── MBConv Stages ──
        self._blocks = nn.ModuleList()
        self._stage_indices: list[int] = []  # Track which block index starts each stage
        self._stage_output_channels: list[int] = []  # Output channels per stage

        block_idx = 0
        for stage_idx, (
            expand_ratio, kernel_size, stride, out_channels, num_layers, se_ratio
        ) in enumerate(self._block_config):
            out_channels = self._round_filters(out_channels)
            num_layers = self._round_repeats(num_layers)

            self._stage_indices.append(block_idx)

            for layer_idx in range(num_layers):
                # Only first layer in stage uses the specified stride
                layer_stride = stride if layer_idx == 0 else 1
                layer_in = (
                    self._stem_channels if block_idx == 0
                    else self._round_filters(self._block_config[stage_idx - 1][3])
                    if layer_idx == 0
                    else out_channels
                )

                # Fix: track correct in-channels for this block
                if block_idx > 0:
                    prev_block = self._blocks[-1]
                    if hasattr(prev_block, '_project_conv'):
                        layer_in = prev_block._project_conv.out_channels
                    elif hasattr(prev_block, 'out_channels'):
                        layer_in = prev_block.out_channels

                # Compute stochastic depth drop rate
                total_blocks = sum(self._round_repeats(n) for _, _, _, _, n, _ in self._block_config)
                drop_rate = self._dropout_rate * block_idx / max(1, total_blocks - 1)

                self._blocks.append(
                    MBConvBlock(
                        in_channels=layer_in,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=layer_stride,
                        expand_ratio=expand_ratio,
                        se_ratio=se_ratio,
                        drop_rate=drop_rate,
                    )
                )
                block_idx += 1

            self._stage_output_channels.append(out_channels)

        # ── Head (final 1×1 for classification — NOT used in segmentation) ──
        head_in = self._round_filters(self._block_config[-1][3])
        head_out = self._round_filters(1280)
        self._head = nn.Sequential(
            nn.Conv2d(head_in, head_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(head_out),
            Swish(),
        )

    def _round_filters(self, filters: int) -> int:
        """Round number of filters based on width coefficient.

        Following paper Section 4: round to multiples of the divisor.
        """
        divisor = 8
        filters *= self._width_coef
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        # Ensure at least 90% of target
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return new_filters

    def _round_repeats(self, repeats: int) -> int:
        """Round number of layer repeats based on depth coefficient."""
        return int(math.ceil(repeats * self._depth_coef))

    def forward(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        """Forward pass returning skip connection feature maps.

        Returns:
            Dict mapping stage index → feature tensor for U-Net skip connections.
            Includes the final head output as stage 'head'.
        """
        skip_features: dict[int, torch.Tensor] = {}

        # Stem
        x = self._stem(x)
        skip_features[0] = x  # 128×128

        # MBConv stages
        for stage_idx, start_idx in enumerate(self._stage_indices):
            end_idx = (
                self._stage_indices[stage_idx + 1]
                if stage_idx + 1 < len(self._stage_indices)
                else len(self._blocks)
            )
            for block_idx in range(start_idx, end_idx):
                x = self._blocks[block_idx](x)
            # Save outputs at skip stages (1-indexed stages: 1,2,4,6,7)
            actual_stage = stage_idx + 1  # 1-indexed like paper
            if actual_stage in self._SKIP_STAGES:
                skip_features[actual_stage] = x
            skip_features[f"stage_{actual_stage}"] = x

        # Head
        x = self._head(x)
        skip_features["head"] = x

        return skip_features

    @property
    def stage_channels(self) -> dict:
        """Return output channels for each skip-connection stage."""
        return {
            stage: self._stage_output_channels[stage - 1]
            for stage in self._SKIP_STAGES
        }

    def get_decoder_channels(self, input_size: int = 256) -> list[int]:
        """Get actual encoder output channels via a dummy forward pass.

        Returns channels from deepest (bottleneck) to shallowest,
        suitable for initializing the ConnectedUnet decoder.
        """
        with torch.no_grad():
            dummy = torch.zeros(1, self._in_channels, input_size, input_size)
            features = self.forward(dummy)

        skip_keys = [7, 6, 4, 2, 1]
        channels = []
        for key in skip_keys:
            if key in features:
                channels.append(features[key].shape[1])
            elif f"stage_{key}" in features:
                channels.append(features[f"stage_{key}"].shape[1])

        if "head" in features:
            channels = [features["head"].shape[1]] + channels[1:] if len(channels) > 1 else [features["head"].shape[1]]

        return channels  # Already deepest-first
