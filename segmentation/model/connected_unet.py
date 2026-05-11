"""
ConnectedUnet: U-Net variant with EfficientNet encoder and dense-connected decoder.

Architecture Overview:
    Encoder: EfficientNet-B0~B4 (MBConv blocks + SE + Swish)
    Decoder: Connected decoder blocks with attention-gated skip connections
    Bottleneck: Bridge between encoder and decoder with dilated convolutions

Key Design Decisions (from ADR-001):
    1. EfficientNet encoder provides multi-scale features with compound scaling
    2. Attention gates filter skip connections to focus on tumor regions
    3. Dense connections within decoder blocks for gradient flow
    4. Deep supervision optional for improved convergence on small tumors

Reference:
    - Tan & Le (2019) — EfficientNet backbone
    - Ronneberger et al. (2015) — Original U-Net
    - Oktay et al. (2018) — Attention U-Net
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

from .efficientnet import EfficientNet, Swish


# ── Decoder Components ───────────────────────────────────────────────

class AttentionGate(nn.Module):
    """Attention Gate for filtering skip connections.

    From Oktay et al. (2018): "Attention U-Net: Learning Where to Look
    for the Pancreas." The gate uses the gating signal (coarser-scale
    decoder feature) to attend to relevant regions in the skip connection
    (encoder feature at same scale).

    Args:
        g_channels: Number of channels in the gating signal (decoder).
        x_channels: Number of channels in the skip connection (encoder).
        inter_channels: Intermediate (bottleneck) channels.
    """

    def __init__(self, g_channels: int, x_channels: int, inter_channels: int | None = None):
        super().__init__()
        if inter_channels is None:
            inter_channels = max(1, x_channels // 2)

        self.W_g = nn.Sequential(
            nn.Conv2d(g_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # g: gating signal from decoder (upsampled to resolution of x)
        # x: skip connection from encoder (original channels preserved)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        attention = self.psi(self.relu(g1 + x1))
        return x * attention


class DecoderBlock(nn.Module):
    """Connected decoder block with optional attention gate.

    The "Connected" design uses dense-like connections:
    - Receives skip connection from encoder at this resolution
    - Concatenates with upsampled deeper features
    - Optional attention gate filters the skip connection
    - Two convolutions with BatchNorm + ReLU

    Args:
        in_channels: Channels from upsampled deeper decoder output.
        skip_channels: Channels from encoder skip connection.
        out_channels: Output channels for this decoder block.
        use_attention: Whether to apply attention gate to skip connection.
        dropout: Dropout probability after first conv.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_attention: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._use_attention = use_attention

        if use_attention:
            self._attention = AttentionGate(in_channels, skip_channels)

        # After concat: in_channels (from below) + skip_channels
        merged_channels = in_channels + skip_channels

        self._conv = nn.Sequential(
            nn.Conv2d(merged_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Upsample x to match skip resolution
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

        # Apply attention gate to skip connection
        if self._use_attention:
            skip = self._attention(x, skip)

        # Concatenate and process
        x = torch.cat([x, skip], dim=1)
        x = self._conv(x)
        return x


class DilatedBottleneck(nn.Module):
    """Bottleneck bridge with dilated convolutions.

    Multi-scale context aggregation using parallel dilated convolutions
    with different dilation rates, then fused via 1×1 convolution.

    This helps capture tumor context at multiple scales, which is
    critical for CBIS-DDSM where tumors vary significantly in size.

    Args:
        in_channels: Input channels from encoder deepest layer.
        out_channels: Output channels.
        dilations: Tuple of dilation rates for parallel branches.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
    ):
        super().__init__()
        mid_channels = out_channels // len(dilations)

        self._branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels, mid_channels, kernel_size=3,
                    padding=d, dilation=d, bias=False,
                ),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
            )
            for d in dilations
        ])
        self._fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = [branch(x) for branch in self._branches]
        x = torch.cat(branches, dim=1)
        x = self._fuse(x)
        return x


# ── ConnectedUnet ────────────────────────────────────────────────────

class ConnectedUnet(nn.Module):
    """ConnectedUnet with EfficientNet encoder for tumor segmentation.

    Full architecture:
        Input (1×256×256)
        → EfficientNet Encoder
        → Dilated Bottleneck
        → Connected Decoder (5 blocks with attention gates)
        → Output Conv (1×1 → sigmoid)
        → Output (1×256×256)

    The "Connected" aspect:
        1. Attention-gated skip connections from encoder → decoder
        2. Deep supervision at multiple decoder levels (optional)
        3. Dense connectivity via concatenation of skip + decoder features

    Args:
        efficientnet_variant: EfficientNet scale ("b0"–"b4").
        in_channels: Input channels (1 for grayscale, 3 for RGB).
        num_classes: Output channels (1 for binary segmentation).
        decoder_channels: Channels for each decoder block (deepest → shallowest).
        use_attention_gate: Enable attention gates on skip connections.
        use_deep_supervision: Output predictions at multiple scales.
        dropout: Dropout probability in decoder blocks.
        pretrained: Load pretrained EfficientNet weights.
    """

    def __init__(
        self,
        efficientnet_variant: Literal["b0", "b1", "b2", "b3", "b4"] = "b0",
        in_channels: int = 1,
        num_classes: int = 1,
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16),
        use_attention_gate: bool = True,
        use_deep_supervision: bool = False,
        dropout: float = 0.1,
        pretrained: bool = False,
    ):
        super().__init__()

        self._use_deep_supervision = use_deep_supervision
        self._num_classes = num_classes

        # ── Encoder ──
        self._encoder = EfficientNet(
            variant=efficientnet_variant,
            in_channels=in_channels,
        )
        # Resolve input_size: needed to determine encoder channels
        encoder_input_size = 256  # default; will be overridden by config if different
        encoder_chs = self._encoder.get_decoder_channels(input_size=encoder_input_size)

        # Pad/trim decoder_channels to match encoder skip count
        num_skips = len(encoder_chs)
        if len(decoder_channels) < num_skips:
            decoder_channels = tuple(decoder_channels) + tuple(
                decoder_channels[-1] // (2 ** (i + 1))
                for i in range(num_skips - len(decoder_channels))
            )
        decoder_channels = decoder_channels[:num_skips]

        # ── Bottleneck ──
        bottleneck_in = encoder_chs[0]  # Deepest encoder output
        bottleneck_out = decoder_channels[0]
        self._bottleneck = DilatedBottleneck(
            in_channels=bottleneck_in,
            out_channels=bottleneck_out,
            dilations=(1, 2, 4, 8),
        )

        # ── Decoder ──
        # decoder_channels[0] is bottleneck output
        # decoder_channels[1:] correspond to encoder_chs[1:]
        self._decoder_blocks = nn.ModuleList()
        for i in range(1, len(decoder_channels)):
            self._decoder_blocks.append(
                DecoderBlock(
                    in_channels=decoder_channels[i - 1],
                    skip_channels=encoder_chs[i],
                    out_channels=decoder_channels[i],
                    use_attention=use_attention_gate,
                    dropout=dropout,
                )
            )

        # ── Final output convolution ──
        self._final_conv = nn.Conv2d(
            decoder_channels[-1], num_classes, kernel_size=1
        )

        # Deep supervision heads (if enabled)
        if use_deep_supervision:
            self._deep_supervision_heads = nn.ModuleList([
                nn.Conv2d(ch, num_classes, kernel_size=1)
                for ch in decoder_channels[1:-1]  # Skip deepest and shallowest
            ])
        else:
            self._deep_supervision_heads = None

        # Weight initialization for decoder
        self._initialize_decoder()

        # Bias the final conv to start slightly negative.
        # sigmoid(-1.0+ε) ≈ 0.27 — the model starts neutral and must
        # learn to activate for tumor pixels. sigmoid(+1.0+ε) ≈ 0.73
        # would bias toward predicting everything as tumor, which at
        # 0.44% coverage produces 99.56% false positives.
        nn.init.constant_(self._final_conv.bias, -1.0)

    def _initialize_decoder(self):
        """Kaiming initialization for decoder convolutions."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            If use_deep_supervision=False:
                Output logits tensor (B, num_classes, H, W).
            If use_deep_supervision=True:
                (final_output, list_of_deep_outputs) for multi-scale loss.
        """
        # ── Encoder ──
        input_size = x.shape[2:]  # Original input resolution (e.g., 256×256)
        encoder_features = self._encoder(x)

        # Extract skip features in decoder order (deepest → shallowest)
        # Encoder returns: {0: stem, 1: s1, 2: s2, 4: s4, 6: s6, 7: s7, 'head': head}
        # We need: [head/s7, s6, s4, s2, s1] in that order
        skip_keys = [7, 6, 4, 2, 1]  # Stages from paper
        skips = []
        for key in skip_keys:
            if key in encoder_features:
                skips.append(encoder_features[key])
            elif f"stage_{key}" in encoder_features:
                skips.append(encoder_features[f"stage_{key}"])

        # Replace deepest skip with head feature (more semantic depth)
        if "head" in encoder_features:
            skips = [encoder_features["head"]] + skips[1:] if len(skips) > 1 else [encoder_features["head"]]

        # ── Bottleneck ──
        x = self._bottleneck(skips[0])

        # ── Decoder ──
        deep_outputs = []
        for i, decoder_block in enumerate(self._decoder_blocks):
            x = decoder_block(x, skips[i + 1])

            # Collect deep supervision outputs
            if self._use_deep_supervision and self._deep_supervision_heads and i < len(self._decoder_blocks) - 1:
                deep_out = self._deep_supervision_heads[i](x)
                deep_out = F.interpolate(
                    deep_out, size=input_size,
                    mode="bilinear", align_corners=False,
                )
                deep_outputs.append(deep_out)

        # ── Final output ──
        x = self._final_conv(x)

        # Upsample to original input resolution (encoder stem does stride 2)
        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)

        if self._use_deep_supervision and deep_outputs:
            return x, deep_outputs
        return x

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Inference-mode prediction with sigmoid and thresholding.

        Args:
            x: Input tensor (B, C, H, W).
            threshold: Binarization threshold (default 0.5).

        Returns:
            Binary mask tensor (B, 1, H, W) with values 0.0 or 1.0.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.sigmoid(logits)
            return (probs > threshold).float()

    def get_model_size(self) -> dict:
        """Report model size statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder_params = sum(p.numel() for p in self._encoder.parameters())
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "encoder_params": encoder_params,
            "decoder_params": total_params - encoder_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # float32
        }
