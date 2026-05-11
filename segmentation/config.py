"""
Configuration for ConnectedUnet breast cancer segmentation pipeline.

All hyperparameters, paths, and training settings are centralized here
following the ML engineering best-practice of single-source-of-truth configs.

Reference: EfficientNet paper (Tan & Le, 2019) compound scaling coefficients:
    EfficientNet-B0: width=1.0, depth=1.0, resolution=224
    EfficientNet-B1: width=1.0, depth=1.1, resolution=240
    EfficientNet-B2: width=1.1, depth=1.2, resolution=260
    EfficientNet-B3: width=1.2, depth=1.4, resolution=300
    EfficientNet-B4: width=1.4, depth=1.8, resolution=380
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal


# ── Project Root Resolution ──────────────────────────────────────────
ROOT = Path(os.environ.get("HARMONIA_ROOT", Path(__file__).resolve().parent.parent))
DATA_DIR = ROOT / "outputs" / "final_npy"
OUTPUT_DIR = ROOT / "outputs" / "segmentation"
LOG_DIR = ROOT / "logs" / "segmentation"


# ── EfficientNet Variant Configs (from paper Table 1) ────────────────
EFFICIENTNET_CONFIGS = {
    "b0": {
        "width_coefficient": 1.0,
        "depth_coefficient": 1.0,
        "resolution": 224,
        "dropout_rate": 0.2,
        "stem_channels": 32,
        "block_config": [
            # (expand_ratio, kernel_size, stride, out_channels, num_layers, se_ratio)
            (1, 3, 1, 16, 1, 0.25),   # Stage 1
            (6, 3, 2, 24, 2, 0.25),   # Stage 2
            (6, 5, 2, 40, 2, 0.25),   # Stage 3
            (6, 3, 2, 80, 3, 0.25),   # Stage 4
            (6, 5, 1, 112, 3, 0.25),  # Stage 5
            (6, 5, 2, 192, 4, 0.25),  # Stage 6
            (6, 3, 1, 320, 1, 0.25),  # Stage 7
        ],
    },
    "b1": {
        "width_coefficient": 1.0,
        "depth_coefficient": 1.1,
        "resolution": 240,
        "dropout_rate": 0.2,
        "stem_channels": 32,
        "block_config": [
            (1, 3, 1, 16, 1, 0.25),
            (6, 3, 2, 24, 2, 0.25),
            (6, 5, 2, 40, 2, 0.25),
            (6, 3, 2, 80, 3, 0.25),
            (6, 5, 1, 112, 3, 0.25),
            (6, 5, 2, 192, 4, 0.25),
            (6, 3, 1, 320, 1, 0.25),
        ],
    },
    "b2": {
        "width_coefficient": 1.1,
        "depth_coefficient": 1.2,
        "resolution": 260,
        "dropout_rate": 0.3,
        "stem_channels": 32,
        "block_config": [
            (1, 3, 1, 16, 1, 0.25),
            (6, 3, 2, 24, 2, 0.25),
            (6, 5, 2, 40, 2, 0.25),
            (6, 3, 2, 80, 3, 0.25),
            (6, 5, 1, 112, 3, 0.25),
            (6, 5, 2, 192, 4, 0.25),
            (6, 3, 1, 320, 1, 0.25),
        ],
    },
    "b3": {
        "width_coefficient": 1.2,
        "depth_coefficient": 1.4,
        "resolution": 300,
        "dropout_rate": 0.3,
        "stem_channels": 40,
        "block_config": [
            (1, 3, 1, 24, 1, 0.25),
            (6, 3, 2, 32, 2, 0.25),
            (6, 5, 2, 48, 2, 0.25),
            (6, 3, 2, 96, 3, 0.25),
            (6, 5, 1, 136, 3, 0.25),
            (6, 5, 2, 232, 4, 0.25),
            (6, 3, 1, 384, 1, 0.25),
        ],
    },
    "b4": {
        "width_coefficient": 1.4,
        "depth_coefficient": 1.8,
        "resolution": 380,
        "dropout_rate": 0.4,
        "stem_channels": 48,
        "block_config": [
            (1, 3, 1, 24, 1, 0.25),
            (6, 3, 2, 32, 2, 0.25),
            (6, 5, 2, 56, 2, 0.25),
            (6, 3, 2, 112, 3, 0.25),
            (6, 5, 1, 160, 3, 0.25),
            (6, 5, 2, 272, 4, 0.25),
            (6, 3, 1, 448, 1, 0.25),
        ],
    },
}


@dataclass
class ModelConfig:
    """EfficientNet encoder + ConnectedUnet decoder architecture config."""

    # EfficientNet variant
    efficientnet_variant: Literal["b0", "b1", "b2", "b3", "b4"] = "b3"

    # Input
    in_channels: int = 1  # Grayscale mammogram
    input_size: int = 256  # Height = Width

    # Decoder channels (reversed: deepest to shallowest)
    # B3 encoder (width=1.2, depth=1.4): wider features → scale decoder up slightly
    decoder_channels: tuple[int, ...] = (320, 160, 80, 40, 20)

    # Skip connection mode
    use_attention_gate: bool = True  # Attention-gated skip connections
    use_deep_supervision: bool = False  # Deep supervision for multi-scale output

    # Regularization
    dropout: float = 0.1

    # Whether to use pretrained weights (ImageNet) — requires in_channels=3 adaptation
    pretrained: bool = True

    # Output
    num_classes: int = 1  # Binary tumor segmentation

    @property
    def name(self) -> str:
        return f"connectedunet-eff{self.efficientnet_variant}-{self.input_size}"


@dataclass
class DataConfig:
    """Data loading and augmentation configuration."""

    # Paths
    data_dir: Path = DATA_DIR
    x_train_path: Path = DATA_DIR / "X_train.npy"
    masks_train_path: Path = DATA_DIR / "masks_train.npy"
    x_val_path: Path = DATA_DIR / "X_val.npy"
    masks_val_path: Path = DATA_DIR / "masks_val.npy"
    x_test_path: Path = DATA_DIR / "X_test.npy"
    masks_test_path: Path = DATA_DIR / "masks_test.npy"

    # Image properties
    image_size: int = 256
    image_channels: int = 1

    # Mask normalization: convert 0/255 to 0.0/1.0
    mask_threshold: float = 127.5

    # Augmentation (training only; validation/test are deterministic)
    augmentation: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())


@dataclass
class AugmentationConfig:
    """Medical image augmentation for robust segmentation."""

    # Affine transforms
    rotation_degrees: float = 15.0
    translation: float = 0.1  # Fraction of image size
    scale: tuple[float, float] = (0.9, 1.1)
    shear_degrees: float = 5.0

    # Intensity augmentations (medical-imaging safe)
    brightness: tuple[float, float] = (0.85, 1.15)
    contrast: tuple[float, float] = (0.85, 1.15)

    # Elastic deformation (important for medical images)
    elastic_alpha: float = 50.0
    elastic_sigma: float = 5.0

    # Horizontal flip (safe for mammograms; vertical flip changes anatomy)
    horizontal_flip_prob: float = 0.5

    # Probability of applying augmentations
    apply_prob: float = 0.8


@dataclass
class TrainingConfig:
    """Training hyperparameters and scheduling."""

    # Optimization
    optimizer: Literal["adamw", "adam", "sgd"] = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    betas: tuple[float, float] = (0.9, 0.999)

    # Learning rate scheduling
    scheduler: Literal["cosine", "plateau", "onecycle", "none"] = "cosine"
    scheduler_min_lr: float = 1e-7
    scheduler_patience: int = 10  # For ReduceLROnPlateau
    warmup_epochs: int = 10

    # Training loop
    epochs: int = 100
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Mixed precision
    use_amp: bool = True  # Automatic Mixed Precision (float16)

    # Early stopping
    early_stopping_patience: int = 25
    early_stopping_min_delta: float = 1e-4

    # Checkpointing
    checkpoint_frequency: int = 10  # Save every N epochs
    save_best_only: bool = True
    save_last: bool = True

    # Reproducibility
    seed: int = 42
    deterministic: bool = True

    # Device
    device: str = "cuda"  # "cuda", "cpu", or "auto"
    num_workers: int = 4


@dataclass
class LossConfig:
    """Loss function configuration for segmentation.

    After the mask alignment fix (step 8), tumor coverage in the ROI crops is
    ~43% (nearly balanced: ~57% background, ~43% foreground). The old config
    assumed 0.44% coverage from full-mammogram-scale masks — those values
    (pos_weight=100, focal_alpha=0.75) would massively over-predict tumor now.
    Current weights are tuned for the corrected ~43% coverage.
    """

    # Primary loss weights (should sum to ~1.0)
    bce_weight: float = 0.3    # BCE with pos_weight≈1.5 for light background correction
    dice_weight: float = 0.5   # Primary overlap metric — good for shape quality
    focal_weight: float = 0.1  # Focus on hard boundary pixels
    tversky_weight: float = 0.1  # Slight FN penalty for boundary completeness

    # BCE — ~57:43 background:tumor → pos_weight ≈ 1.32; use 1.5 for light positive bias
    bce_pos_weight: float = 1.5

    # Dice
    dice_smooth: float = 1e-4  # Float16-safe: float16 min normal is ~6.1e-5

    # Focal — alpha=0.5 = balanced (no class imbalance to compensate)
    focal_alpha: float = 0.5
    focal_gamma: float = 2.0

    # Tversky (β > α = slightly penalize missed foreground over false alarms)
    tversky_alpha: float = 0.4
    tversky_beta: float = 0.6

    # Boundary loss (for small tumor boundary precision)
    boundary_weight: float = 0.0  # 0 = disabled


@dataclass
class LoggingConfig:
    """Logging and experiment tracking."""

    # TensorBoard
    tensorboard: bool = True
    log_frequency: int = 10  # Steps between logging

    # Console
    verbose: bool = True
    print_frequency: int = 5  # Epochs between print summaries

    # Output directories
    log_dir: Path = LOG_DIR
    output_dir: Path = OUTPUT_DIR

    # Experiment name for grouping
    experiment_name: str = "connectedunet_cbis_ddsm"


# ── Master Config ────────────────────────────────────────────────────
@dataclass
class SegmentationConfig:
    """Master configuration aggregating all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def save(self, path: Path | None = None):
        """Save config as JSON for reproducibility."""
        path = path or self.logging.output_dir / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        config_dict = {
            "model": _dataclass_to_dict(self.model),
            "data": _dataclass_to_dict(self.data),
            "training": _dataclass_to_dict(self.training),
            "loss": _dataclass_to_dict(self.loss),
            "logging": _dataclass_to_dict(self.logging),
        }
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

    @classmethod
    def from_json(cls, path: Path) -> "SegmentationConfig":
        """Load config from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            model=ModelConfig(**data.get("model", {})),
            data=DataConfig(**data.get("data", {})),
            training=TrainingConfig(**data.get("training", {})),
            loss=LossConfig(**data.get("loss", {})),
            logging=LoggingConfig(**data.get("logging", {})),
        )

    def __repr__(self) -> str:
        return (
            f"SegmentationConfig(\n"
            f"  model={self.model.name},\n"
            f"  training=({self.training.epochs} epochs, bs={self.training.batch_size}, "
            f"lr={self.training.learning_rate}),\n"
            f"  loss=(bce_w={self.loss.bce_weight}, dice_w={self.loss.dice_weight}),\n"
            f"  output={self.logging.output_dir}\n"
            f")"
        )


def _dataclass_to_dict(obj) -> dict:
    """Convert dataclass to JSON-serializable dict, handling Path."""
    d = {}
    for k, v in asdict(obj).items():
        if isinstance(v, Path):
            d[k] = str(v)
        elif isinstance(v, dict):
            d[k] = {kk: str(vv) if isinstance(vv, Path) else vv for kk, vv in v.items()}
        else:
            d[k] = v
    return d


# ── Default Config Instance ──────────────────────────────────────────
default_config = SegmentationConfig()
