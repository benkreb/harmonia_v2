#!/usr/bin/env python3
"""
Main entry point for ConnectedUnet breast cancer tumor segmentation pipeline.

Usage:
    python run.py train                  # Train a new model
    python run.py train --resume PATH    # Resume from checkpoint
    python run.py evaluate               # Evaluate on test set
    python run.py evaluate --ckpt PATH   # Evaluate with specific checkpoint
    python run.py train-eval             # Train + evaluate

Configuration is loaded from segmentation/config.py defaults.
Override via environment variables or by editing default_config.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from segmentation.config import SegmentationConfig, default_config
from segmentation.model import ConnectedUnet
from segmentation.data import CBISDDSMDataset, get_train_transforms, get_val_transforms
from segmentation.data.transforms import ComposeTransforms
from segmentation.training import CombinedLoss, DeepSupervisionLoss, SegmentationMetrics
from segmentation.training.trainer import Trainer
from segmentation.evaluation.evaluator import Evaluator

import torch
from torch.utils.data import DataLoader


def build_model(config: SegmentationConfig):
    """Instantiate ConnectedUnet from config."""
    model = ConnectedUnet(
        efficientnet_variant=config.model.efficientnet_variant,
        in_channels=config.model.in_channels,
        num_classes=config.model.num_classes,
        decoder_channels=config.model.decoder_channels,
        use_attention_gate=config.model.use_attention_gate,
        use_deep_supervision=config.model.use_deep_supervision,
        dropout=config.model.dropout,
        pretrained=config.model.pretrained,
    )

    size_info = model.get_model_size()
    print(f"[Model] {config.model.name}")
    print(f"[Model] Params: {size_info['total_params']:,} "
          f"(trainable: {size_info['trainable_params']:,}) "
          f"Size: {size_info['model_size_mb']:.1f} MB")
    print(f"[Model] Encoder: {size_info['encoder_params']:,} params")
    print(f"[Model] Decoder: {size_info['decoder_params']:,} params")

    return model


def build_dataloaders(config: SegmentationConfig):
    """Create train/val/test dataloaders."""
    data_cfg = config.data

    train_transforms = ComposeTransforms(get_train_transforms())
    val_transforms = ComposeTransforms(get_val_transforms())

    train_dataset = CBISDDSMDataset(
        data_dir=data_cfg.data_dir,
        split="train",
        transform=train_transforms,
    )
    val_dataset = CBISDDSMDataset(
        data_dir=data_cfg.data_dir,
        split="val",
        transform=val_transforms,
    )
    test_dataset = CBISDDSMDataset(
        data_dir=data_cfg.data_dir,
        split="test",
        transform=val_transforms,
    )

    print(f"[Data] Train: {train_dataset.n_samples} samples")
    print(f"[Data] Val:   {val_dataset.n_samples} samples")
    print(f"[Data] Test:  {test_dataset.n_samples} samples")

    mask_stats = train_dataset.get_mask_coverage_stats()
    print(f"[Data] Tumor coverage: {mask_stats['mean_coverage']:.4f} "
          f"({mask_stats['n_empty_masks']} empty masks)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def build_loss(config: SegmentationConfig):
    """Create loss function from config."""
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
        loss_fn = DeepSupervisionLoss(base_loss)
        print("[Loss] Using deep supervision loss")
    else:
        loss_fn = base_loss

    print(f"[Loss] BCE(w={config.loss.bce_weight}, pos_w={config.loss.bce_pos_weight}), "
          f"Dice(w={config.loss.dice_weight}), "
          f"Focal(w={config.loss.focal_weight}, α={config.loss.focal_alpha}, γ={config.loss.focal_gamma}), "
          f"Tversky(w={config.loss.tversky_weight}, α={config.loss.tversky_alpha}, β={config.loss.tversky_beta})")
    return loss_fn


def cmd_train(config: SegmentationConfig, resume_path: str | None = None):
    """Run training."""
    model = build_model(config)
    train_loader, val_loader, _ = build_dataloaders(config)
    loss_fn = build_loss(config)

    trainer = Trainer(
        model=model,
        config=config,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    if resume_path:
        trainer.resume_from(resume_path)

    config.save(config.logging.output_dir / "config.json")
    history = trainer.train()
    return trainer


def cmd_evaluate(config: SegmentationConfig, checkpoint_path: str | None = None):
    """Run evaluation on test set."""
    model = build_model(config)
    _, _, test_loader = build_dataloaders(config)

    evaluator = Evaluator(model, config, test_loader)

    if checkpoint_path:
        evaluator.load_checkpoint(checkpoint_path)
    else:
        # Try to find best model
        best_path = config.logging.output_dir / "checkpoints" / "best_model.pt"
        if best_path.exists():
            evaluator.load_checkpoint(best_path)
        else:
            print("[Evaluate] No checkpoint found. Evaluating with random weights.")

    results = evaluator.evaluate(save_predictions=True, save_visualizations=True)
    print(evaluator.summarize(results))
    return evaluator


def cmd_train_eval(config: SegmentationConfig, resume_path: str | None = None):
    """Train and then evaluate."""
    trainer = cmd_train(config, resume_path)
    best_path = config.logging.output_dir / "checkpoints" / "best_model.pt"
    cmd_evaluate(config, str(best_path))


def main():
    parser = argparse.ArgumentParser(
        description="ConnectedUnet Breast Cancer Tumor Segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py train
  python run.py train --resume outputs/segmentation/checkpoints/best_model.pt
  python run.py evaluate
  python run.py evaluate --ckpt outputs/segmentation/checkpoints/best_model.pt
  python run.py train-eval
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # train
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from",
    )

    # evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate on test set")
    eval_parser.add_argument(
        "--ckpt", type=str, default=None,
        help="Path to specific checkpoint (default: best_model.pt)",
    )

    # train-eval
    subparsers.add_parser("train-eval", help="Train and then evaluate")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    config = default_config
    config.logging.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"ConnectedUnet + EfficientNet Segmentation Pipeline")
    print(f"Config: {config}")
    print(f"{'=' * 60}\n")

    if args.command == "train":
        cmd_train(config, resume_path=args.resume)
    elif args.command == "evaluate":
        cmd_evaluate(config, checkpoint_path=args.ckpt)
    elif args.command == "train-eval":
        cmd_train_eval(config, resume_path=getattr(args, 'resume', None))


if __name__ == "__main__":
    main()
