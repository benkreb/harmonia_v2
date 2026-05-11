"""
Training loop for ConnectedUnet breast cancer tumor segmentation.

Handles:
    - Training/validation loop with configurable metrics
    - Automatic Mixed Precision (AMP) for memory efficiency
    - Gradient accumulation for large effective batch sizes
    - Cosine annealing / ReduceLROnPlateau / OneCycle LR scheduling
    - Warmup phase for stable early training
    - Checkpointing: best model (by val Dice) + periodic saves
    - Early stopping with configurable patience
    - TensorBoard logging
    - Deep supervision support
"""

from __future__ import annotations

import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from pathlib import Path
from typing import Any, Callable

from segmentation.config import SegmentationConfig
from .metrics import SegmentationMetrics


class Trainer:
    """Training engine for ConnectedUnet segmentation.

    Manages the full training lifecycle: data iteration, optimization,
    LR scheduling, validation, logging, checkpointing, and early stopping.

    Args:
        model: ConnectedUnet instance.
        config: Full segmentation configuration.
        loss_fn: Combined loss function (or DeepSupervisionLoss).
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        callbacks: Optional list of per-epoch callbacks.
    """

    def __init__(
        self,
        model: nn.Module,
        config: SegmentationConfig,
        loss_fn: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        callbacks: list[Callable] | None = None,
    ):
        self._cfg = config
        self._tcfg = config.training
        self._lcfg = config.logging

        self._model = model
        self._loss_fn = loss_fn
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._callbacks = callbacks or []

        # Device
        self._device = self._resolve_device()
        self._model.to(self._device)
        self._loss_fn.to(self._device)

        # Optimizer
        self._optimizer = self._build_optimizer()

        # Scheduler
        self._scheduler, self._scheduler_step_metric = self._build_scheduler()

        # AMP
        self._scaler = GradScaler("cuda", enabled=self._tcfg.use_amp)

        # State tracking
        self._current_epoch: int = 0
        self._global_step: int = 0
        self._best_val_dice: float = 0.0
        self._best_epoch: int = 0
        self._patience_counter: int = 0
        self._epochs_without_improvement: int = 0
        self._history: list[dict] = []

        # Output dirs
        self._checkpoint_dir = self._lcfg.output_dir / "checkpoints"
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        self._writer = None
        if self._lcfg.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(
                    log_dir=str(self._lcfg.log_dir / self._lcfg.experiment_name)
                )
            except ImportError:
                pass

        # Seed
        self._set_seed()

    # ── Device ───────────────────────────────────────────────────

    def _resolve_device(self) -> torch.device:
        device_str = self._tcfg.device
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_str == "cuda" and not torch.cuda.is_available():
            print("[Trainer] CUDA requested but not available; falling back to CPU.")
            return torch.device("cpu")
        return torch.device(device_str)

    def _set_seed(self):
        torch.manual_seed(self._tcfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._tcfg.seed)
        if self._tcfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # ── Optimizer ────────────────────────────────────────────────

    def _build_optimizer(self):
        opt_name = self._tcfg.optimizer
        lr = self._tcfg.learning_rate
        wd = self._tcfg.weight_decay

        if opt_name == "adamw":
            return torch.optim.AdamW(
                self._model.parameters(),
                lr=lr,
                betas=self._tcfg.betas,
                weight_decay=wd,
            )
        elif opt_name == "adam":
            return torch.optim.Adam(
                self._model.parameters(),
                lr=lr,
                betas=self._tcfg.betas,
                weight_decay=wd,
            )
        elif opt_name == "sgd":
            return torch.optim.SGD(
                self._model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=wd,
                nesterov=True,
            )
        raise ValueError(f"Unknown optimizer: {opt_name}")

    # ── LR Scheduler ─────────────────────────────────────────────

    def _build_scheduler(self):
        sched = self._tcfg.scheduler
        if sched == "none":
            return None, None

        if sched == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self._optimizer,
                T_max=self._tcfg.epochs - self._tcfg.warmup_epochs,
                eta_min=self._tcfg.scheduler_min_lr,
            )
            return scheduler, "epoch"

        if sched == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self._optimizer,
                mode="max",
                factor=0.5,
                patience=self._tcfg.scheduler_patience,
                min_lr=self._tcfg.scheduler_min_lr,
            )
            return scheduler, "metric"

        if sched == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self._optimizer,
                max_lr=self._tcfg.learning_rate,
                epochs=self._tcfg.epochs,
                steps_per_epoch=len(self._train_loader),
                pct_start=0.1,
                div_factor=25.0,
                final_div_factor=1e4,
            )
            return scheduler, "step"

        raise ValueError(f"Unknown scheduler: {sched}")

    # ── Training Loop ────────────────────────────────────────────

    def train(self) -> dict:
        """Run the full training pipeline.

        Returns:
            Training history with metrics per epoch.
        """
        print(f"[Trainer] Starting training on {self._device}")
        print(f"[Trainer] Model: {self._model.__class__.__name__}")
        print(f"[Trainer] Epochs: {self._tcfg.epochs}, Batch size: {self._tcfg.batch_size}")
        print(f"[Trainer] Optimizer: {self._tcfg.optimizer}, LR: {self._tcfg.learning_rate}")
        print(f"[Trainer] Scheduler: {self._tcfg.scheduler}, AMP: {self._tcfg.use_amp}")

        for epoch in range(self._tcfg.epochs):
            self._current_epoch = epoch

            # Warmup
            if epoch < self._tcfg.warmup_epochs:
                warmup_lr = self._tcfg.learning_rate * (epoch + 1) / self._tcfg.warmup_epochs
                for param_group in self._optimizer.param_groups:
                    param_group["lr"] = warmup_lr

            epoch_start = time.time()

            # Train
            train_metrics = self._run_epoch(self._train_loader, training=True)

            # Validate
            val_metrics = self._run_epoch(self._val_loader, training=False)

            # Step scheduler
            self._step_scheduler(val_metrics.get("dice_mean", 0.0))

            # Log epoch
            elapsed = time.time() - epoch_start
            self._log_epoch(epoch, train_metrics, val_metrics, elapsed)

            # Checkpointing
            val_dice = val_metrics.get("dice_mean", 0.0)
            is_best = val_dice > self._best_val_dice
            if is_best:
                self._best_val_dice = val_dice
                self._best_epoch = epoch
                self._patience_counter = 0
            else:
                self._patience_counter += 1

            self._save_checkpoint(epoch, is_best)

            # Early stopping
            if self._patience_counter >= self._tcfg.early_stopping_patience:
                print(f"[Trainer] Early stopping triggered at epoch {epoch + 1}")
                break

            # Callbacks
            for cb in self._callbacks:
                cb(epoch, train_metrics, val_metrics)

        # Cleanup
        if self._writer:
            self._writer.close()

        print(f"[Trainer] Training complete. Best Dice: {self._best_val_dice:.4f} at epoch {self._best_epoch + 1}")
        return self._history

    def _run_epoch(self, loader: DataLoader, training: bool) -> dict[str, float]:
        if training:
            self._model.train()
        else:
            self._model.eval()

        metrics = SegmentationMetrics()
        total_loss = 0.0
        total_samples = 0

        for batch_idx, batch in enumerate(loader):
            images = batch["image"].to(self._device, non_blocking=True)
            masks = batch["mask"].to(self._device, non_blocking=True)

            if training:
                self._optimizer.zero_grad()

                with autocast("cuda", enabled=self._tcfg.use_amp):
                    outputs = self._model(images)
                    loss = self._loss_fn(outputs, masks)
                    loss = loss / self._tcfg.gradient_accumulation_steps

                self._scaler.scale(loss).backward()

                if (batch_idx + 1) % self._tcfg.gradient_accumulation_steps == 0:
                    self._scaler.unscale_(self._optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self._model.parameters(), self._tcfg.max_grad_norm
                    )
                    self._scaler.step(self._optimizer)
                    self._scaler.update()
                    self._optimizer.zero_grad()

                # Step OneCycle scheduler per batch
                if (
                    self._scheduler is not None
                    and self._scheduler_step_metric == "step"
                ):
                    self._scheduler.step()

                self._global_step += 1
            else:
                with torch.no_grad():
                    with autocast("cuda", enabled=self._tcfg.use_amp):
                        outputs = self._model(images)

            # For metrics, handle deep supervision (take final output)
            if isinstance(outputs, tuple):
                final_output = outputs[0]
            else:
                final_output = outputs

            metrics.update(final_output, masks)

            bs = images.size(0)
            total_samples += bs

            if training:
                total_loss += loss.item() * bs * self._tcfg.gradient_accumulation_steps

            # TensorBoard step logging
            if self._writer and training and batch_idx % self._lcfg.log_frequency == 0:
                self._writer.add_scalar(
                    "train/loss_step", loss.item() * self._tcfg.gradient_accumulation_steps,
                    self._global_step,
                )

        computed = metrics.compute()
        if training:
            computed["loss"] = total_loss / max(total_samples, 1)
        return computed

    # ── Scheduler Step ───────────────────────────────────────────

    def _step_scheduler(self, val_dice: float):
        if self._scheduler is None:
            return
        if self._current_epoch < self._tcfg.warmup_epochs:
            return

        if self._scheduler_step_metric == "epoch":
            self._scheduler.step()
        elif self._scheduler_step_metric == "metric":
            self._scheduler.step(val_dice)

    # ── Logging ──────────────────────────────────────────────────

    def _log_epoch(
        self,
        epoch: int,
        train_metrics: dict,
        val_metrics: dict,
        elapsed: float,
    ):
        current_lr = self._optimizer.param_groups[0]["lr"]

        entry = {
            "epoch": epoch + 1,
            "lr": current_lr,
            "train_loss": round(float(train_metrics.get("loss", 0)), 6),
            "val_dice": round(float(val_metrics.get("dice_mean", 0)), 6),
            "val_iou": round(float(val_metrics.get("iou_mean", 0)), 6),
            "val_sensitivity": round(float(val_metrics.get("sensitivity_mean", 0)), 6),
            "val_specificity": round(float(val_metrics.get("specificity_mean", 0)), 6),
            "val_precision": round(float(val_metrics.get("precision_mean", 0)), 6),
            "time_sec": round(elapsed, 1),
        }
        self._history.append(entry)

        if self._lcfg.verbose and (epoch % self._lcfg.print_frequency == 0 or epoch < 5):
            # Per-loss breakdown if available (CombinedLoss exposes last_losses)
            loss_parts = ""
            if hasattr(self._loss_fn, 'last_losses'):
                parts = self._loss_fn.last_losses
                loss_parts = " | ".join(f"{k}={parts[k]:.4f}" for k in sorted(parts.keys()))
            print(
                f"Epoch {epoch + 1:3d}/{self._tcfg.epochs} | "
                f"LR: {current_lr:.2e} | "
                f"Loss: {entry['train_loss']:.4f} | "
                f"Val Dice: {entry['val_dice']:.4f} | "
                f"Val IoU: {entry['val_iou']:.4f} | "
                f"Time: {elapsed:.1f}s"
            )
            if loss_parts:
                print(f"  └─ {loss_parts}")

        if self._writer:
            self._writer.add_scalar("train/loss_epoch", entry["train_loss"], epoch)
            self._writer.add_scalar("val/dice", entry["val_dice"], epoch)
            self._writer.add_scalar("val/iou", entry["val_iou"], epoch)
            self._writer.add_scalar("val/sensitivity", entry["val_sensitivity"], epoch)
            self._writer.add_scalar("val/specificity", entry["val_specificity"], epoch)
            self._writer.add_scalar("val/precision", entry["val_precision"], epoch)
            self._writer.add_scalar("train/lr", current_lr, epoch)

    # ── Checkpointing ────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, is_best: bool):
        ckpt = {
            "epoch": epoch + 1,
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "scheduler_state_dict": self._scheduler.state_dict() if self._scheduler else None,
            "scaler_state_dict": self._scaler.state_dict(),
            "best_val_dice": self._best_val_dice,
            "best_epoch": self._best_epoch,
            "config": self._cfg,
            "history": self._history,
        }

        # Periodic save
        if self._tcfg.save_last or (
            self._tcfg.checkpoint_frequency > 0
            and (epoch + 1) % self._tcfg.checkpoint_frequency == 0
        ):
            path = self._checkpoint_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt"
            torch.save(ckpt, path)

        # Best model save
        if is_best and self._tcfg.save_best_only:
            path = self._checkpoint_dir / "best_model.pt"
            torch.save(ckpt, path)

        # Save history
        history_path = self._lcfg.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self._history, f, indent=2, default=str)

    # ── Resume ───────────────────────────────────────────────────

    def resume_from(self, checkpoint_path: Path | str):
        """Resume training from a saved checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=self._device, weights_only=False)
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self._scheduler and ckpt.get("scheduler_state_dict"):
            self._scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ckpt.get("scaler_state_dict"):
            self._scaler.load_state_dict(ckpt["scaler_state_dict"])
        self._best_val_dice = ckpt.get("best_val_dice", 0.0)
        self._best_epoch = ckpt.get("best_epoch", 0)
        self._current_epoch = ckpt.get("epoch", 0)
        self._history = ckpt.get("history", [])
        print(f"[Trainer] Resumed from epoch {self._current_epoch} | Best Dice: {self._best_val_dice:.4f}")

    # ── Properties ───────────────────────────────────────────────

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def best_val_dice(self) -> float:
        return self._best_val_dice

    @property
    def history(self) -> list[dict]:
        return self._history
