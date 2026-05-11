"""Round-based local training for Flower FL clients."""

from __future__ import annotations

import time

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from fl.config import FLClientConfig
from fl.parameters import get_parameters_as_numpy, load_parameters_into_model
from .metrics import SegmentationMetrics


class FLLocalTrainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        config: FLClientConfig,
        device: torch.device,
    ):
        self._model = model.to(device)
        self._loss_fn = loss_fn.to(device)
        self._cfg = config
        self._device = device

    def get_parameters(self) -> list:
        return get_parameters_as_numpy(self._model)

    def set_parameters(self, parameters: list) -> None:
        load_parameters_into_model(self._model, parameters)

    def fit(self, train_loader: DataLoader, local_epochs: int, round_num: int) -> dict:
        del round_num
        start = time.time()
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self._cfg.learning_rate,
            weight_decay=self._cfg.weight_decay,
            betas=(0.9, 0.999),
        )
        scaler = GradScaler("cuda", enabled=self._cfg.use_amp and self._device.type == "cuda")
        metrics = SegmentationMetrics()
        total_loss = 0.0
        total_samples = 0

        self._model.train()
        for _ in range(local_epochs):
            for batch in train_loader:
                images = batch["image"].to(self._device, non_blocking=True)
                masks = batch["mask"].to(self._device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with autocast("cuda", enabled=self._cfg.use_amp and self._device.type == "cuda"):
                    outputs = self._model(images)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    loss = self._loss_fn(outputs, masks)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

                batch_size = images.size(0)
                total_loss += float(loss.detach().item()) * batch_size
                total_samples += batch_size
                metrics.update(outputs.detach(), masks)

        computed = metrics.compute()
        return {
            "train_loss": total_loss / max(total_samples, 1),
            "train_dice": computed.get("dice_mean", 0.0),
            "train_iou": computed.get("iou_mean", 0.0),
            "n_samples": total_samples,
            "epochs_completed": local_epochs,
            "wall_time_sec": time.time() - start,
        }

    def evaluate(self, loader: DataLoader) -> dict:
        self._model.eval()
        metrics = SegmentationMetrics()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self._device, non_blocking=True)
                masks = batch["mask"].to(self._device, non_blocking=True)
                with autocast("cuda", enabled=self._cfg.use_amp and self._device.type == "cuda"):
                    outputs = self._model(images)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    loss = self._loss_fn(outputs, masks)
                batch_size = images.size(0)
                total_loss += float(loss.detach().item()) * batch_size
                total_samples += batch_size
                metrics.update(outputs, masks)
        computed = metrics.compute()
        return {
            "loss": total_loss / max(total_samples, 1),
            "dice": computed.get("dice_mean", 0.0),
            "iou": computed.get("iou_mean", 0.0),
            "n_samples": total_samples,
        }

