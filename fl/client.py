"""Flower client implementation for Harmonia FL."""

from __future__ import annotations

from typing import Any

from flwr.client import NumPyClient
from torch.utils.data import DataLoader

from segmentation.training.fl_trainer import FLLocalTrainer


class HarmoniaFLClient(NumPyClient):
    def __init__(
        self,
        client_id: str,
        trainer: FLLocalTrainer,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        crypto: Any | None = None,
    ):
        self._client_id = client_id
        self._trainer = trainer
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._crypto = crypto

    def get_parameters(self, config: dict) -> list:
        del config
        params = self._trainer.get_parameters()
        return self._crypto.encrypt_parameters(params) if self._crypto else params

    def fit(self, parameters, config) -> tuple[list, int, dict]:
        if self._crypto:
            parameters = self._crypto.decrypt_parameters(parameters)
        self._trainer.set_parameters(parameters)
        metrics = self._trainer.fit(
            self._train_loader,
            local_epochs=int(config.get("local_epochs", 5)),
            round_num=int(config.get("round_num", 0)),
        )
        new_params = self._trainer.get_parameters()
        if self._crypto:
            new_params = self._crypto.encrypt_parameters(new_params)
        metrics["client_id"] = self._client_id
        return new_params, int(metrics["n_samples"]), metrics

    def evaluate(self, parameters, config) -> tuple[float, int, dict]:
        del config
        if self._val_loader is None:
            return 0.0, 0, {}
        if self._crypto:
            parameters = self._crypto.decrypt_parameters(parameters)
        self._trainer.set_parameters(parameters)
        metrics = self._trainer.evaluate(self._val_loader)
        return float(metrics["loss"]), int(metrics["n_samples"]), metrics

