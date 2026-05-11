"""Server-side Flower strategy and evaluation hooks."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from flwr.common import Metrics, NDArrays, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from torch.utils.data import DataLoader

from fl.communication import CommunicationTracker, parameters_bytes
from fl.config import FLBenchmarkConfig
from fl.parameters import load_parameters_into_model, serialize_parameters_to_file
from segmentation.evaluation.fl_evaluator import run_segmentation_eval


def weighted_mean_metrics(metrics: list[tuple[int, Metrics]]) -> Metrics:
    total = sum(num_examples for num_examples, _ in metrics)
    if total == 0:
        return {}
    keys = {"train_loss", "train_dice", "train_iou", "wall_time_sec"}
    aggregated: Metrics = {}
    for key in keys:
        vals = [num_examples * float(m[key]) for num_examples, m in metrics if key in m]
        if vals:
            aggregated[key] = sum(vals) / total
    return aggregated


class HarmoniaFedAvg(FedAvg):
    def __init__(
        self,
        *,
        output_dir: Path,
        fl_config: FLBenchmarkConfig,
        communication_tracker: CommunicationTracker,
        crypto=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output_dir = output_dir
        self.fl_config = fl_config
        self.communication_tracker = communication_tracker
        self.crypto = crypto
        self.best_val_dice = -float("inf")
        self.best_round = 0
        self.rounds_no_improvement = 0
        self.should_stop = False
        self._round_starts: dict[int, float] = {}
        self._last_fit_metrics: dict[int, Metrics] = {}

    def configure_fit(self, server_round, parameters, client_manager):
        self._round_starts[server_round] = time.time()
        instructions = super().configure_fit(server_round, parameters, client_manager)
        try:
            ndarrays = parameters_to_ndarrays(parameters)
            byte_count = parameters_bytes(ndarrays)
            for client_proxy, _ in instructions:
                self.communication_tracker.record_download(
                    server_round, client_proxy.cid, byte_count, encrypted=self.fl_config.phe.enabled
                )
        except Exception:
            pass
        return instructions

    def aggregate_fit(self, server_round, results, failures):
        for client_proxy, fit_res in results:
            try:
                ndarrays = parameters_to_ndarrays(fit_res.parameters)
                self.communication_tracker.record_upload(
                    server_round,
                    client_proxy.cid,
                    parameters_bytes(ndarrays),
                    encrypted=self.fl_config.phe.enabled,
                )
            except Exception:
                pass
        if self.crypto is not None and results:
            total_examples = sum(fit_res.num_examples for _, fit_res in results)
            weights = [fit_res.num_examples / total_examples for _, fit_res in results]
            encrypted_per_client = [
                parameters_to_ndarrays(fit_res.parameters)
                for _, fit_res in results
            ]
            aggregated_encrypted = self.crypto.homomorphic_weighted_sum(encrypted_per_client, weights)
            metrics = {}
            if self.fit_metrics_aggregation_fn is not None:
                metrics = self.fit_metrics_aggregation_fn([
                    (fit_res.num_examples, fit_res.metrics) for _, fit_res in results
                ])
            self.communication_tracker.flush()
            if metrics:
                self._last_fit_metrics[server_round] = metrics
            return ndarrays_to_parameters(aggregated_encrypted), metrics
        aggregated = super().aggregate_fit(server_round, results, failures)
        self.communication_tracker.flush()
        if aggregated[1]:
            self._last_fit_metrics[server_round] = aggregated[1]
        return aggregated

    def record_server_eval(self, server_round: int, loss: float, metrics: dict, parameters: NDArrays) -> None:
        val_dice = float(metrics.get("dice", metrics.get("dice_mean", 0.0)))
        min_delta = self.fl_config.server.convergence_min_delta
        improved = val_dice > self.best_val_dice + min_delta
        if improved:
            self.best_val_dice = val_dice
            self.best_round = server_round
            self.rounds_no_improvement = 0
            serialize_parameters_to_file(parameters, self.output_dir / "best_params.pt")
        else:
            self.rounds_no_improvement += 1

        serialize_parameters_to_file(parameters, self.output_dir / "final_params.pt")
        self.should_stop = self.rounds_no_improvement >= self.fl_config.server.convergence_patience

        elapsed = time.time() - self._round_starts.get(server_round, time.time())
        self._append_server_eval(server_round, loss, metrics)
        self._append_round_log(server_round, elapsed, loss, metrics)

    def _append_server_eval(self, server_round: int, loss: float, metrics: dict) -> None:
        path = self.output_dir / "server_eval.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "round": server_round,
            "val_loss": loss,
            "val_dice": metrics.get("dice", metrics.get("dice_mean", 0.0)),
            "val_iou": metrics.get("iou", metrics.get("iou_mean", 0.0)),
            "val_sens": metrics.get("sens", metrics.get("sensitivity_mean", 0.0)),
            "val_spec": metrics.get("spec", metrics.get("specificity_mean", 0.0)),
            "val_prec": metrics.get("prec", metrics.get("precision_mean", 0.0)),
        }
        write_header = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _append_round_log(self, server_round: int, elapsed: float, loss: float, metrics: dict) -> None:
        row = {
            "round": server_round,
            "wall_time_sec": elapsed,
            "aggregated": self._last_fit_metrics.get(server_round, {}),
            "server_eval": {"val_loss": loss, **metrics},
            "convergence": {
                "best_val_dice": self.best_val_dice,
                "best_round": self.best_round,
                "rounds_no_improvement": self.rounds_no_improvement,
            },
            "communication": self.communication_tracker.round_total(server_round),
        }
        with open(self.output_dir / "round_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=_json_default) + "\n")


def make_server_eval_fn(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    output_dir: Path,
    strategy_ref: dict,
    use_amp: bool = True,
) -> Callable[[int, NDArrays, dict[str, Scalar]], tuple[float, dict[str, Scalar]]]:
    def evaluate_fn(server_round: int, parameters: NDArrays, config: dict[str, Scalar]):
        del config
        output_dir.mkdir(parents=True, exist_ok=True)
        strategy = strategy_ref.get("strategy")
        if strategy is not None and strategy.crypto is not None:
            parameters = strategy.crypto.decrypt_parameters(parameters)
        load_parameters_into_model(model, parameters)
        loss, metrics = run_segmentation_eval(model, val_loader, device, loss_fn, use_amp=use_amp)
        if strategy is not None:
            strategy.record_server_eval(server_round, loss, metrics, parameters)
        return loss, {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float, np.floating))}

    return evaluate_fn


def initial_parameters_from_ndarrays(params: list[np.ndarray]):
    return ndarrays_to_parameters(params)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)

