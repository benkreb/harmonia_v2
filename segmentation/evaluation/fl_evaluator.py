"""Evaluation helpers for FL global models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader

from fl.config import FLBenchmarkConfig
from fl.factories import build_loss_from_config, build_model_from_config
from fl.parameters import deserialize_parameters_from_file, load_parameters_into_model
from segmentation.data import CBISDDSMDataset, get_val_transforms
from segmentation.data.transforms import ComposeTransforms
from segmentation.evaluation.evaluator import Evaluator
from segmentation.training.metrics import SegmentationMetrics


def run_segmentation_eval(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module | None = None,
    use_amp: bool = True,
) -> tuple[float, dict]:
    model.eval()
    if loss_fn is not None:
        loss_fn = loss_fn.to(device)
    metrics = SegmentationMetrics()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            with autocast("cuda", enabled=use_amp and device.type == "cuda"):
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = loss_fn(outputs, masks) if loss_fn is not None else None
            batch_size = images.size(0)
            if loss is not None:
                total_loss += float(loss.detach().item()) * batch_size
            total_samples += batch_size
            metrics.update(outputs, masks)

    computed = metrics.compute()
    concise = {
        "dice": computed.get("dice_mean", 0.0),
        "iou": computed.get("iou_mean", 0.0),
        "sens": computed.get("sensitivity_mean", 0.0),
        "spec": computed.get("specificity_mean", 0.0),
        "prec": computed.get("precision_mean", 0.0),
        "pixel_acc": computed.get("pixel_accuracy", 0.0),
        **computed,
    }
    return total_loss / max(total_samples, 1), concise


def evaluate_on_test(
    best_params_path: Path,
    test_loader: DataLoader | FLBenchmarkConfig | None = None,
    config: FLBenchmarkConfig | None = None,
) -> dict:
    if config is None:
        if isinstance(test_loader, FLBenchmarkConfig):
            config = test_loader
            test_loader = None
        else:
            raise ValueError("config is required when test_loader is not an FLBenchmarkConfig")
    if test_loader is None:
        test_loader = _build_test_loader(config)

    model = build_model_from_config(config.segmentation)
    params = deserialize_parameters_from_file(best_params_path)
    load_parameters_into_model(model, params)
    evaluator = Evaluator(model, config.segmentation, test_loader)
    results = evaluator.evaluate(save_predictions=True, save_visualizations=False)

    evaluation_predictions_path = config.output_dir / "evaluation" / "test_predictions.npy"
    if evaluation_predictions_path.exists():
        predictions = np.load(evaluation_predictions_path)
        if predictions.ndim == 3:
            predictions = predictions[..., np.newaxis]
        np.save(config.output_dir / "test_predictions.npy", predictions.astype(np.float32))

    output_path = config.output_dir / "test_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_json_default)

    per_sample_path = config.output_dir / "test_results_per_sample.json"
    with open(per_sample_path, "w", encoding="utf-8") as f:
        json.dump(results.get("per_sample", {}), f, indent=2, default=_json_default)
    return results


def _build_test_loader(config: FLBenchmarkConfig) -> DataLoader:
    device = torch.device(config.segmentation.training.device)
    dataset = CBISDDSMDataset(
        data_dir=config.segmentation.data.data_dir,
        split="test",
        transform=ComposeTransforms(get_val_transforms()),
    )
    return DataLoader(
        dataset,
        batch_size=config.server.eval_batch_size,
        shuffle=False,
        num_workers=config.client.num_workers,
        pin_memory=device.type == "cuda",
    )


def build_loss_for_config(config: FLBenchmarkConfig) -> nn.Module:
    return build_loss_from_config(config.segmentation)


def _json_default(value):
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
    except ImportError:
        pass
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

