"""Utilities for converting PyTorch model parameters to Flower NumPy arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def get_parameters_as_numpy(model: nn.Module) -> list[np.ndarray]:
    return [
        value.detach().cpu().numpy().astype(np.float32, copy=True)
        for value in model.state_dict().values()
    ]


def load_parameters_into_model(model: nn.Module, params: list[np.ndarray]) -> None:
    state_dict = model.state_dict()
    if len(params) != len(state_dict):
        raise ValueError(f"Parameter count mismatch: {len(params)} != {len(state_dict)}")

    loaded = {}
    for (key, current), param in zip(state_dict.items(), params):
        tensor = torch.tensor(np.asarray(param), dtype=current.dtype, device=current.device)
        if tuple(tensor.shape) != tuple(current.shape):
            raise ValueError(f"Shape mismatch for {key}: {tuple(tensor.shape)} != {tuple(current.shape)}")
        loaded[key] = tensor
    model.load_state_dict(loaded, strict=True)


def parameter_shapes(model: nn.Module) -> list[tuple[int, ...]]:
    return [tuple(value.shape) for value in model.state_dict().values()]


def serialize_parameters_to_file(params: list[np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"parameters": params}, path)


def deserialize_parameters_from_file(path: Path) -> list[np.ndarray]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "parameters" in payload:
        return payload["parameters"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported parameter file format: {path}")


def parameters_to_state_dict(model: nn.Module, params: list[np.ndarray]) -> dict[str, torch.Tensor]:
    state_dict = model.state_dict()
    return {
        key: torch.as_tensor(param, dtype=current.dtype)
        for (key, current), param in zip(state_dict.items(), params)
    }

