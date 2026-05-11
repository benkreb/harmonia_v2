"""Configuration dataclasses and YAML loading for FL benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, get_args, get_origin

import torch
import yaml

from segmentation.config import SegmentationConfig


@dataclass
class FLClientConfig:
    local_epochs: int = 5
    optimizer: Literal["adamw"] = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    use_amp: bool = True
    max_grad_norm: float = 1.0
    batch_size: int = 8
    num_workers: int = 2


@dataclass
class FLServerConfig:
    max_rounds: int = 20
    convergence_patience: int = 7
    convergence_min_delta: float = 1e-4
    min_fit_clients: int = 2
    min_available_clients: int = 2
    eval_batch_size: int = 16


@dataclass
class FLPartitionConfig:
    scenario: Literal["iid", "noniid_80_20"] = "iid"
    partition_json_path: Path = Path("outputs/partitions/iid.json")
    n_clients: int = 2


@dataclass
class FLPHEConfig:
    enabled: bool = False
    poly_modulus_degree: int = 8192
    coeff_mod_bit_sizes: list[int] = field(default_factory=lambda: [60, 40, 40, 60])
    global_scale_bits: int = 40
    chunk_size: int = 8192


@dataclass
class FLSimulationConfig:
    num_cpus_per_client: int = 2
    num_gpus_per_client: float = 1.0
    ray_log_to_driver: bool = False


@dataclass
class FLBenchmarkConfig:
    name: str = "f1_iid"
    seed: int = 42
    output_dir: Path = Path("outputs/fl/f1_iid")
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    client: FLClientConfig = field(default_factory=FLClientConfig)
    server: FLServerConfig = field(default_factory=FLServerConfig)
    partition: FLPartitionConfig = field(default_factory=FLPartitionConfig)
    phe: FLPHEConfig = field(default_factory=FLPHEConfig)
    simulation: FLSimulationConfig = field(default_factory=FLSimulationConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "FLBenchmarkConfig":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        config = cls()
        _update_dataclass(config, raw)
        config.output_dir = Path(str(config.output_dir).replace("<name>", config.name))
        config.segmentation.logging.output_dir = config.output_dir
        config.segmentation.logging.log_dir = Path("logs") / "fl" / config.name
        config.segmentation.training.batch_size = config.client.batch_size
        config.segmentation.training.num_workers = config.client.num_workers
        config.segmentation.training.use_amp = config.client.use_amp
        if config.segmentation.training.device == "cuda" and not torch.cuda.is_available():
            config.segmentation.training.device = "cpu"
        return config

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(_to_plain(self), f, sort_keys=False)

    def apply_smoke_overrides(self) -> None:
        self.name = f"{self.name}_smoke"
        self.output_dir = self.output_dir.parent / self.name
        self.segmentation.logging.output_dir = self.output_dir
        self.server.max_rounds = 2
        self.server.convergence_patience = 99
        self.client.local_epochs = 1


def _update_dataclass(instance: Any, values: dict[str, Any]) -> None:
    field_map = {field.name: field for field in fields(instance)}
    for key, value in values.items():
        if key not in field_map:
            continue
        current = getattr(instance, key)
        target_type = field_map[key].type
        if is_dataclass(current) and isinstance(value, dict):
            _update_dataclass(current, value)
        else:
            setattr(instance, key, _coerce_value(value, target_type))


def _coerce_value(value: Any, target_type: Any) -> Any:
    if value is None:
        return None
    if target_type is Path:
        return Path(value)
    origin = get_origin(target_type)
    if origin is Literal:
        return value
    if origin is list:
        return list(value)
    if origin is tuple:
        return tuple(value)
    args = get_args(target_type)
    if Path in args:
        return Path(value)
    return value


def _to_plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {field.name: _to_plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {key: _to_plain(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return value

