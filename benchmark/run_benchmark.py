"""Helpers to run Harmonia FL benchmark configurations."""

from __future__ import annotations

from pathlib import Path

from fl.config import FLBenchmarkConfig
from fl.simulation import run_simulation


CONFIGS = [
    Path("benchmark/configs/f1_iid.yaml"),
    Path("benchmark/configs/f3_noniid.yaml"),
    Path("benchmark/configs/f2_iid_phe.yaml"),
    Path("benchmark/configs/f4_noniid_phe.yaml"),
]


def run_config(path: Path, smoke: bool = False):
    config = FLBenchmarkConfig.from_yaml(path)
    if smoke:
        config.apply_smoke_overrides()
    return run_simulation(config)


def run_all(smoke_first: bool = True) -> None:
    for path in CONFIGS:
        if smoke_first:
            run_config(path, smoke=True)
        run_config(path, smoke=False)

