"""Regenerate FL test predictions and per-sample metric files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fl.config import FLBenchmarkConfig
from segmentation.evaluation.fl_evaluator import evaluate_on_test


CONFIGS = ["f1_iid", "f2_iid_phe", "f3_noniid", "f4_noniid_phe"]


def _validate_metrics(run_dir: Path, results: dict) -> None:
    per_sample = results.get("per_sample", {})
    dice = np.asarray(per_sample.get("dice", []), dtype=np.float64)
    expected = float(results["metrics"]["dice_mean"])
    actual = float(dice.mean())
    if not np.isclose(actual, expected, atol=1e-4):
        raise RuntimeError(f"{run_dir.name}: per-sample dice mean {actual:.6f} != aggregate {expected:.6f}")

    pred_path = run_dir / "test_predictions.npy"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    preds = np.load(pred_path, mmap_mode="r")
    if preds.shape != (len(dice), 256, 256, 1):
        raise RuntimeError(f"{run_dir.name}: unexpected prediction shape {preds.shape}")

    per_sample_path = run_dir / "test_results_per_sample.json"
    with per_sample_path.open(encoding="utf-8") as f:
        saved = json.load(f)
    if len(saved.get("dice", [])) != len(dice):
        raise RuntimeError(f"{run_dir.name}: per-sample JSON length mismatch")


def main() -> None:
    for name in CONFIGS:
        print(f"\n=== {name} ===")
        cfg = FLBenchmarkConfig.from_yaml(ROOT / f"benchmark/configs/{name}.yaml")
        results = evaluate_on_test(
            best_params_path=ROOT / f"outputs/fl/{name}/best_params.pt",
            config=cfg,
        )
        _validate_metrics(cfg.output_dir, results)
        print(f"Saved {cfg.output_dir / 'test_predictions.npy'}")


if __name__ == "__main__":
    main()
