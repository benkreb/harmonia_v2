"""Select 15 defense-demo samples by C1 per-sample Dice."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("outputs")
UI_DATA_DIR = Path("../harmonia-defense-ui/public/data")


def _load_per_config_metrics() -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    sources = {
        "C1": ROOT / "segmentation/evaluation/evaluation_results.json",
        "F1": ROOT / "fl/f1_iid/test_results_per_sample.json",
        "F2": ROOT / "fl/f2_iid_phe/test_results_per_sample.json",
        "F3": ROOT / "fl/f3_noniid/test_results_per_sample.json",
        "F4": ROOT / "fl/f4_noniid_phe/test_results_per_sample.json",
    }
    dice: dict[str, list[float]] = {}
    iou: dict[str, list[float]] = {}
    for cfg, path in sources.items():
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if cfg == "C1":
            dice[cfg] = data["per_sample"]["dice"]
            iou[cfg] = data["per_sample"]["iou"]
        else:
            dice[cfg] = data["dice"]
            iou[cfg] = data["iou"]
    return dice, iou


def main() -> None:
    dice, iou = _load_per_config_metrics()
    c1_dice = np.asarray(dice["C1"], dtype=np.float64)
    sorted_idx = np.argsort(c1_dice)
    n = len(c1_dice)

    best_5 = sorted_idx[-5:][::-1].tolist()
    worst_5 = sorted_idx[:5].tolist()
    median_start = (n // 2) - 2
    median_5 = sorted_idx[median_start : median_start + 5].tolist()

    meta = pd.read_csv(ROOT / "final_npy/metadata_test.csv")
    samples = []
    for category, indices in [("best", best_5), ("median", median_5), ("worst", worst_5)]:
        for idx in indices:
            row = meta.iloc[idx]
            samples.append(
                {
                    "sample_idx": int(idx),
                    "category": category,
                    "patient_id": str(row.get("patient_id", f"P_{idx:05d}")),
                    "pathology": str(row.get("pathology", "unknown")),
                    "breast_density": int(row["breast_density"]) if "breast_density" in meta.columns and not pd.isna(row.get("breast_density")) else None,
                    "dice": {cfg: float(values[idx]) for cfg, values in dice.items()},
                    "iou": {cfg: float(values[idx]) for cfg, values in iou.items()},
                }
            )

    UI_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = UI_DATA_DIR / "samples_curated.json"
    out_path.write_text(json.dumps({"samples": samples, "n_samples": len(samples)}, indent=2), encoding="utf-8")
    print(f"Wrote {len(samples)} curated samples to {out_path}")


if __name__ == "__main__":
    main()
