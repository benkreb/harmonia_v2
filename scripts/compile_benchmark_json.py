"""Compile benchmark metrics into a UI-ready JSON payload."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("outputs")
OUT = Path("../harmonia-defense-ui/public/data/benchmark.json")


def _clean(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _metric(row: pd.Series, prefix: str) -> dict[str, float | None]:
    return {
        "mean": _clean(row.get(f"{prefix}_mean")),
        "std": _clean(row.get(f"{prefix}_std")),
    }


def main() -> None:
    df = pd.read_csv(ROOT / "benchmark_report/comparison_table.csv")
    configs = []
    for _, row in df.iterrows():
        configs.append(
            {
                "id": row["config"],
                "scenario": row["scenario"],
                "phe": bool(row["phe"]),
                "dice": _metric(row, "dice"),
                "iou": _metric(row, "iou"),
                "sens": _metric(row, "sens"),
                "spec": _metric(row, "spec"),
                "prec": _metric(row, "prec"),
                "pixel_acc": _clean(row.get("pixel_acc")),
                "wall_time_total_sec": _clean(row.get("wall_time_total_sec")),
                "wall_time_avg_sec": _clean(row.get("wall_time_avg_sec")),
                "n_rounds": _clean(row.get("n_rounds")),
                "n_rounds_to_best": _clean(row.get("n_rounds_to_best")),
                "total_comm_mb": _clean(row.get("total_comm_mb")),
                "comm_per_round_mb": _clean(row.get("comm_per_round_mb")),
            }
        )

    convergence = {}
    for cfg_id in ["f1_iid", "f2_iid_phe", "f3_noniid", "f4_noniid_phe"]:
        path = ROOT / f"fl/{cfg_id}/round_log.jsonl"
        if not path.exists():
            continue
        rounds = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        convergence[cfg_id] = [
            {
                "round": row["round"],
                "val_dice": row.get("server_eval", {}).get("dice"),
                "val_iou": row.get("server_eval", {}).get("iou"),
                "val_loss": row.get("server_eval", {}).get("val_loss"),
                "train_loss": row.get("aggregated", {}).get("train_loss"),
                "wall_time_sec": row.get("wall_time_sec"),
            }
            for row in rounds
        ]

    per_sample = {}
    sources = {
        "C1": ROOT / "segmentation/evaluation/evaluation_results.json",
        "F1": ROOT / "fl/f1_iid/test_results_per_sample.json",
        "F2": ROOT / "fl/f2_iid_phe/test_results_per_sample.json",
        "F3": ROOT / "fl/f3_noniid/test_results_per_sample.json",
        "F4": ROOT / "fl/f4_noniid_phe/test_results_per_sample.json",
    }
    for cfg_id, src in sources.items():
        with src.open(encoding="utf-8") as f:
            data = json.load(f)
        if cfg_id == "C1":
            per_sample[cfg_id] = {
                "dice": data["per_sample"]["dice"],
                "iou": data["per_sample"]["iou"],
            }
        else:
            per_sample[cfg_id] = {
                "dice": data["dice"],
                "iou": data["iou"],
            }

    c1 = next(row for row in configs if row["id"] == "C1")
    payload = {
        "configs": configs,
        "convergence": convergence,
        "per_sample": per_sample,
        "c1_reference": {"dice_mean": c1["dice"]["mean"]},
        "generated_at": pd.Timestamp.now().isoformat(),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote benchmark.json: {OUT.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
