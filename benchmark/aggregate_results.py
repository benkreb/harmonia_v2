"""Aggregate C1 and FL benchmark outputs into report artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from benchmark.plotting import generate_plots


REPORT_DIR = Path("outputs/benchmark_report")
FL_DIR = Path("outputs/fl")
C1_RESULTS = Path("outputs/segmentation/evaluation/evaluation_results.json")


def generate_report(output_dir: Path = REPORT_DIR) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [_centralized_row()]
    for config_name in ["f1_iid", "f2_iid_phe", "f3_noniid", "f4_noniid_phe"]:
        rows.append(_fl_row(config_name))
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "comparison_table.csv", index=False)
    (output_dir / "comparison_table.md").write_text(comparison.to_markdown(index=False), encoding="utf-8")
    generate_plots(comparison, output_dir)
    _write_report_md(comparison, output_dir)
    return comparison


def _centralized_row() -> dict:
    with open(C1_RESULTS, encoding="utf-8") as f:
        data = json.load(f)
    metrics = data["metrics"]
    return {
        "config": "C1",
        "scenario": "centralized",
        "phe": False,
        "dice_mean": metrics.get("dice_mean"),
        "dice_std": metrics.get("dice_std"),
        "iou_mean": metrics.get("iou_mean"),
        "iou_std": metrics.get("iou_std"),
        "sens_mean": metrics.get("sensitivity_mean"),
        "sens_std": metrics.get("sensitivity_std"),
        "spec_mean": metrics.get("specificity_mean"),
        "spec_std": metrics.get("specificity_std"),
        "prec_mean": metrics.get("precision_mean"),
        "prec_std": metrics.get("precision_std"),
        "pixel_acc": metrics.get("pixel_accuracy"),
        "pixel_tp": metrics.get("pixel_tp"),
        "pixel_fp": metrics.get("pixel_fp"),
        "pixel_fn": metrics.get("pixel_fn"),
        "pixel_tn": metrics.get("pixel_tn"),
        "n_test_samples": metrics.get("n_samples"),
        "wall_time_total_sec": None,
        "wall_time_avg_sec": None,
        "n_rounds": None,
        "n_rounds_to_best": None,
        "best_val_dice": None,
        "best_val_iou": None,
        "best_val_loss": None,
        "final_val_dice": None,
        "final_val_iou": None,
        "final_val_loss": None,
        "upload_comm_mb": 0.0,
        "download_comm_mb": 0.0,
        "total_comm_mb": 0.0,
        "comm_per_round_mb": None,
        "phe_overhead_pct": None,
        "phe_per_round_overhead_pct": None,
    }


def _fl_row(config_name: str) -> dict:
    run_dir = FL_DIR / config_name
    test_path = run_dir / "test_results.json"
    if not test_path.exists():
        return {"config": config_name}

    with open(test_path, encoding="utf-8") as f:
        test = json.load(f)
    metrics = test.get("metrics", {})
    server_eval = _read_server_eval(run_dir / "server_eval.csv")
    round_summary = _read_round_log(run_dir / "round_log.jsonl")
    comm_mb = round_summary.get("total_comm_mb", 0.0)
    wall_time = round_summary.get("wall_time_total_sec")
    return {
        "config": config_name,
        "scenario": _scenario(config_name),
        "phe": "_phe" in config_name,
        "dice_mean": metrics.get("dice_mean"),
        "dice_std": metrics.get("dice_std", test.get("dice_std")),
        "iou_mean": metrics.get("iou_mean"),
        "iou_std": metrics.get("iou_std"),
        "sens_mean": metrics.get("sensitivity_mean"),
        "sens_std": metrics.get("sensitivity_std"),
        "spec_mean": metrics.get("specificity_mean"),
        "spec_std": metrics.get("specificity_std"),
        "prec_mean": metrics.get("precision_mean"),
        "prec_std": metrics.get("precision_std"),
        "pixel_acc": metrics.get("pixel_accuracy"),
        "pixel_tp": metrics.get("pixel_tp"),
        "pixel_fp": metrics.get("pixel_fp"),
        "pixel_fn": metrics.get("pixel_fn"),
        "pixel_tn": metrics.get("pixel_tn"),
        "n_test_samples": metrics.get("n_samples"),
        "wall_time_total_sec": wall_time,
        "wall_time_avg_sec": round_summary.get("wall_time_avg_sec"),
        "n_rounds": round_summary.get("n_rounds"),
        "n_rounds_to_best": server_eval.get("best_round"),
        "best_val_dice": server_eval.get("best_val_dice"),
        "best_val_iou": server_eval.get("best_val_iou"),
        "best_val_loss": server_eval.get("best_val_loss"),
        "final_val_dice": server_eval.get("final_val_dice"),
        "final_val_iou": server_eval.get("final_val_iou"),
        "final_val_loss": server_eval.get("final_val_loss"),
        "upload_comm_mb": round_summary.get("upload_comm_mb", 0.0),
        "download_comm_mb": round_summary.get("download_comm_mb", 0.0),
        "total_comm_mb": comm_mb,
        "comm_per_round_mb": round_summary.get("comm_per_round_mb"),
        "phe_overhead_pct": _phe_overhead(config_name, comm_mb),
        "phe_per_round_overhead_pct": _phe_per_round_overhead(config_name, round_summary.get("comm_per_round_mb")),
    }


def _read_server_eval(path: Path) -> dict:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty or "val_dice" not in df:
        return {}
    idx = df["val_dice"].astype(float).idxmax()
    final = df.iloc[-1]
    return {
        "best_round": int(df.loc[idx, "round"]),
        "best_val_dice": float(df.loc[idx, "val_dice"]),
        "best_val_iou": float(df.loc[idx, "val_iou"]),
        "best_val_loss": float(df.loc[idx, "val_loss"]),
        "final_val_dice": float(final["val_dice"]),
        "final_val_iou": float(final["val_iou"]),
        "final_val_loss": float(final["val_loss"]),
    }


def _communication_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    df = pd.read_csv(path)
    return float(df["bytes"].sum() / (1024 * 1024)) if "bytes" in df else 0.0


def _read_round_log(path: Path) -> dict:
    if not path.exists():
        return {}
    total = 0.0
    upload_bytes = 0
    download_bytes = 0
    total_bytes = 0
    n_rounds = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                n_rounds += 1
                total += float(row.get("wall_time_sec", 0.0))
                communication = row.get("communication", {})
                upload_bytes += int(communication.get("upload_bytes", 0))
                download_bytes += int(communication.get("download_bytes", 0))
                total_bytes += int(communication.get("total_bytes", 0))
    mb = 1024 * 1024
    return {
        "n_rounds": n_rounds,
        "wall_time_total_sec": total,
        "wall_time_avg_sec": total / n_rounds if n_rounds else None,
        "upload_comm_mb": upload_bytes / mb,
        "download_comm_mb": download_bytes / mb,
        "total_comm_mb": total_bytes / mb,
        "comm_per_round_mb": (total_bytes / mb) / n_rounds if n_rounds else None,
    }


def _phe_overhead(config_name: str, comm_mb: float) -> float | None:
    baseline = {"f2_iid_phe": "f1_iid", "f4_noniid_phe": "f3_noniid"}.get(config_name)
    if baseline is None:
        return None
    base_mb = _read_round_log(FL_DIR / baseline / "round_log.jsonl").get("total_comm_mb", 0.0)
    if base_mb <= 0:
        return None
    return ((comm_mb - base_mb) / base_mb) * 100.0


def _phe_per_round_overhead(config_name: str, comm_per_round_mb: float | None) -> float | None:
    baseline = {"f2_iid_phe": "f1_iid", "f4_noniid_phe": "f3_noniid"}.get(config_name)
    if baseline is None or comm_per_round_mb is None:
        return None
    base_mb = _read_round_log(FL_DIR / baseline / "round_log.jsonl").get("comm_per_round_mb", 0.0)
    if base_mb <= 0:
        return None
    return ((comm_per_round_mb - base_mb) / base_mb) * 100.0


def _scenario(config_name: str) -> str:
    if config_name == "f1_iid" or config_name == "f2_iid_phe":
        return "iid"
    if config_name == "f3_noniid" or config_name == "f4_noniid_phe":
        return "noniid_80_20"
    return "unknown"


def _write_report_md(comparison: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "# Harmonia FL Benchmark Report",
        "",
        "## Summary",
        "",
        comparison.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- C1 is the frozen centralized baseline from `outputs/segmentation/evaluation/evaluation_results.json`.",
        "- F1/F2 are single-seed FedAvg runs on IID patient-level partitions; F3/F4 use 80/20 Non-IID patient-level skew.",
        "- F2 and F4 use the TenSEAL CKKS PHE path and were verified with decrypt/evaluate roundtrip checks.",
        "- Communication totals are reconstructed from per-round upload/download byte counters in `round_log.jsonl`.",
        "- `phe_overhead_pct` compares total communication for completed runs; `phe_per_round_overhead_pct` compares per-round communication to avoid round-count effects.",
        "",
        "## Limitations",
        "",
        "- Single seeded run per F-config; no rerun variance is reported.",
        "- PHE uses a server-held CKKS secret key in this phase, so it measures overhead rather than a production trust boundary.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    generate_report()

