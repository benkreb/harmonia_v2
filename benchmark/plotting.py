"""Plot generation for FL benchmark reports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def generate_plots(comparison: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    _bar_plot(comparison, output_dir / "dice_by_config.png", "dice_mean", "Dice")
    _bar_plot(comparison, output_dir / "iou_by_config.png", "iou_mean", "IoU")
    _bar_plot(comparison, output_dir / "communication_total.png", "total_comm_mb", "Total Communication (MB)")
    if "phe_overhead_pct" in comparison:
        _bar_plot(comparison, output_dir / "phe_overhead.png", "phe_overhead_pct", "PHE Overhead (%)")

    fig, ax = plt.subplots(figsize=(8, 4))
    for config_dir in Path("outputs/fl").glob("*/server_eval.csv"):
        df = pd.read_csv(config_dir)
        name = config_dir.parent.name
        if "val_dice" in df:
            ax.plot(df["round"], df["val_dice"], label=name)
    ax.set_xlabel("Round")
    ax.set_ylabel("Validation Dice")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "convergence_overlay.png", dpi=140)
    plt.close(fig)


def _bar_plot(df: pd.DataFrame, path: Path, column: str, ylabel: str) -> None:
    if column not in df:
        return
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df["config"], df[column].fillna(0.0))
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)

