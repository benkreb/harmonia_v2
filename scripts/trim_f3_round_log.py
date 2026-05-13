"""Normalize the F3 round logs after an interrupted/stale run overlap.

The F3 log currently contains two interleaved sequential fallback runs. The
second run is the one that matches the saved best params and test metrics, so
we keep that run and back up the originals before rewriting report inputs.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


RUN_DIR = Path("outputs/fl/f3_noniid")
ROUND_LOG = RUN_DIR / "round_log.jsonl"
SERVER_EVAL = RUN_DIR / "server_eval.csv"


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _second_run(entries: list[dict]) -> list[dict]:
    second_start = next(
        idx
        for idx, row in enumerate(entries)
        if idx > 0 and int(row["round"]) == 1
    )
    keep = []
    expected_round = 1
    for row in entries[second_start:]:
        if int(row["round"]) == expected_round:
            keep.append(row)
            expected_round += 1
    return keep


def _write_round_log(entries: list[dict]) -> None:
    ROUND_LOG.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in entries) + "\n",
        encoding="utf-8",
    )


def _write_server_eval(entries: list[dict]) -> None:
    fieldnames = ["round", "val_loss", "val_dice", "val_iou", "val_sens", "val_spec", "val_prec"]
    with SERVER_EVAL.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in entries:
            metrics = row.get("server_eval", {})
            writer.writerow(
                {
                    "round": row["round"],
                    "val_loss": metrics.get("val_loss"),
                    "val_dice": metrics.get("dice"),
                    "val_iou": metrics.get("iou"),
                    "val_sens": metrics.get("sens"),
                    "val_spec": metrics.get("spec"),
                    "val_prec": metrics.get("prec"),
                }
            )


def main() -> None:
    for path in [ROUND_LOG, SERVER_EVAL]:
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(path, backup)

    entries = _load_jsonl(ROUND_LOG.with_suffix(ROUND_LOG.suffix + ".bak"))
    keep = _second_run(entries)

    _write_round_log(keep)
    _write_server_eval(keep)

    best = max(keep, key=lambda row: row.get("server_eval", {}).get("dice", 0.0))
    print(
        f"Kept {len(keep)} F3 rounds from second run; "
        f"best_round={best['round']} best_val_dice={best['server_eval']['dice']:.6f}"
    )


if __name__ == "__main__":
    main()
