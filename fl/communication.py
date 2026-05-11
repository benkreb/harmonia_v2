"""Communication accounting for FL benchmark runs."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


class CommunicationTracker:
    def __init__(self, output_path: Path):
        self._output_path = output_path
        self._rows: list[dict] = []

    def record_upload(self, round_num: int, client_id: str, bytes_: int, encrypted: bool) -> None:
        self._record(round_num, client_id, "upload", bytes_, encrypted)

    def record_download(self, round_num: int, client_id: str, bytes_: int, encrypted: bool) -> None:
        self._record(round_num, client_id, "download", bytes_, encrypted)

    def round_total(self, round_num: int) -> dict:
        rows = [row for row in self._rows if row["round"] == round_num]
        return {
            "round": round_num,
            "upload_bytes": sum(row["bytes"] for row in rows if row["direction"] == "upload"),
            "download_bytes": sum(row["bytes"] for row in rows if row["direction"] == "download"),
            "total_bytes": sum(row["bytes"] for row in rows),
        }

    def flush(self) -> None:
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["round", "client_id", "direction", "bytes", "dtype", "encrypted"],
            )
            writer.writeheader()
            writer.writerows(self._rows)

    def _record(self, round_num: int, client_id: str, direction: str, bytes_: int, encrypted: bool) -> None:
        self._rows.append({
            "round": round_num,
            "client_id": client_id,
            "direction": direction,
            "bytes": int(bytes_),
            "dtype": "ciphertext" if encrypted else "float32",
            "encrypted": bool(encrypted),
        })


def parameters_bytes(params: list[np.ndarray]) -> int:
    return int(sum(param.size * param.itemsize for param in params))


def encrypted_bytes(encrypted_lists: list[list[bytes]] | list[bytes]) -> int:
    if not encrypted_lists:
        return 0
    if isinstance(encrypted_lists[0], bytes):
        return int(sum(len(blob) for blob in encrypted_lists))  # type: ignore[arg-type]
    return int(sum(len(blob) for sub in encrypted_lists for blob in sub))  # type: ignore[union-attr]

