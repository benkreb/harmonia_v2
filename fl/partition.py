"""Patient-level FL partition generation for Harmonia Vision."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


Scenario = Literal["iid", "noniid_80_20"]
GENERATOR_VERSION = "2.0"
DEFAULT_SEED = 42
DEFAULT_N_CLIENTS = 2


def load_train_metadata(data_dir: Path) -> pd.DataFrame:
    metadata_path = data_dir / "metadata_train.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Training metadata not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    required = {"patient_id", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"metadata_train.csv missing required columns: {sorted(missing)}")
    return df.reset_index(drop=True)


def generate_iid_partition(
    metadata: pd.DataFrame,
    n_clients: int = DEFAULT_N_CLIENTS,
    seed: int = DEFAULT_SEED,
) -> dict:
    _require_two_clients(n_clients)
    rng = np.random.default_rng(seed)
    patients = metadata["patient_id"].drop_duplicates().to_numpy()
    rng.shuffle(patients)
    chunks = np.array_split(patients, n_clients)
    assignments = {
        f"client_{idx + 1}": chunk.astype(str).tolist()
        for idx, chunk in enumerate(chunks)
    }
    partition = _build_partition("iid", metadata, assignments, seed)
    validate_partition(partition, metadata)
    return partition


def generate_noniid_partition(
    metadata: pd.DataFrame,
    n_clients: int = DEFAULT_N_CLIENTS,
    seed: int = DEFAULT_SEED,
) -> dict:
    _require_two_clients(n_clients)
    rng = np.random.default_rng(seed)

    patient_labels = metadata.groupby("patient_id")["label"].max()
    benign_pool = patient_labels[patient_labels == 0].index.to_numpy()
    malignant_pool = patient_labels[patient_labels == 1].index.to_numpy()
    rng.shuffle(benign_pool)
    rng.shuffle(malignant_pool)

    target_client_1 = int(round(len(patient_labels) / 2))
    benign_client_1 = min(int(round(target_client_1 * 0.8)), len(benign_pool))
    malignant_client_1 = target_client_1 - benign_client_1
    if malignant_client_1 > len(malignant_pool):
        malignant_client_1 = len(malignant_pool)
        benign_client_1 = target_client_1 - malignant_client_1

    client_1 = np.concatenate([
        benign_pool[:benign_client_1],
        malignant_pool[:malignant_client_1],
    ])
    client_2 = np.concatenate([
        benign_pool[benign_client_1:],
        malignant_pool[malignant_client_1:],
    ])
    rng.shuffle(client_1)
    rng.shuffle(client_2)

    assignments = {
        "client_1": client_1.astype(str).tolist(),
        "client_2": client_2.astype(str).tolist(),
    }
    partition = _build_partition("noniid_80_20", metadata, assignments, seed)
    validate_partition(partition, metadata)
    return partition


def write_partition(partition: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(partition, f, indent=2)


def generate_partitions(
    data_dir: Path,
    output_dir: Path,
    scenario: Literal["all", "iid", "noniid_80_20", "noniid"],
    n_clients: int = DEFAULT_N_CLIENTS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Path]:
    metadata = load_train_metadata(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    if scenario in {"all", "iid"}:
        iid = generate_iid_partition(metadata, n_clients=n_clients, seed=seed)
        path = output_dir / "iid.json"
        write_partition(iid, path)
        written["iid"] = path

    if scenario in {"all", "noniid_80_20", "noniid"}:
        noniid = generate_noniid_partition(metadata, n_clients=n_clients, seed=seed)
        path = output_dir / "noniid_80_20.json"
        write_partition(noniid, path)
        written["noniid_80_20"] = path

    return written


def validate_partition(partition: dict, metadata: pd.DataFrame) -> None:
    clients = partition.get("clients", {})
    if len(clients) != partition.get("n_clients"):
        raise ValueError("Partition client count does not match n_clients")

    all_patient_sets = [set(client["patient_ids"]) for client in clients.values()]
    union = set().union(*all_patient_sets)
    expected = set(metadata["patient_id"].drop_duplicates().astype(str))

    if union != expected:
        missing = sorted(expected - union)[:5]
        extra = sorted(union - expected)[:5]
        raise ValueError(f"Partition patient union mismatch; missing={missing}, extra={extra}")

    seen: set[str] = set()
    for client_id, patient_ids in zip(clients, all_patient_sets):
        overlap = seen.intersection(patient_ids)
        if overlap:
            raise ValueError(f"{client_id} overlaps patients: {sorted(overlap)[:5]}")
        seen.update(patient_ids)

    total_samples = sum(client["n_samples"] for client in clients.values())
    if total_samples != len(metadata):
        raise ValueError(f"Sample count mismatch: {total_samples} != {len(metadata)}")

    scenario = partition["scenario"]
    skew = partition["stats"]["skew_metric"]
    if scenario == "iid" and skew >= 0.10:
        raise ValueError(f"IID skew too high: {skew:.4f}")
    if scenario == "noniid_80_20" and not 0.40 <= skew <= 0.80:
        raise ValueError(f"Non-IID skew outside [0.40, 0.80]: {skew:.4f}")


def _build_partition(
    scenario: Scenario,
    metadata: pd.DataFrame,
    assignments: dict[str, list[str]],
    seed: int,
) -> dict:
    clients = {}
    for client_id, patient_ids in assignments.items():
        patient_set = set(patient_ids)
        rows = metadata.index[metadata["patient_id"].astype(str).isin(patient_set)].to_numpy()
        rows = np.sort(rows).astype(int)
        labels = metadata.loc[rows, "label"].to_numpy()
        clients[client_id] = {
            "patient_ids": sorted(patient_ids),
            "n_patients": int(len(patient_ids)),
            "n_samples": int(len(rows)),
            "n_benign": int((labels == 0).sum()),
            "n_malignant": int((labels == 1).sum()),
            "row_indices": rows.tolist(),
        }

    benign_fractions = [
        _safe_fraction(client["n_benign"], client["n_samples"])
        for client in clients.values()
    ]
    samples = [client["n_samples"] for client in clients.values()]
    stats = {
        "samples_per_client": samples,
        "benign_fraction": benign_fractions,
        "skew_metric": float(max(benign_fractions) - min(benign_fractions)),
        "sample_imbalance": float(
            (max(samples) - min(samples)) / max(sum(samples), 1)
        ),
    }

    return {
        "scenario": scenario,
        "seed": seed,
        "n_clients": len(assignments),
        "n_total_patients": int(metadata["patient_id"].nunique()),
        "n_total_samples": int(len(metadata)),
        "clients": clients,
        "stats": stats,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator_version": GENERATOR_VERSION,
    }


def _safe_fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _require_two_clients(n_clients: int) -> None:
    if n_clients != 2:
        raise ValueError("This PRD phase supports exactly 2 FL clients")

