"""Federated partition wrapper for CBIS-DDSM training data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Literal

import torch
from torch.utils.data import Dataset

from .dataset import CBISDDSMDataset
from .transforms import ComposeTransforms


class FLPartitionedDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        split: Literal["train"],
        partition_json_path: Path,
        client_id: str,
        transform: ComposeTransforms | Callable | None = None,
    ):
        if split != "train":
            raise ValueError("FLPartitionedDataset only partitions the train split")

        with open(partition_json_path, encoding="utf-8") as f:
            partition = json.load(f)
        clients = partition.get("clients", {})
        if client_id not in clients:
            raise KeyError(f"{client_id} not found in {partition_json_path}")

        client = clients[client_id]
        self._client_id = client_id
        self._row_indices = [int(idx) for idx in client["row_indices"]]
        self._n_benign = int(client["n_benign"])
        self._n_malignant = int(client["n_malignant"])
        self._base = CBISDDSMDataset(data_dir=data_dir, split=split, transform=transform)

    def __len__(self) -> int:
        return len(self._row_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self._base[self._row_indices[idx]]

    @property
    def n_samples(self) -> int:
        return len(self)

    @property
    def n_benign(self) -> int:
        return self._n_benign

    @property
    def n_malignant(self) -> int:
        return self._n_malignant

    @property
    def client_id(self) -> str:
        return self._client_id

