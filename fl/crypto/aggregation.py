"""Encrypted aggregation helpers."""

from __future__ import annotations

import numpy as np

from .tenseal_context import CryptoContext


def homomorphic_weighted_sum(
    crypto: CryptoContext,
    encrypted_per_client: list[list[np.ndarray]],
    weights: list[float],
) -> list[np.ndarray]:
    return crypto.homomorphic_weighted_sum(encrypted_per_client, weights)

