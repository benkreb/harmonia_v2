"""TenSEAL CKKS context and parameter encryption utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class CryptoContext:
    context: object
    role: Literal["server", "client"]
    chunk_size: int = 8192

    @classmethod
    def create_server_context(
        cls,
        poly_modulus_degree: int = 8192,
        coeff_mod_bit_sizes: list[int] | None = None,
        global_scale_bits: int = 40,
        chunk_size: int = 8192,
    ) -> "CryptoContext":
        import tenseal as ts

        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes or [60, 40, 40, 60],
        )
        context.global_scale = 2 ** global_scale_bits
        return cls(context=context, role="server", chunk_size=chunk_size)

    def export_public_context_bytes(self) -> bytes:
        return self.context.serialize(
            save_public_key=True,
            save_secret_key=False,
            save_galois_keys=False,
            save_relin_keys=False,
        )

    @classmethod
    def from_public_bytes(cls, data: bytes, chunk_size: int = 8192) -> "CryptoContext":
        import tenseal as ts

        return cls(context=ts.context_from(data), role="client", chunk_size=chunk_size)

    def encrypt_parameters(self, parameters: list[np.ndarray], chunk_size: int | None = None) -> list[np.ndarray]:
        import tenseal as ts

        chunks = chunk_size or self.chunk_size
        encrypted: list[np.ndarray] = []
        for param in parameters:
            flat = param.astype(np.float64, copy=False).ravel()
            header = np.asarray(param.shape, dtype=np.int64).tobytes()
            for start in range(0, flat.size, chunks):
                vec = ts.ckks_vector(self.context, flat[start:start + chunks].tolist())
                encrypted.append(_bytes_to_array(header + b"|" + vec.serialize()))
        return encrypted

    def decrypt_parameters(self, encrypted: list[np.ndarray], shapes: list[tuple] | None = None) -> list[np.ndarray]:
        import tenseal as ts

        grouped: list[list[np.ndarray]] = []
        current_shape: tuple[int, ...] | None = None
        current_chunks: list[np.ndarray] = []
        for item in encrypted:
            blob = _array_to_bytes(item)
            header, payload = blob.split(b"|", 1)
            shape = tuple(np.frombuffer(header, dtype=np.int64).tolist())
            if current_shape is not None and shape != current_shape:
                grouped.append(current_chunks)
                current_chunks = []
            current_shape = shape
            current_chunks.append(item)
        if current_chunks:
            grouped.append(current_chunks)

        decrypted = []
        for group in grouped:
            values: list[float] = []
            shape = None
            for item in group:
                blob = _array_to_bytes(item)
                header, payload = blob.split(b"|", 1)
                shape = tuple(np.frombuffer(header, dtype=np.int64).tolist())
                values.extend(ts.ckks_vector_from(self.context, payload).decrypt())
            decrypted.append(np.asarray(values, dtype=np.float32).reshape(shape))
        return decrypted

    def homomorphic_weighted_sum(
        self,
        encrypted_per_client: list[list[np.ndarray]],
        weights: list[float],
        shapes: list[tuple] | None = None,
    ) -> list[np.ndarray]:
        import tenseal as ts

        del shapes
        if not encrypted_per_client:
            return []
        aggregated: list[np.ndarray] = []
        for chunks in zip(*encrypted_per_client):
            acc = None
            header = None
            for chunk, weight in zip(chunks, weights):
                blob = _array_to_bytes(chunk)
                header, payload = blob.split(b"|", 1)
                vec = ts.ckks_vector_from(self.context, payload) * float(weight)
                acc = vec if acc is None else acc + vec
            aggregated.append(_bytes_to_array(header + b"|" + acc.serialize()))
        return aggregated


def _bytes_to_array(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.uint8).copy()


def _array_to_bytes(array: np.ndarray | bytes) -> bytes:
    if isinstance(array, bytes):
        return array
    return np.asarray(array, dtype=np.uint8).tobytes()

