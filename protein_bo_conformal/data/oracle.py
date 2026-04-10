"""Oracle access and scoring helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from data.data_loader import DatasetRecord


@dataclass(frozen=True)
class OracleResult:
    """Single oracle lookup result."""

    sequence: str
    fitness: float


class Oracle:
    """Immutable sequence-to-fitness lookup used by the offline closed loop."""

    def __init__(self, records: tuple[DatasetRecord, ...] | list[DatasetRecord], logger: Any | None = None) -> None:
        self._mapping: dict[str, float] = {}
        self._logger = logger
        for record in records:
            existing = self._mapping.get(record.sequence)
            if existing is not None and existing != record.fitness:
                raise ValueError(f"Conflicting oracle value for sequence '{record.sequence}'.")
            self._mapping[record.sequence] = record.fitness
        self._dataset_hash = self._compute_dataset_hash()

    def _compute_dataset_hash(self) -> str:
        payload = "|".join(f"{sequence}:{fitness}" for sequence, fitness in sorted(self._mapping.items()))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def dataset_hash(self) -> str:
        """Return the hash of the immutable sequence-to-fitness table."""
        return self._dataset_hash

    def query(self, sequence: str) -> OracleResult:
        """Query a single sequence against the immutable oracle table."""
        key = sequence.strip().upper()
        if key not in self._mapping:
            raise KeyError(f"Sequence '{sequence}' not found in oracle.")
        result = OracleResult(sequence=key, fitness=self._mapping[key])
        if self._logger is not None:
            self._logger.info("Oracle query: %s -> %s", result.sequence, result.fitness)
        return result

    def batch_query(self, sequences: list[str] | tuple[str, ...]) -> list[OracleResult]:
        """Query multiple sequences and preserve input ordering."""
        return [self.query(sequence) for sequence in sequences]
