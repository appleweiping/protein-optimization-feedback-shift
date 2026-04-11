"""Oracle access and scoring helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from data.data_loader import DatasetRecord


def canonicalize_sequence(sequence: str) -> str:
    """Normalize user-provided sequences into a canonical oracle key."""
    normalized = "".join(sequence.split()).upper()
    if not normalized:
        raise ValueError("Oracle queries must provide a non-empty sequence.")
    return normalized


@dataclass(frozen=True)
class OracleResult:
    """Single oracle lookup result with audit metadata."""

    sequence: str
    fitness: float
    query_id: int
    is_new: bool


class Oracle:
    """Immutable sequence-to-fitness lookup used by the offline closed loop."""

    def __init__(
        self,
        records: tuple[DatasetRecord, ...] | list[DatasetRecord],
        logger: Any | None = None,
        enable_query_logging: bool = False,
    ) -> None:
        mapping: dict[str, float] = {}
        self._logger = logger
        self._enable_query_logging = enable_query_logging
        self._query_counter = 0
        self._seen_sequences: set[str] = set()
        self._query_history: list[dict[str, Any]] = []

        for record in records:
            key = canonicalize_sequence(record.sequence)
            existing = mapping.get(key)
            if existing is not None and existing != record.fitness:
                raise ValueError(f"Conflicting oracle value for sequence '{record.sequence}'.")
            mapping[key] = record.fitness
        self._mapping = MappingProxyType(mapping)
        self._mapping_size = len(mapping)
        self._dataset_hash = self._compute_dataset_hash()
        self.check_consistency(deep=True)

    def _compute_dataset_hash(self) -> str:
        payload = "|".join(f"{sequence}:{fitness}" for sequence, fitness in sorted(self._mapping.items()))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def dataset_hash(self) -> str:
        """Return the hash of the immutable sequence-to-fitness table."""
        return self._dataset_hash

    @property
    def query_history(self) -> tuple[dict[str, Any], ...]:
        """Expose an immutable snapshot of prior query metadata."""
        return tuple(dict(entry) for entry in self._query_history)

    def check_consistency(self, deep: bool = False) -> None:
        """Assert that the immutable table has not drifted in memory."""
        if len(self._mapping) != self._mapping_size:
            raise RuntimeError("Oracle mapping size drift detected after initialization.")
        if deep and self._compute_dataset_hash() != self._dataset_hash:
            raise RuntimeError("Oracle mapping drift detected after initialization.")

    def query(
        self,
        sequence: str,
        source: str = "query",
        log_query: bool | None = None,
        record_history: bool = True,
    ) -> OracleResult:
        """Query a single sequence against the immutable oracle table."""
        self.check_consistency()
        key = canonicalize_sequence(sequence)
        if key not in self._mapping:
            raise KeyError(f"Sequence '{sequence}' not found in oracle.")

        self._query_counter += 1
        is_new = key not in self._seen_sequences
        self._seen_sequences.add(key)
        result = OracleResult(
            sequence=key,
            fitness=self._mapping[key],
            query_id=self._query_counter,
            is_new=is_new,
        )

        if record_history:
            history_entry = {
                "query_id": result.query_id,
                "sequence": result.sequence,
                "fitness": result.fitness,
                "is_new": result.is_new,
                "source": source,
            }
            self._query_history.append(history_entry)

        should_log = self._enable_query_logging if log_query is None else log_query
        if self._logger is not None and should_log:
            self._logger.info(
                "Oracle query [%s] (%s, new=%s): %s -> %s",
                result.query_id,
                source,
                result.is_new,
                result.sequence,
                result.fitness,
            )
        return result

    def batch_query(
        self,
        sequences: list[str] | tuple[str, ...],
        source: str = "batch_query",
        log_query: bool | None = None,
        record_history: bool = True,
    ) -> list[OracleResult]:
        """Query multiple sequences and preserve input ordering."""
        return [
            self.query(
                sequence,
                source=source,
                log_query=log_query,
                record_history=record_history,
            )
            for sequence in sequences
        ]
