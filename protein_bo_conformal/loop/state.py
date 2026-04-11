"""State objects for closed-loop optimization."""

from __future__ import annotations

from dataclasses import dataclass

from data.data_loader import DatasetRecord
from data.oracle import canonicalize_sequence


@dataclass(frozen=True)
class LoopState:
    """Explicit state of the closed-loop optimization process."""

    observed_pool: tuple[DatasetRecord, ...]
    candidate_pool: tuple[DatasetRecord, ...]
    round_index: int
    best_so_far: float
    queried_sequences: frozenset[str]
    initial_observed_count: int
    total_queries: int
    initial_best_so_far: float

    @classmethod
    def initialize(
        cls,
        observed_pool: tuple[DatasetRecord, ...] | list[DatasetRecord],
        candidate_pool: tuple[DatasetRecord, ...] | list[DatasetRecord],
    ) -> "LoopState":
        observed_records = tuple(observed_pool)
        candidate_records = tuple(candidate_pool)
        if not observed_records:
            raise ValueError("Closed-loop state requires a non-empty observed pool.")
        queried_sequences = frozenset(
            canonicalize_sequence(record.sequence)
            for record in observed_records
        )
        best_so_far = max(float(record.fitness) for record in observed_records)
        return cls(
            observed_pool=observed_records,
            candidate_pool=candidate_records,
            round_index=0,
            best_so_far=best_so_far,
            queried_sequences=queried_sequences,
            initial_observed_count=len(observed_records),
            total_queries=0,
            initial_best_so_far=best_so_far,
        )

    @property
    def observed_count(self) -> int:
        return len(self.observed_pool)

    @property
    def candidate_count(self) -> int:
        return len(self.candidate_pool)

    @property
    def best_improvement(self) -> float:
        return float(self.best_so_far - self.initial_best_so_far)

    def to_dict(self) -> dict[str, object]:
        """Return a compact JSON-safe state summary."""
        return {
            "round_index": self.round_index,
            "observed_count": self.observed_count,
            "candidate_count": self.candidate_count,
            "best_so_far": self.best_so_far,
            "initial_observed_count": self.initial_observed_count,
            "total_queries": self.total_queries,
            "queried_sequence_count": len(self.queried_sequences),
            "best_improvement": self.best_improvement,
        }
