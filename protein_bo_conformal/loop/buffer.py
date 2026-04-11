"""Buffers for observations and candidate pools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from acquisition.registry import AcquisitionSelection
from data.data_loader import DatasetRecord
from data.oracle import OracleResult, canonicalize_sequence
from loop.state import LoopState


@dataclass(frozen=True)
class BufferUpdate:
    """Structured result of one candidate-to-observation buffer update."""

    next_state: LoopState
    selected_records: tuple[DatasetRecord, ...]
    selected_oracle_results: tuple[OracleResult, ...]
    removed_candidate_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "next_state": self.next_state.to_dict(),
            "selected_sequences": [record.sequence for record in self.selected_records],
            "selected_fitness": [float(result.fitness) for result in self.selected_oracle_results],
            "removed_candidate_indices": list(self.removed_candidate_indices),
        }


class ClosedLoopBuffer:
    """Manage pool updates after each oracle query batch."""

    def __init__(self, logger: Any | None = None) -> None:
        self.logger = logger

    def apply_selection(
        self,
        state: LoopState,
        selection: AcquisitionSelection,
        oracle_results: list[OracleResult] | tuple[OracleResult, ...],
    ) -> BufferUpdate:
        selected_indices = tuple(int(index) for index in selection.selected_indices)
        if not selected_indices:
            raise ValueError("Closed-loop updates require at least one selected candidate.")
        if len(selected_indices) != len(set(selected_indices)):
            raise ValueError("Selection contains duplicated candidate indices.")
        if len(selected_indices) != len(oracle_results):
            raise ValueError("Selection and oracle result batch sizes do not match.")

        candidate_pool = list(state.candidate_pool)
        selected_records: list[DatasetRecord] = []
        updated_sequences = set(state.queried_sequences)

        for offset, candidate_index in enumerate(selected_indices):
            if candidate_index < 0 or candidate_index >= len(candidate_pool):
                raise IndexError(f"Candidate index {candidate_index} is out of range.")
            record = candidate_pool[candidate_index]
            oracle_result = oracle_results[offset]
            canonical_sequence = canonicalize_sequence(record.sequence)
            if canonical_sequence in updated_sequences:
                raise ValueError(f"Sequence '{record.sequence}' was selected for query more than once.")
            if canonical_sequence != oracle_result.sequence:
                raise ValueError(
                    f"Oracle returned sequence '{oracle_result.sequence}' for selected record '{record.sequence}'."
                )
            if float(record.fitness) != float(oracle_result.fitness):
                raise ValueError(
                    f"Oracle mismatch for '{record.sequence}': {record.fitness} vs {oracle_result.fitness}."
                )
            updated_sequences.add(canonical_sequence)
            selected_records.append(record)

        removed_indices = set(selected_indices)
        remaining_candidates = tuple(
            record
            for index, record in enumerate(candidate_pool)
            if index not in removed_indices
        )
        observed_pool = tuple(list(state.observed_pool) + selected_records)
        next_best = max(
            [float(state.best_so_far), *[float(result.fitness) for result in oracle_results]]
        )
        next_state = LoopState(
            observed_pool=observed_pool,
            candidate_pool=remaining_candidates,
            round_index=state.round_index + 1,
            best_so_far=next_best,
            queried_sequences=frozenset(updated_sequences),
            initial_observed_count=state.initial_observed_count,
            total_queries=state.total_queries + len(selected_records),
            initial_best_so_far=state.initial_best_so_far,
        )

        if self.logger is not None:
            self.logger.info(
                "Buffer update applied: round=%s selected=%s observed=%s candidate=%s best_so_far=%.6f",
                state.round_index,
                len(selected_records),
                next_state.observed_count,
                next_state.candidate_count,
                next_state.best_so_far,
            )

        return BufferUpdate(
            next_state=next_state,
            selected_records=tuple(selected_records),
            selected_oracle_results=tuple(oracle_results),
            removed_candidate_indices=selected_indices,
        )
