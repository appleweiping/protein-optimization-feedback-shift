"""Tests for closed-loop state, buffer, and stopping helpers."""

from __future__ import annotations

import unittest

from acquisition.registry import AcquisitionSelection
from data.data_loader import DatasetRecord
from data.oracle import OracleResult
from loop.buffer import ClosedLoopBuffer
from loop.state import LoopState
from loop.stopping import LoopStopping


def _record(sequence: str, fitness: float) -> DatasetRecord:
    return DatasetRecord(
        sequence=sequence,
        fitness=fitness,
        wild_type="AAAA",
        mutation_annotation="WT" if sequence == "AAAA" else "A1B",
        position_index=(),
        mutation_count=0 if sequence == "AAAA" else 1,
        benchmark="flip",
        task="toy",
        assay_id="toy",
        extra_metadata={},
    )


class LoopStateTest(unittest.TestCase):
    def test_initialize_sets_best_and_query_tracking(self) -> None:
        state = LoopState.initialize(
            observed_pool=[_record("AAAA", 0.2), _record("AAAB", 0.8)],
            candidate_pool=[_record("AABA", 0.5)],
        )
        self.assertEqual(state.observed_count, 2)
        self.assertEqual(state.candidate_count, 1)
        self.assertAlmostEqual(state.best_so_far, 0.8)
        self.assertIn("AAAA", state.queried_sequences)
        self.assertIn("AAAB", state.queried_sequences)


class BufferUpdateTest(unittest.TestCase):
    def test_apply_selection_moves_records_and_updates_best(self) -> None:
        state = LoopState.initialize(
            observed_pool=[_record("AAAA", 0.2)],
            candidate_pool=[_record("AAAB", 0.8), _record("AABA", 0.5)],
        )
        selection = AcquisitionSelection(
            name="greedy",
            batch_size=1,
            selected_indices=(0,),
            selected_scores=(0.9,),
            selected_details=({"index": 0, "score": 0.9, "mean": 0.9, "sigma": 0.1},),
            score_summary={"max_score": 0.9, "min_score": 0.1, "mean_score": 0.5, "std_score": 0.4},
        )
        oracle_results = [OracleResult(sequence="AAAB", fitness=0.8, query_id=1, is_new=True)]
        update = ClosedLoopBuffer().apply_selection(state, selection, oracle_results)
        self.assertEqual(update.next_state.observed_count, 2)
        self.assertEqual(update.next_state.candidate_count, 1)
        self.assertAlmostEqual(update.next_state.best_so_far, 0.8)
        self.assertEqual(update.next_state.total_queries, 1)
        self.assertEqual(update.next_state.round_index, 1)

    def test_apply_selection_rejects_duplicate_queries(self) -> None:
        state = LoopState.initialize(
            observed_pool=[_record("AAAA", 0.2)],
            candidate_pool=[_record("AAAB", 0.8)],
        )
        selection = AcquisitionSelection(
            name="greedy",
            batch_size=1,
            selected_indices=(0,),
            selected_scores=(0.9,),
            selected_details=({"index": 0, "score": 0.9, "mean": 0.9, "sigma": 0.1},),
            score_summary={"max_score": 0.9, "min_score": 0.9, "mean_score": 0.9, "std_score": 0.0},
        )
        oracle_results = [OracleResult(sequence="AAAB", fitness=0.8, query_id=1, is_new=True)]
        buffer = ClosedLoopBuffer()
        update = buffer.apply_selection(state, selection, oracle_results)
        repeated_state = LoopState(
            observed_pool=update.next_state.observed_pool,
            candidate_pool=(_record("AAAB", 0.8),),
            round_index=update.next_state.round_index,
            best_so_far=update.next_state.best_so_far,
            queried_sequences=update.next_state.queried_sequences,
            initial_observed_count=update.next_state.initial_observed_count,
            total_queries=update.next_state.total_queries,
            initial_best_so_far=update.next_state.initial_best_so_far,
        )
        with self.assertRaises(ValueError):
            buffer.apply_selection(repeated_state, selection, oracle_results)


class LoopStoppingTest(unittest.TestCase):
    def test_decide_respects_round_and_budget_limits(self) -> None:
        stopping = LoopStopping(total_rounds=2, total_budget=3, query_batch_size=2)
        state = LoopState.initialize(
            observed_pool=[_record("AAAA", 0.2)],
            candidate_pool=[_record("AAAB", 0.8), _record("AABA", 0.5)],
        )
        decision = stopping.decide(state)
        self.assertFalse(decision.stop)
        self.assertEqual(decision.next_batch_size, 2)

        state = LoopState(
            observed_pool=state.observed_pool + (_record("AAAB", 0.8), _record("AABA", 0.5)),
            candidate_pool=(),
            round_index=1,
            best_so_far=0.8,
            queried_sequences=frozenset({"AAAA", "AAAB", "AABA"}),
            initial_observed_count=1,
            total_queries=2,
            initial_best_so_far=0.2,
        )
        decision = stopping.decide(state)
        self.assertTrue(decision.stop)
        self.assertEqual(decision.reason, "candidate_pool_exhausted")


if __name__ == "__main__":
    unittest.main()
