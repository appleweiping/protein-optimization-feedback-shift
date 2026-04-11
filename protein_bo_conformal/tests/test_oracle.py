"""Unit tests for Day 3 oracle behavior."""

from __future__ import annotations

import unittest

from data.data_loader import DatasetRecord
from data.oracle import Oracle, canonicalize_sequence


def _record(sequence: str, fitness: float) -> DatasetRecord:
    return DatasetRecord(
        sequence=sequence,
        fitness=fitness,
        wild_type=sequence,
        mutation_annotation="WT",
        position_index=(),
        mutation_count=0,
        benchmark="unit",
        task="oracle",
        assay_id="oracle",
        extra_metadata={},
    )


class OracleTests(unittest.TestCase):
    def test_canonicalize_sequence_strips_whitespace_and_uppercases(self) -> None:
        self.assertEqual(canonicalize_sequence("  ac d \n"), "ACD")

    def test_rejects_conflicting_duplicate_sequence(self) -> None:
        with self.assertRaises(ValueError):
            Oracle([_record("AAA", 1.0), _record("AAA", 2.0)])

    def test_query_is_canonical_and_tracks_newness(self) -> None:
        oracle = Oracle([_record("AAA", 1.0), _record("BBB", 2.0)])
        first = oracle.query(" aaa ")
        second = oracle.query("AAA")
        self.assertEqual(first.sequence, "AAA")
        self.assertEqual(first.fitness, 1.0)
        self.assertTrue(first.is_new)
        self.assertFalse(second.is_new)
        self.assertEqual(second.query_id, first.query_id + 1)

    def test_batch_query_preserves_input_order(self) -> None:
        oracle = Oracle([_record("AAA", 1.0), _record("BBB", 2.0), _record("CCC", 3.0)])
        results = oracle.batch_query(["CCC", "AAA", "BBB"], record_history=False)
        self.assertEqual([result.sequence for result in results], ["CCC", "AAA", "BBB"])
        self.assertEqual([result.fitness for result in results], [3.0, 1.0, 2.0])

    def test_query_history_can_be_disabled(self) -> None:
        oracle = Oracle([_record("AAA", 1.0)])
        oracle.query("AAA", record_history=False)
        self.assertEqual(len(oracle.query_history), 0)


if __name__ == "__main__":
    unittest.main()
