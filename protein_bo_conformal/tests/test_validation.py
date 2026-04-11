"""Unit tests for Day 3 validation helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from data.data_loader import DatasetBundle, DatasetRecord
from data.oracle import Oracle
from data.split import build_split
from data.validation import validate_oracle_consistency, validate_split_against_oracle


def _record(
    sequence: str,
    fitness: float,
    mutation_annotation: str,
    position_index: tuple[int, ...],
    mutation_count: int,
) -> DatasetRecord:
    return DatasetRecord(
        sequence=sequence,
        fitness=fitness,
        wild_type="AAAA",
        mutation_annotation=mutation_annotation,
        position_index=position_index,
        mutation_count=mutation_count,
        benchmark="unit",
        task="validation",
        assay_id="validation",
        extra_metadata={},
    )


class ValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = DatasetBundle(
            name="unit.validation",
            benchmark="unit",
            task="validation",
            source_path="memory",
            records=(
                _record("AAAA", 0.1, "WT", (), 0),
                _record("AAAB", 0.2, "A4B", (4,), 1),
                _record("AABA", 0.3, "A3B", (3,), 1),
                _record("ABAA", 0.4, "A2B", (2,), 1),
                _record("BAAA", 0.5, "A1B", (1,), 1),
            ),
            metadata={},
        )
        self.config = {
            "split_type": "low_resource",
            "split_seed": 7,
            "initial_train_size": 2,
            "candidate_pool_size": 3,
            "validation": {
                "enable_oracle_check": True,
                "enable_split_check": True,
                "validation_seed": 7,
                "repeat_query_sequences": 3,
                "repeat_query_rounds": 3,
                "max_validation_samples": 0,
            },
        }

    def test_validate_oracle_consistency_succeeds(self) -> None:
        oracle = Oracle(self.bundle.records)
        report = validate_oracle_consistency(self.bundle, oracle, self.config)
        self.assertTrue(report["successful"])
        self.assertEqual(report["dataset_record_count"], len(self.bundle.records))

    def test_validate_split_against_oracle_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            split_result = build_split(self.bundle, self.config, Path(tmp_dir))
        oracle = Oracle(self.bundle.records)
        report = validate_split_against_oracle(split_result, oracle, self.config)
        self.assertTrue(report["successful"])
        self.assertEqual(report["train_count"], 2)


if __name__ == "__main__":
    unittest.main()
