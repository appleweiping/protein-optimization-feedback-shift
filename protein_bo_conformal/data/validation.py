"""Validation helpers for dataset splits and immutable oracle behavior."""

from __future__ import annotations

import random
from typing import Any

from data.data_loader import DatasetBundle
from data.oracle import Oracle, canonicalize_sequence
from data.split import SplitResult


def validate_oracle_consistency(
    dataset: DatasetBundle,
    oracle: Oracle,
    config: dict[str, Any],
    logger: Any | None = None,
) -> dict[str, Any]:
    """Run the Day 3 oracle validation suite against a loaded dataset."""
    validation_config = dict(config.get("validation", {}))
    repeat_sample_count = min(
        int(validation_config.get("repeat_query_sequences", 16)),
        len(dataset.records),
    )
    repeat_rounds = max(2, int(validation_config.get("repeat_query_rounds", 3)))
    validation_seed = int(validation_config.get("validation_seed", config.get("split_seed", 7)))
    rng = random.Random(validation_seed)
    oracle.check_consistency(deep=True)

    full_scan_mismatches: list[dict[str, Any]] = []
    for record in dataset.records:
        result = oracle.query(
            record.sequence,
            source="full_scan",
            log_query=False,
            record_history=False,
        )
        if result.fitness != record.fitness:
            full_scan_mismatches.append(
                {
                    "sequence": record.sequence,
                    "expected": record.fitness,
                    "observed": result.fitness,
                }
            )
            if len(full_scan_mismatches) >= 10:
                break

    sampled_records = rng.sample(list(dataset.records), repeat_sample_count) if repeat_sample_count else []

    repeat_mismatches: list[dict[str, Any]] = []
    for record in sampled_records:
        observed = [
            oracle.query(
                record.sequence,
                source="repeat_query",
                log_query=False,
                record_history=False,
            ).fitness
            for _ in range(repeat_rounds)
        ]
        if any(value != record.fitness for value in observed):
            repeat_mismatches.append(
                {
                    "sequence": record.sequence,
                    "expected": record.fitness,
                    "observed": observed,
                }
            )

    batch_order_mismatches: list[dict[str, Any]] = []
    if sampled_records:
        original_sequences = [record.sequence for record in sampled_records]
        reversed_sequences = list(reversed(original_sequences))
        original_results = oracle.batch_query(
            original_sequences,
            source="batch_order_original",
            log_query=False,
            record_history=False,
        )
        reversed_results = oracle.batch_query(
            reversed_sequences,
            source="batch_order_reversed",
            log_query=False,
            record_history=False,
        )
        original_mapping = {result.sequence: result.fitness for result in original_results}
        reversed_mapping = {result.sequence: result.fitness for result in reversed_results}
        for sequence in original_mapping:
            if reversed_mapping.get(sequence) != original_mapping[sequence]:
                batch_order_mismatches.append(
                    {
                        "sequence": sequence,
                        "original": original_mapping[sequence],
                        "reversed": reversed_mapping.get(sequence),
                    }
                )

    canonicalization_examples: list[dict[str, Any]] = []
    canonicalization_failures: list[dict[str, Any]] = []
    for record in sampled_records[: min(3, len(sampled_records))]:
        variant = f"  {record.sequence.lower()}  "
        try:
            result = oracle.query(
                variant,
                source="canonicalization_check",
                log_query=False,
                record_history=False,
            )
            canonicalization_examples.append(
                {
                    "input": variant,
                    "canonical_sequence": canonicalize_sequence(variant),
                    "fitness": result.fitness,
                }
            )
            if result.fitness != record.fitness:
                canonicalization_failures.append(
                    {
                        "input": variant,
                        "expected": record.fitness,
                        "observed": result.fitness,
                    }
                )
        except Exception as exc:
            canonicalization_failures.append(
                {
                    "input": variant,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    report = {
        "dataset_hash": oracle.dataset_hash,
        "dataset_record_count": len(dataset.records),
        "full_scan": {
            "checked_count": len(dataset.records),
            "successful": not full_scan_mismatches,
            "mismatches": full_scan_mismatches,
        },
        "repeat_query_stability": {
            "sample_count": repeat_sample_count,
            "repeat_rounds": repeat_rounds,
            "successful": not repeat_mismatches,
            "mismatches": repeat_mismatches,
        },
        "batch_order_invariance": {
            "sample_count": repeat_sample_count,
            "successful": not batch_order_mismatches,
            "mismatches": batch_order_mismatches,
        },
        "canonicalization_check": {
            "sample_count": len(canonicalization_examples),
            "successful": not canonicalization_failures,
            "examples": canonicalization_examples,
            "failures": canonicalization_failures,
        },
        "query_history_size": len(oracle.query_history),
    }
    report["successful"] = all(
        (
            report["full_scan"]["successful"],
            report["repeat_query_stability"]["successful"],
            report["batch_order_invariance"]["successful"],
            report["canonicalization_check"]["successful"],
        )
    )
    oracle.check_consistency(deep=True)

    if logger is not None:
        logger.info(
            "Oracle validation finished for '%s': full_scan=%s, repeat=%s, order=%s, canonical=%s",
            dataset.name,
            report["full_scan"]["successful"],
            report["repeat_query_stability"]["successful"],
            report["batch_order_invariance"]["successful"],
            report["canonicalization_check"]["successful"],
        )
    if not report["successful"]:
        raise ValueError(f"Oracle consistency validation failed for '{dataset.name}'.")
    return report


def validate_split_against_oracle(
    split_result: SplitResult,
    oracle: Oracle,
    config: dict[str, Any],
    logger: Any | None = None,
) -> dict[str, Any]:
    """Validate that every retained split sequence is queryable by the oracle."""
    validation_config = dict(config.get("validation", {}))
    max_validation_samples = int(validation_config.get("max_validation_samples", 0))

    train_records = list(split_result.train_records)
    candidate_records = list(split_result.candidate_records)
    combined_records = train_records + candidate_records
    if max_validation_samples > 0:
        combined_records = combined_records[:max_validation_samples]

    missing_sequences: list[str] = []
    for record in combined_records:
        try:
            oracle.query(
                record.sequence,
                source="split_validation",
                log_query=False,
                record_history=False,
            )
        except KeyError:
            missing_sequences.append(record.sequence)

    report = {
        "checked_count": len(combined_records),
        "successful": not missing_sequences,
        "missing_sequences": missing_sequences,
        "train_count": len(train_records),
        "candidate_count": len(candidate_records),
    }
    if logger is not None:
        logger.info(
            "Split oracle validation for '%s': checked=%s missing=%s",
            split_result.split_id,
            report["checked_count"],
            len(missing_sequences),
        )
    if missing_sequences:
        raise ValueError(f"Split '{split_result.split_id}' contains sequences missing from the oracle.")
    return report
