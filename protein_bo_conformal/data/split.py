"""Dataset splitting utilities."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data.data_loader import DatasetBundle, DatasetRecord, summarize_records


@dataclass(frozen=True)
class SplitResult:
    """Structured split result for closed-loop environment construction."""

    split_name: str
    split_id: str
    train_records: tuple[DatasetRecord, ...]
    candidate_records: tuple[DatasetRecord, ...]
    statistics: dict[str, Any]
    cache_paths: dict[str, str]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _hamming_distance(seq_a: str, seq_b: str) -> int:
    shared = sum(char_a != char_b for char_a, char_b in zip(seq_a, seq_b))
    return shared + abs(len(seq_a) - len(seq_b))


def _min_cross_distances(train_records: list[DatasetRecord], candidate_records: list[DatasetRecord]) -> list[int]:
    """Compute each candidate's nearest train-set Hamming distance."""
    distances: list[int] = []
    if not train_records:
        return distances
    for candidate in candidate_records:
        nearest = min(_hamming_distance(candidate.sequence, train.sequence) for train in train_records)
        distances.append(nearest)
    return distances


def _alphabet(records: list[DatasetRecord]) -> list[str]:
    chars = sorted({char for record in records for char in record.sequence})
    return chars + ["_"]


def _onehot_centroid(
    records: list[DatasetRecord],
    alphabet: list[str],
    max_length: int,
) -> list[float]:
    if not records:
        return []
    char_index = {char: index for index, char in enumerate(alphabet)}
    width = len(alphabet)
    centroid = [0.0] * (max_length * width)
    for record in records:
        sequence = record.sequence + ("_" * (max_length - len(record.sequence)))
        for position, char in enumerate(sequence):
            centroid[position * width + char_index[char]] += 1.0
    scale = float(len(records))
    return [value / scale for value in centroid]


def _l2_distance(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    return sum((left - right) ** 2 for left, right in zip(vec_a, vec_b)) ** 0.5


def _limit_records(
    records: list[DatasetRecord],
    limit: int | None,
    rng: random.Random,
) -> list[DatasetRecord]:
    """Apply an optional deterministic size limit to a record pool."""
    if limit is None or limit <= 0 or len(records) <= limit:
        return list(records)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    selected = sorted(indices[:limit])
    return [records[index] for index in selected]


def _split_low_resource(
    records: list[DatasetRecord],
    config: dict[str, Any],
    rng: random.Random,
) -> tuple[list[DatasetRecord], list[DatasetRecord], dict[str, Any]]:
    shuffled = list(records)
    rng.shuffle(shuffled)
    train_size = min(int(config.get("initial_train_size", 8)), len(shuffled))
    train_records = shuffled[:train_size]
    candidate_records = shuffled[train_size:]
    return train_records, candidate_records, {"strategy": "low_resource"}


def _split_mutation_extrapolation(
    records: list[DatasetRecord],
    config: dict[str, Any],
    rng: random.Random,
) -> tuple[list[DatasetRecord], list[DatasetRecord], dict[str, Any]]:
    max_train_mutations = int(config.get("mutation_train_max", 1))
    train_records = [record for record in records if record.mutation_count <= max_train_mutations]
    candidate_records = [record for record in records if record.mutation_count > max_train_mutations]
    train_records = _limit_records(train_records, int(config.get("initial_train_size", 0)), rng)
    candidate_records = _limit_records(candidate_records, int(config.get("candidate_pool_size", 0)), rng)
    return train_records, candidate_records, {"strategy": "mutation_extrapolation", "mutation_train_max": max_train_mutations}


def _split_position_extrapolation(
    records: list[DatasetRecord],
    config: dict[str, Any],
    rng: random.Random,
) -> tuple[list[DatasetRecord], list[DatasetRecord], dict[str, Any]]:
    positions = sorted({position for record in records for position in record.position_index})
    if not positions:
        raise ValueError("Position extrapolation requires at least one mutated position.")
    holdout_count = max(1, min(int(config.get("position_holdout_count", 1)), len(positions)))
    holdout_positions = sorted(rng.sample(positions, holdout_count))
    holdout_set = set(holdout_positions)
    train_records = [record for record in records if not holdout_set.intersection(record.position_index)]
    candidate_records = [record for record in records if holdout_set.intersection(record.position_index)]
    train_records = _limit_records(train_records, int(config.get("initial_train_size", 0)), rng)
    candidate_records = _limit_records(candidate_records, int(config.get("candidate_pool_size", 0)), rng)
    return train_records, candidate_records, {"strategy": "position_extrapolation", "holdout_positions": holdout_positions}


def _split_fitness_extrapolation(
    records: list[DatasetRecord],
    config: dict[str, Any],
    rng: random.Random,
) -> tuple[list[DatasetRecord], list[DatasetRecord], dict[str, Any]]:
    quantile = float(config.get("fitness_holdout_quantile", 0.75))
    if not 0.0 < quantile < 1.0:
        raise ValueError("fitness_holdout_quantile must be between 0 and 1.")
    ordered = sorted(record.fitness for record in records)
    cutoff_index = min(len(ordered) - 1, max(0, int(len(ordered) * quantile)))
    cutoff = ordered[cutoff_index]
    train_records = [record for record in records if record.fitness < cutoff]
    candidate_records = [record for record in records if record.fitness >= cutoff]
    train_records = _limit_records(train_records, int(config.get("initial_train_size", 0)), rng)
    candidate_records = _limit_records(candidate_records, int(config.get("candidate_pool_size", 0)), rng)
    return train_records, candidate_records, {"strategy": "fitness_extrapolation", "fitness_cutoff": cutoff}


def _split_predefined(
    records: list[DatasetRecord],
    config: dict[str, Any],
    rng: random.Random,
) -> tuple[list[DatasetRecord], list[DatasetRecord], dict[str, Any]]:
    train_labels = {
        label.strip().lower()
        for label in str(config.get("predefined_train_labels", "train")).split(",")
        if label.strip()
    }
    candidate_labels = {
        label.strip().lower()
        for label in str(config.get("predefined_candidate_labels", "test,validation,valid")).split(",")
        if label.strip()
    }

    labeled_records = [
        (record, str(record.extra_metadata.get("set", "")).strip().lower())
        for record in records
    ]
    train_records = [record for record, label in labeled_records if label in train_labels]
    candidate_records = [record for record, label in labeled_records if label in candidate_labels]
    if not candidate_records:
        candidate_records = [
            record
            for record, label in labeled_records
            if label and label not in train_labels
        ]

    train_records = _limit_records(train_records, int(config.get("initial_train_size", 0)), rng)
    candidate_records = _limit_records(candidate_records, int(config.get("candidate_pool_size", 0)), rng)
    observed_labels = sorted({label for _, label in labeled_records if label})
    return train_records, candidate_records, {
        "strategy": "predefined",
        "train_labels": sorted(train_labels),
        "candidate_labels": sorted(candidate_labels),
        "observed_labels": observed_labels,
    }


def _build_split_statistics(
    train_records: list[DatasetRecord],
    candidate_records: list[DatasetRecord],
) -> dict[str, Any]:
    """Build quantitative diagnostics for a realized split."""
    cross_distances = _min_cross_distances(train_records, candidate_records)
    support_overlap = sum(distance <= 1 for distance in cross_distances) / len(cross_distances) if cross_distances else 0.0
    alphabet = _alphabet(train_records + candidate_records)
    max_length = max(len(record.sequence) for record in train_records + candidate_records)
    train_centroid = _onehot_centroid(train_records, alphabet, max_length)
    candidate_centroid = _onehot_centroid(candidate_records, alphabet, max_length)
    return {
        "train_summary": summarize_records(train_records),
        "candidate_summary": summarize_records(candidate_records),
        "cross_distance": {
            "candidate_min_distance_mean": _mean([float(value) for value in cross_distances]),
            "candidate_min_distance_max": max(cross_distances) if cross_distances else 0,
            "candidate_min_distance_min": min(cross_distances) if cross_distances else 0,
            "sample_count": len(cross_distances),
        },
        "embedding_distance": {
            "method": "padded_onehot_centroid_l2",
            "alphabet_size": len(alphabet),
            "shared_max_length": max_length,
            "train_candidate_centroid_l2": _l2_distance(train_centroid, candidate_centroid),
        },
        "support_overlap_proxy": support_overlap,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_split(
    bundle: DatasetBundle,
    config: dict[str, Any],
    processed_dir: Path,
    logger: Any | None = None,
) -> SplitResult:
    """Create a split that makes distribution shift explicit and cache its artifacts."""
    split_type = str(config.get("split_type", "low_resource"))
    split_seed = int(config.get("split_seed", 7))
    rng = random.Random(split_seed)
    records = list(bundle.records)

    if split_type == "low_resource":
        train_records, candidate_records, split_metadata = _split_low_resource(records, config, rng)
    elif split_type == "mutation_extrapolation":
        train_records, candidate_records, split_metadata = _split_mutation_extrapolation(records, config, rng)
    elif split_type == "position_extrapolation":
        train_records, candidate_records, split_metadata = _split_position_extrapolation(records, config, rng)
    elif split_type == "fitness_extrapolation":
        train_records, candidate_records, split_metadata = _split_fitness_extrapolation(records, config, rng)
    elif split_type == "predefined":
        train_records, candidate_records, split_metadata = _split_predefined(records, config, rng)
    else:
        raise ValueError(f"Unsupported split_type '{split_type}'.")

    if not train_records:
        raise ValueError(f"Split '{split_type}' produced an empty train pool.")
    if not candidate_records:
        raise ValueError(f"Split '{split_type}' produced an empty candidate pool.")

    stats = _build_split_statistics(train_records, candidate_records)
    stats["split_metadata"] = split_metadata
    stats["dataset_name"] = bundle.name
    stats["split_seed"] = split_seed

    split_id = f"{bundle.name.replace('.', '_')}_{split_type}_seed{split_seed}"
    split_dir = processed_dir / "splits"
    metadata_dir = processed_dir / "metadata"
    split_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    split_payload = {
        "split_id": split_id,
        "train_sequences": [record.sequence for record in train_records],
        "candidate_sequences": [record.sequence for record in candidate_records],
    }
    split_cache_path = split_dir / f"{split_id}.json"
    metadata_cache_path = metadata_dir / f"{split_id}_summary.json"
    _write_json(split_cache_path, split_payload)
    _write_json(metadata_cache_path, stats)

    if logger is not None:
        logger.info(
            "Built split '%s' with %s train records and %s candidate records.",
            split_id,
            len(train_records),
            len(candidate_records),
        )

    return SplitResult(
        split_name=split_type,
        split_id=split_id,
        train_records=tuple(train_records),
        candidate_records=tuple(candidate_records),
        statistics=stats,
        cache_paths={
            "split_cache": str(split_cache_path),
            "metadata_cache": str(metadata_cache_path),
        },
    )
