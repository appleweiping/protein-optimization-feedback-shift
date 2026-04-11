"""Data loading utilities for benchmark datasets."""

from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from data.dataset_registry import DatasetSpec


VALID_AA_PATTERN = re.compile(r"^[A-Z]+$")
MUTATION_SPLIT_PATTERN = re.compile(r"[;,:\s|]+")
MUTATION_POSITION_PATTERN = re.compile(r"(\d+)")
MUTATION_TOKEN_PATTERN = re.compile(r"^([A-Z])(\d+)([A-Z])$")


@dataclass(frozen=True)
class DatasetRecord:
    """Standardized protein sequence record used by the offline environment."""

    sequence: str
    fitness: float
    wild_type: str
    mutation_annotation: str
    position_index: tuple[int, ...]
    mutation_count: int
    benchmark: str
    task: str
    assay_id: str
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the record to a JSON-serializable dictionary."""
        return {
            "sequence": self.sequence,
            "fitness": self.fitness,
            "wild_type": self.wild_type,
            "mutation_annotation": self.mutation_annotation,
            "position_index": list(self.position_index),
            "mutation_count": self.mutation_count,
            "benchmark": self.benchmark,
            "task": self.task,
            "assay_id": self.assay_id,
            "extra_metadata": dict(self.extra_metadata),
        }


@dataclass(frozen=True)
class DatasetBundle:
    """Normalized dataset bundle containing records and summary metadata."""

    name: str
    benchmark: str
    task: str
    source_path: str
    records: tuple[DatasetRecord, ...]
    metadata: dict[str, Any]

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a compact dataset summary."""
        return {
            "name": self.name,
            "benchmark": self.benchmark,
            "task": self.task,
            "source_path": self.source_path,
            "record_count": len(self.records),
            "metadata": dict(self.metadata),
        }


def normalize_sequence(sequence: str) -> str:
    """Normalize a protein sequence and validate that it only contains letters."""
    normalized = sequence.strip().upper()
    if not normalized or not VALID_AA_PATTERN.match(normalized):
        raise ValueError(f"Invalid sequence '{sequence}'.")
    return normalized


def normalize_mutation_annotation(annotation: str) -> str:
    """Normalize mutation annotation strings to a stable semicolon-separated format."""
    stripped = annotation.strip()
    if not stripped or stripped.lower() in {"wt", "wildtype", "wild_type", "none"}:
        return "WT"
    tokens = [token for token in MUTATION_SPLIT_PATTERN.split(stripped.upper()) if token]
    return ";".join(tokens) if tokens else "WT"


def infer_mutation_annotation(sequence: str, wild_type: str) -> str:
    """Infer mutation annotation by comparing a sequence against the wild type."""
    tokens: list[str] = []
    for index, (wild, current) in enumerate(zip(wild_type, sequence), start=1):
        if wild != current:
            tokens.append(f"{wild}{index}{current}")
    return ";".join(tokens) if tokens else "WT"


def extract_positions(annotation: str) -> tuple[int, ...]:
    """Extract sorted mutated positions from a normalized mutation annotation."""
    if annotation == "WT":
        return ()
    positions = sorted(
        {
            int(match.group(1))
            for token in annotation.split(";")
            for match in [MUTATION_POSITION_PATTERN.search(token)]
            if match
        }
    )
    return tuple(positions)


def _detect_delimiter(path: Path) -> str:
    """Detect the delimiter for a CSV/TSV style file."""
    if path.suffix.lower() == ".tsv":
        return "\t"
    sample = path.read_text(encoding="utf-8").splitlines()[0]
    if "\t" in sample:
        return "\t"
    return ","


def _row_value(row: dict[str, str], *candidates: str, default: str = "") -> str:
    """Fetch the first matching value from a row using flexible column aliases."""
    lowered = {key.lower(): value for key, value in row.items()}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return default


def _sequence_distance(seq_a: str, seq_b: str) -> int:
    """Compute a robust sequence distance that tolerates length shifts."""
    shared = sum(char_a != char_b for char_a, char_b in zip(seq_a, seq_b))
    return shared + abs(len(seq_a) - len(seq_b))


def summarize_records(records: list[DatasetRecord] | tuple[DatasetRecord, ...]) -> dict[str, Any]:
    """Summarize standardized records for reporting and split diagnostics."""
    if not records:
        return {
            "count": 0,
            "sequence_length": None,
        "fitness": {},
        "mutation_histogram": {},
        "position_histogram": {},
    }

    fitness_values = [record.fitness for record in records]
    fitness_mean = sum(fitness_values) / len(fitness_values)
    fitness_variance = sum((value - fitness_mean) ** 2 for value in fitness_values) / len(fitness_values)
    mutation_histogram: dict[str, int] = {}
    position_histogram: dict[str, int] = {}
    for record in records:
        mutation_key = str(record.mutation_count)
        mutation_histogram[mutation_key] = mutation_histogram.get(mutation_key, 0) + 1
        for position in record.position_index:
            key = str(position)
            position_histogram[key] = position_histogram.get(key, 0) + 1

    return {
        "count": len(records),
        "sequence_length": len(records[0].sequence),
        "fitness": {
            "min": min(fitness_values),
            "max": max(fitness_values),
            "mean": fitness_mean,
            "variance": fitness_variance,
            "std": fitness_variance ** 0.5,
        },
        "mutation_histogram": mutation_histogram,
        "position_histogram": position_histogram,
    }


def _parse_mutation_token(token: str) -> tuple[str, int, str]:
    normalized = token.strip().upper()
    match = MUTATION_TOKEN_PATTERN.fullmatch(normalized)
    if not match:
        raise ValueError(f"Invalid mutation token '{token}'.")
    return match.group(1), int(match.group(2)), match.group(3)


def _sort_mutation_tokens(tokens: list[str]) -> list[str]:
    return sorted(
        [token.upper() for token in tokens],
        key=lambda token: (_parse_mutation_token(token)[1], token),
    )


def _read_fasta_sequences(path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    entries.append((header, normalize_sequence("".join(chunks))))
                header = line[1:]
                chunks = []
            else:
                chunks.append(line)
    if header is not None:
        entries.append((header, normalize_sequence("".join(chunks))))
    return entries


def _combine_annotations(*annotations: str) -> str:
    token_map: dict[int, str] = {}
    for annotation in annotations:
        normalized = normalize_mutation_annotation(annotation)
        if normalized == "WT":
            continue
        for token in normalized.split(";"):
            _, position, _ = _parse_mutation_token(token)
            existing = token_map.get(position)
            if existing is not None and existing != token:
                raise ValueError(
                    f"Conflicting mutation annotations at position {position}: {existing} vs {token}"
                )
            token_map[position] = token
    if not token_map:
        return "WT"
    return ";".join(_sort_mutation_tokens(list(token_map.values())))


def _apply_annotation_to_wild_type(
    annotation: str,
    wild_type: str,
    position_order: tuple[int, ...],
) -> str:
    if len(wild_type) != len(position_order):
        raise ValueError("Wild-type sequence length must match the mutable-position order.")
    sequence = list(wild_type)
    position_to_index = {position: index for index, position in enumerate(position_order)}
    normalized = normalize_mutation_annotation(annotation)
    if normalized == "WT":
        return wild_type
    for token in normalized.split(";"):
        expected_wild, position, mutated = _parse_mutation_token(token)
        if position not in position_to_index:
            raise ValueError(f"Mutation position {position} is not represented in the scaffold.")
        index = position_to_index[position]
        if sequence[index] != expected_wild:
            raise ValueError(
                f"Wild-type mismatch for token '{token}': expected {expected_wild}, found {sequence[index]}."
            )
        sequence[index] = mutated
    return "".join(sequence)


def _reconstruct_wild_type_from_annotation(sequence: str, annotation: str) -> str:
    """Reverse substitution annotations to recover a wild-type sequence."""
    normalized = normalize_mutation_annotation(annotation)
    if normalized == "WT":
        return sequence
    wild_type = list(sequence)
    for token in normalized.split(";"):
        expected_wild, position, mutated = _parse_mutation_token(token)
        index = position - 1
        if index < 0 or index >= len(wild_type):
            raise ValueError(f"Mutation token '{token}' is out of bounds for sequence length {len(sequence)}.")
        if wild_type[index] != mutated:
            raise ValueError(
                f"Sequence residue mismatch for token '{token}': expected mutated residue {mutated}, "
                f"found {wild_type[index]}."
            )
        wild_type[index] = expected_wild
    return "".join(wild_type)


def _build_bundle(
    spec: DatasetSpec,
    source_paths: tuple[Path, ...],
    records: list[DatasetRecord],
    logger: Any | None,
    extra_metadata: dict[str, Any] | None = None,
) -> DatasetBundle:
    metadata = summarize_records(records)
    metadata["source_paths"] = [str(path) for path in source_paths]
    metadata["average_distance_to_wild_type"] = (
        sum(_sequence_distance(record.sequence, record.wild_type) for record in records) / len(records)
        if records
        else 0.0
    )
    if extra_metadata:
        metadata.update(extra_metadata)

    bundle = DatasetBundle(
        name=spec.name,
        benchmark=spec.benchmark,
        task=spec.task,
        source_path=" | ".join(str(path) for path in source_paths),
        records=tuple(records),
        metadata=metadata,
    )
    if logger is not None:
        logger.info(
            "Loaded dataset '%s' with %s records from %s",
            spec.name,
            len(records),
            ", ".join(str(path) for path in source_paths),
        )
    return bundle


def _infer_group_consensus(records: list[DatasetRecord]) -> tuple[list[DatasetRecord], dict[str, Any]]:
    """Infer a reference sequence for unresolved tabular datasets like FLIP2."""
    grouped: dict[tuple[str, int], list[DatasetRecord]] = {}
    for record in records:
        if not (record.mutation_annotation == "WT" and record.wild_type == record.sequence):
            continue
        key = (record.assay_id, len(record.sequence))
        grouped.setdefault(key, []).append(record)

    inferred_refs: dict[tuple[str, int], str] = {}
    for key, group_records in grouped.items():
        if not group_records:
            continue
        train_like = [
            record
            for record in group_records
            if str(record.extra_metadata.get("set", "")).strip().lower() == "train"
        ]
        reference_records = train_like or group_records
        if len({record.sequence for record in reference_records}) <= 1:
            continue

        sequence_length = len(reference_records[0].sequence)
        consensus_chars: list[str] = []
        for index in range(sequence_length):
            counts = Counter(record.sequence[index] for record in reference_records)
            consensus_chars.append(counts.most_common(1)[0][0])
        inferred_refs[key] = "".join(consensus_chars)

    if not inferred_refs:
        return records, {
            "inferred_reference_groups": 0,
            "records_with_inferred_reference": 0,
            "observed_set_labels": sorted(
                {
                    str(record.extra_metadata.get("set", "")).strip().lower()
                    for record in records
                    if str(record.extra_metadata.get("set", "")).strip()
                }
            ),
        }

    updated_records: list[DatasetRecord] = []
    updated_count = 0
    for record in records:
        key = (record.assay_id, len(record.sequence))
        inferred_wild_type = inferred_refs.get(key)
        if inferred_wild_type is None or not (
            record.mutation_annotation == "WT" and record.wild_type == record.sequence
        ):
            updated_records.append(record)
            continue

        annotation = infer_mutation_annotation(record.sequence, inferred_wild_type)
        if record.wild_type == inferred_wild_type and record.mutation_annotation == annotation:
            updated_records.append(record)
            continue

        extra_metadata = dict(record.extra_metadata)
        extra_metadata.setdefault("reference_source", "inferred_consensus")
        updated_records.append(
            replace(
                record,
                wild_type=inferred_wild_type,
                mutation_annotation=annotation,
                position_index=extract_positions(annotation),
                mutation_count=len(extract_positions(annotation)),
                extra_metadata=extra_metadata,
            )
        )
        updated_count += 1

    return updated_records, {
        "inferred_reference_groups": len(inferred_refs),
        "records_with_inferred_reference": updated_count,
        "observed_set_labels": sorted(
            {
                str(record.extra_metadata.get("set", "")).strip().lower()
                for record in updated_records
                if str(record.extra_metadata.get("set", "")).strip()
            }
        ),
    }


def _load_generic_tabular(spec: DatasetSpec, project_root: Path, logger: Any | None = None) -> DatasetBundle:
    """Load a generic CSV or TSV dataset into the standardized format."""
    source_path = spec.resolve_path(project_root)
    delimiter = _detect_delimiter(source_path)
    rows: list[DatasetRecord] = []
    seen_sequences: dict[str, float] = {}

    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for raw_row in reader:
            sequence = normalize_sequence(_row_value(raw_row, "sequence", "seq", "mutated_sequence"))
            raw_annotation = _row_value(
                raw_row,
                "mutation_annotation",
                "mutations",
                "mutant",
                default="",
            )
            annotation = normalize_mutation_annotation(raw_annotation)
            fitness = float(_row_value(raw_row, "fitness", "score", "label", "target", "DMS_score"))
            raw_wild_type = _row_value(raw_row, "wild_type", "wt_sequence", default="")
            if raw_wild_type:
                wild_type = normalize_sequence(raw_wild_type)
            elif annotation != "WT":
                wild_type = _reconstruct_wild_type_from_annotation(sequence, annotation)
            else:
                wild_type = sequence
            if annotation == "WT" and sequence != wild_type:
                annotation = infer_mutation_annotation(sequence, wild_type)

            if len(sequence) != len(wild_type):
                raise ValueError(
                    f"Sequence length mismatch for '{sequence}' and wild type '{wild_type}'."
                )

            if sequence in seen_sequences:
                if seen_sequences[sequence] != fitness:
                    raise ValueError(
                        f"Conflicting fitness values for duplicate sequence '{sequence}'."
                    )
                continue
            seen_sequences[sequence] = fitness

            assay_id = _row_value(
                raw_row,
                "assay_id",
                "assay",
                "dataset",
                "DMS_id",
                default=spec.task,
            )
            used_keys = {
                "sequence",
                "seq",
                "mutated_sequence",
                "fitness",
                "score",
                "label",
                "target",
                "dms_score",
                "wild_type",
                "wt_sequence",
                "mutation_annotation",
                "mutations",
                "mutant",
                "assay_id",
                "assay",
                "dataset",
                "dms_id",
            }
            extra = {key: value for key, value in raw_row.items() if key.lower() not in used_keys}
            positions = extract_positions(annotation)

            rows.append(
                DatasetRecord(
                    sequence=sequence,
                    fitness=fitness,
                    wild_type=wild_type,
                    mutation_annotation=annotation,
                    position_index=positions,
                    mutation_count=len(positions),
                    benchmark=spec.benchmark,
                    task=spec.task,
                    assay_id=assay_id,
                    extra_metadata=extra,
                )
            )

    rows, extra_metadata = _infer_group_consensus(rows)
    return _build_bundle(spec, (source_path,), rows, logger, extra_metadata=extra_metadata)


def _load_flip_gb1_landscape(spec: DatasetSpec, project_root: Path, logger: Any | None = None) -> DatasetBundle:
    source_path = spec.resolve_path(project_root)
    rows: list[DatasetRecord] = []
    seen_sequences: set[str] = set()

    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"GB1 landscape file '{source_path}' is missing a header row.")
        mutation_columns = [field for field in reader.fieldnames if field not in {"Background", "Fit"}]
        position_to_wild: dict[int, str] = {}
        for token in mutation_columns:
            wild_type_char, position, _ = _parse_mutation_token(token)
            existing = position_to_wild.get(position)
            if existing is not None and existing != wild_type_char:
                raise ValueError(
                    f"Inconsistent GB1 wild-type residue at position {position}: {existing} vs {wild_type_char}."
                )
            position_to_wild[position] = wild_type_char

        position_order = tuple(sorted(position_to_wild))
        mutable_wild_type = "".join(position_to_wild[position] for position in position_order)
        background_count = 0
        pairwise_count = 0

        def add_record(annotation: str, fitness: float, extra: dict[str, Any]) -> None:
            normalized_annotation = normalize_mutation_annotation(annotation)
            sequence = _apply_annotation_to_wild_type(
                normalized_annotation,
                mutable_wild_type,
                position_order,
            )
            if sequence in seen_sequences:
                return
            seen_sequences.add(sequence)
            positions = extract_positions(normalized_annotation)
            rows.append(
                DatasetRecord(
                    sequence=sequence,
                    fitness=fitness,
                    wild_type=mutable_wild_type,
                    mutation_annotation=normalized_annotation,
                    position_index=positions,
                    mutation_count=len(positions),
                    benchmark=spec.benchmark,
                    task=spec.task,
                    assay_id=spec.task,
                    extra_metadata=extra,
                )
            )

        for raw_row in reader:
            background = normalize_mutation_annotation(raw_row["Background"])
            background_positions = extract_positions(background)
            add_record(
                background,
                float(raw_row["Fit"]),
                {
                    "source": "background_fit",
                    "background": background,
                },
            )
            background_count += 1

            background_position = background_positions[0] if background_positions else None
            for mutation_column in mutation_columns:
                raw_value = raw_row[mutation_column].strip()
                if not raw_value or raw_value.upper() == "NA":
                    continue
                column_position = _parse_mutation_token(mutation_column)[1]
                if background_position is not None and column_position <= background_position:
                    continue
                combined = _combine_annotations(background, mutation_column)
                add_record(
                    combined,
                    float(raw_value),
                    {
                        "source": "pairwise_landscape",
                        "background": background,
                        "added_mutation": mutation_column,
                    },
                )
                pairwise_count += 1

    return _build_bundle(
        spec,
        (source_path,),
        rows,
        logger,
        extra_metadata={
            "representation_scope": "mutable_positions_only",
            "mutable_position_order": list(position_order),
            "single_mutant_background_count": background_count,
            "pairwise_observation_count": pairwise_count,
        },
    )


def _decode_aav_mask(
    mask: str,
    reference_region: str,
) -> tuple[list[str], dict[int, list[str]]]:
    residues: list[str] = []
    insertions: dict[int, list[str]] = {index: [] for index in range(len(reference_region) + 1)}
    ref_index = 0
    for char in mask.strip():
        if char == "_":
            if ref_index >= len(reference_region):
                raise ValueError("AAV mask consumed more reference positions than available.")
            residues.append(reference_region[ref_index])
            ref_index += 1
        elif char.isupper():
            if ref_index >= len(reference_region):
                raise ValueError("AAV mask consumed more reference positions than available.")
            residues.append(char.upper())
            ref_index += 1
        elif char.islower():
            insertions[ref_index].append(char.upper())
        else:
            raise ValueError(f"Unsupported AAV mask token '{char}'.")
    if ref_index != len(reference_region):
        raise ValueError("AAV mask did not cover the full reference region.")
    return residues, insertions


def _flatten_aav_region(
    residues: list[str],
    insertions: dict[int, list[str]],
) -> str:
    parts: list[str] = []
    for index, residue in enumerate(residues):
        parts.extend(insertions.get(index, []))
        parts.append(residue)
    parts.extend(insertions.get(len(residues), []))
    return "".join(parts)


def _align_aav_region(
    residues: list[str],
    insertions: dict[int, list[str]],
    max_insertions: list[int],
) -> str:
    parts: list[str] = []
    for index, residue in enumerate(residues):
        inserted = insertions.get(index, [])
        parts.extend(inserted)
        parts.extend("X" for _ in range(max_insertions[index] - len(inserted)))
        parts.append(residue)
    terminal_insertions = insertions.get(len(residues), [])
    parts.extend(terminal_insertions)
    parts.extend("X" for _ in range(max_insertions[len(residues)] - len(terminal_insertions)))
    return "".join(parts)


def _load_flip_aav(spec: DatasetSpec, project_root: Path, logger: Any | None = None) -> DatasetBundle:
    score_path, fasta_path = spec.resolve_paths(project_root)
    fasta_entries = _read_fasta_sequences(fasta_path)
    if len(fasta_entries) != 1:
        raise ValueError(f"Expected a single AAV reference sequence in '{fasta_path}'.")
    full_reference_sequence = fasta_entries[0][1]
    rows: list[DatasetRecord] = []
    seen_sequences: dict[str, float] = {}
    decoded_rows: list[dict[str, Any]] = []
    reference_region: str | None = None
    max_insertions: list[int] | None = None

    with score_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            current_reference_region = normalize_sequence(raw_row["reference_region"])
            if reference_region is None:
                reference_region = current_reference_region
                max_insertions = [0 for _ in range(len(reference_region) + 1)]
            elif current_reference_region != reference_region:
                raise ValueError("AAV processed data contains multiple reference regions.")

            residues, insertions = _decode_aav_mask(raw_row["mask"], current_reference_region)
            mutated_region = normalize_sequence(raw_row["mutated_region"])
            if _flatten_aav_region(residues, insertions) != mutated_region:
                raise ValueError("AAV mask decoding did not reconstruct the mutated region.")

            for boundary, inserted in insertions.items():
                max_insertions[boundary] = max(max_insertions[boundary], len(inserted))

            decoded_rows.append(
                {
                    "score": float(raw_row["score"]),
                    "mask": raw_row["mask"],
                    "mutated_region": mutated_region,
                    "full_aa_sequence": normalize_sequence(raw_row["full_aa_sequence"]),
                    "residues": residues,
                    "insertions": insertions,
                }
            )

    if reference_region is None or max_insertions is None:
        raise ValueError(f"AAV dataset '{score_path}' does not contain any rows.")

    aligned_wild_type = _align_aav_region(
        residues=list(reference_region),
        insertions={index: [] for index in range(len(reference_region) + 1)},
        max_insertions=max_insertions,
    )

    for decoded in decoded_rows:
        sequence = _align_aav_region(
            residues=decoded["residues"],
            insertions=decoded["insertions"],
            max_insertions=max_insertions,
        )
        fitness = decoded["score"]
        annotation = infer_mutation_annotation(sequence, aligned_wild_type)
        if sequence in seen_sequences:
            if seen_sequences[sequence] != fitness:
                raise ValueError(
                    f"Conflicting fitness values for duplicate AAV sequence '{decoded['mutated_region']}'."
                )
            continue
        seen_sequences[sequence] = fitness
        positions = extract_positions(annotation)
        rows.append(
            DatasetRecord(
                sequence=sequence,
                fitness=fitness,
                wild_type=aligned_wild_type,
                mutation_annotation=annotation,
                position_index=positions,
                mutation_count=len(positions),
                benchmark=spec.benchmark,
                task=spec.task,
                assay_id=spec.task,
                extra_metadata={
                    "mask": decoded["mask"],
                    "reference_region": reference_region,
                    "mutated_region": decoded["mutated_region"],
                    "full_aa_sequence": decoded["full_aa_sequence"],
                },
            )
        )

    return _build_bundle(
        spec,
        (score_path, fasta_path),
        rows,
        logger,
        extra_metadata={
            "representation_scope": "engineered_region_alignment",
            "full_reference_sequence_length": len(full_reference_sequence),
            "reference_region_length": len(reference_region),
            "aligned_region_length": len(aligned_wild_type),
            "max_insertions_per_boundary": list(max_insertions),
        },
    )


def _choose_longest(existing: str | None, candidate: str) -> str:
    if not existing:
        return candidate
    return candidate if len(candidate) > len(existing) else existing


def _load_flip_meltome_human(spec: DatasetSpec, project_root: Path, logger: Any | None = None) -> DatasetBundle:
    assay_path, sequence_path = spec.resolve_paths(project_root)
    gene_to_sequence: dict[str, str] = {}
    with sequence_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for raw_row in reader:
            gene = raw_row.get("Gene names  (primary )", "").strip()
            sequence = raw_row.get("Sequence", "").strip()
            if not gene or not sequence:
                continue
            gene_to_sequence[gene] = _choose_longest(gene_to_sequence.get(gene), normalize_sequence(sequence))

    aggregated: dict[str, dict[str, Any]] = {}
    missing_sequence_count = 0
    with assay_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            gene = raw_row["gene_name"].strip()
            sequence = gene_to_sequence.get(gene)
            value = raw_row.get("quan_norm_meltPoint") or raw_row.get("meltPoint") or ""
            if not sequence:
                missing_sequence_count += 1
                continue
            if not value or value.upper() == "NA":
                continue
            payload = aggregated.setdefault(
                sequence,
                {
                    "fitness_values": [],
                    "gene_names": set(),
                    "cell_types": set(),
                },
            )
            payload["fitness_values"].append(float(value))
            payload["gene_names"].add(gene)
            cell_type = raw_row.get("cell_line_or_type", "").strip()
            if cell_type:
                payload["cell_types"].add(cell_type)

    rows: list[DatasetRecord] = []
    for sequence, payload in aggregated.items():
        values = payload["fitness_values"]
        rows.append(
            DatasetRecord(
                sequence=sequence,
                fitness=sum(values) / len(values),
                wild_type=sequence,
                mutation_annotation="WT",
                position_index=(),
                mutation_count=0,
                benchmark=spec.benchmark,
                task=spec.task,
                assay_id=spec.task,
                extra_metadata={
                    "gene_names": sorted(payload["gene_names"]),
                    "cell_types": sorted(payload["cell_types"]),
                    "measurement_count": len(values),
                },
            )
        )

    return _build_bundle(
        spec,
        (assay_path, sequence_path),
        rows,
        logger,
        extra_metadata={
            "aggregation_level": "sequence_mean",
            "missing_sequence_rows": missing_sequence_count,
        },
    )


def _load_flip_meltome_cross_species(
    spec: DatasetSpec,
    project_root: Path,
    logger: Any | None = None,
) -> DatasetBundle:
    assay_path, fasta_path = spec.resolve_paths(project_root)
    entry_to_sequence: dict[str, str] = {}
    for header, sequence in _read_fasta_sequences(fasta_path):
        header_parts = header.split("|")
        if len(header_parts) >= 2:
            entry_to_sequence[header_parts[1]] = sequence

    aggregated: dict[str, dict[str, Any]] = {}
    missing_sequence_count = 0
    with assay_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            protein_id = raw_row["Protein_ID"].strip()
            entry_id = protein_id.split("_", 1)[0]
            sequence = entry_to_sequence.get(entry_id)
            value = raw_row.get("meltPoint", "")
            if not sequence:
                missing_sequence_count += 1
                continue
            if not value or value.upper() == "NA":
                continue
            payload = aggregated.setdefault(
                sequence,
                {
                    "fitness_values": [],
                    "entry_ids": set(),
                    "gene_names": set(),
                    "run_names": set(),
                },
            )
            payload["fitness_values"].append(float(value))
            payload["entry_ids"].add(entry_id)
            gene_name = raw_row.get("gene_name", "").strip()
            run_name = raw_row.get("run_name", "").strip()
            if gene_name:
                payload["gene_names"].add(gene_name)
            if run_name:
                payload["run_names"].add(run_name)

    rows: list[DatasetRecord] = []
    for sequence, payload in aggregated.items():
        values = payload["fitness_values"]
        rows.append(
            DatasetRecord(
                sequence=sequence,
                fitness=sum(values) / len(values),
                wild_type=sequence,
                mutation_annotation="WT",
                position_index=(),
                mutation_count=0,
                benchmark=spec.benchmark,
                task=spec.task,
                assay_id=spec.task,
                extra_metadata={
                    "entry_ids": sorted(payload["entry_ids"]),
                    "gene_names": sorted(payload["gene_names"]),
                    "run_names": sorted(payload["run_names"]),
                    "measurement_count": len(values),
                },
            )
        )

    return _build_bundle(
        spec,
        (assay_path, fasta_path),
        rows,
        logger,
        extra_metadata={
            "aggregation_level": "sequence_mean",
            "missing_sequence_rows": missing_sequence_count,
        },
    )


def load_dataset(spec: DatasetSpec, project_root: Path, logger: Any | None = None) -> DatasetBundle:
    """Load a registered dataset into the standardized in-memory format."""
    if spec.format_name == "flip_gb1_landscape":
        return _load_flip_gb1_landscape(spec, project_root, logger=logger)
    if spec.format_name == "flip_aav":
        return _load_flip_aav(spec, project_root, logger=logger)
    if spec.format_name == "flip_meltome_human":
        return _load_flip_meltome_human(spec, project_root, logger=logger)
    if spec.format_name == "flip_meltome_cross_species":
        return _load_flip_meltome_cross_species(spec, project_root, logger=logger)
    return _load_generic_tabular(spec, project_root, logger=logger)
