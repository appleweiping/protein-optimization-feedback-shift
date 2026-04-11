"""Closed-loop experiment runner shell for the Day 2/3 data foundation work."""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from data.data_loader import load_dataset
from data.dataset_registry import resolve_dataset
from data.oracle import Oracle
from data.split import SplitResult, build_split
from data.validation import validate_oracle_consistency, validate_split_against_oracle
from utils.config import ConfigNode


class ClosedLoopRunner:
    """Minimal runner that validates the data environment and oracle foundation."""

    def __init__(self, config: ConfigNode, logger: Any, context: dict[str, Any]) -> None:
        self.config = config
        self.logger = logger
        self.context = context
        self.artifacts_dir = Path(context["paths"]["artifacts_dir"])
        self.plots_dir = Path(context["paths"]["plots_dir"])

    def run(self) -> dict[str, Any]:
        """Run the Day 2/3 data-environment validation flow."""
        self.logger.info("Runner initialized for experiment '%s'.", self.config.experiment.name)
        self.logger.info("Validating config sections and constructing the data environment.")

        required_sections = (
            "experiment",
            "runtime",
            "dataset",
            "representation",
            "model",
            "uq",
            "acquisition",
            "proposal",
            "loop",
            "evaluation",
        )
        missing = [section for section in required_sections if section not in self.config.to_dict()]
        if missing:
            raise ValueError(f"Missing required config sections: {missing}")

        project_root = Path(self.context["project_root"])
        dataset_spec = resolve_dataset(
            registry_name=getattr(self.config.dataset, "registry_name", None),
            benchmark=self.config.dataset.benchmark,
            task=self.config.dataset.task,
        )
        dataset_bundle = load_dataset(dataset_spec, project_root=project_root, logger=self.logger)
        split_result, split_suite_summary = self._build_split_suite(
            dataset_bundle,
            processed_dir=project_root / "data" / "processed",
        )
        validation_config = dict(self.config.dataset.to_dict().get("validation", {}))
        oracle = Oracle(
            dataset_bundle.records,
            logger=self.logger,
            enable_query_logging=bool(validation_config.get("enable_query_logging", False)),
        )
        oracle_validation = self._validate_oracle(split_result, dataset_bundle, oracle)
        plot_paths = self._write_split_plots(split_result)

        dataset_summary_path = self.artifacts_dir / "dataset_summary.json"
        split_summary_path = self.artifacts_dir / "split_summary.json"
        split_suite_summary_path = self.artifacts_dir / "split_suite_summary.json"
        oracle_validation_path = self.artifacts_dir / "oracle_validation.json"
        dataset_summary_path.write_text(
            json.dumps(dataset_bundle.to_summary_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        split_summary_path.write_text(
            json.dumps(split_result.statistics, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        split_suite_summary_path.write_text(
            json.dumps(split_suite_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        oracle_validation_path.write_text(
            json.dumps(oracle_validation, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        trace_payload = {
            "run_id": self.context["run_id"],
            "stage": "day3_oracle_validation",
            "status": "ok",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "message": "Data environment, split validation, and oracle consistency checks completed.",
        }
        trace_path = self.artifacts_dir / "runner_trace.jsonl"
        trace_path.write_text(json.dumps(trace_payload) + "\n", encoding="utf-8")

        summary = {
            "run_id": self.context["run_id"],
            "status": "completed",
            "message": "Day 3 oracle and split validation completed successfully.",
            "project_root": str(self.context["project_root"]),
            "config_hash": self.context["config_hash"],
            "seed_report": self.context["seed_report"],
            "device_info": self.context["device_info"],
            "dataset": dataset_bundle.to_summary_dict(),
            "split_id": split_result.split_id,
            "split_statistics": split_result.statistics,
            "split_suite_summary": split_suite_summary,
            "oracle_validation": oracle_validation,
            "plot_paths": plot_paths,
            "outputs": {
                key: str(value)
                for key, value in self.context["paths"].items()
                if key != "run_dir"
            },
            "validated_sections": list(required_sections),
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }

        summary_path = self.artifacts_dir / "runner_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        self.logger.info("Runner summary written to %s", summary_path)
        return summary

    def _build_split_suite(
        self,
        dataset_bundle: Any,
        processed_dir: Path,
    ) -> tuple[SplitResult, dict[str, Any]]:
        """Build the primary split plus a Day 2 diagnostic suite of core split types."""
        dataset_config = self.config.dataset.to_dict()
        primary_split_type = str(dataset_config.get("split_type", "low_resource"))
        requested_suite = dataset_config.get("diagnostic_split_types") or [
            primary_split_type,
        ]

        ordered_split_types: list[str] = []
        for split_type in [primary_split_type, *requested_suite]:
            normalized = str(split_type).strip()
            if normalized and normalized not in ordered_split_types:
                ordered_split_types.append(normalized)

        split_suite_summary: dict[str, Any] = {}
        primary_result: SplitResult | None = None
        for split_type in ordered_split_types:
            split_config = dict(dataset_config)
            split_config["split_type"] = split_type
            try:
                split_result = build_split(
                    dataset_bundle,
                    split_config,
                    processed_dir=processed_dir,
                    logger=self.logger,
                )
            except Exception as exc:
                split_suite_summary[split_type] = {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
                self.logger.warning("Split '%s' failed during suite construction: %s", split_type, exc)
                if split_type == primary_split_type:
                    raise
                continue

            split_suite_summary[split_type] = {
                "status": "ok",
                "split_id": split_result.split_id,
                "cache_paths": split_result.cache_paths,
                "statistics": split_result.statistics,
            }
            if split_type == primary_split_type:
                primary_result = split_result

        if primary_result is None:
            raise ValueError(f"Primary split '{primary_split_type}' was not constructed successfully.")
        return primary_result, split_suite_summary

    def _validate_oracle(
        self,
        split_result: SplitResult,
        dataset_bundle: Any,
        oracle: Oracle,
    ) -> dict[str, Any]:
        """Run the Day 3 oracle and split validation suite."""
        dataset_config = self.config.dataset.to_dict()
        validation_config = dict(dataset_config.get("validation", {}))
        report: dict[str, Any] = {
            "dataset_hash": oracle.dataset_hash,
            "validation_config": validation_config,
        }

        if bool(validation_config.get("enable_oracle_check", True)):
            report["oracle_consistency"] = validate_oracle_consistency(
                dataset_bundle,
                oracle,
                dataset_config,
                logger=self.logger,
            )

        if bool(validation_config.get("enable_split_check", True)):
            report["split_queryability"] = validate_split_against_oracle(
                split_result,
                oracle,
                dataset_config,
                logger=self.logger,
            )

        sample_count = min(
            int(getattr(self.config.dataset, "query_sequence_count", 3)),
            len(split_result.candidate_records),
        )
        rng = random.Random(int(self.config.dataset.split_seed))
        candidate_records = list(split_result.candidate_records)
        sampled_records = rng.sample(candidate_records, sample_count)
        queried = oracle.batch_query(
            [record.sequence for record in sampled_records],
            source="runner_sample_check",
            log_query=False,
            record_history=False,
        )

        mismatches: list[dict[str, Any]] = []
        for record, result in zip(sampled_records, queried):
            if record.fitness != result.fitness:
                mismatches.append(
                    {
                        "sequence": record.sequence,
                        "expected": record.fitness,
                        "observed": result.fitness,
                    }
                )

        report["sampled_query_check"] = {
            "sample_count": sample_count,
            "successful": not mismatches,
            "sampled_sequences": [record.sequence for record in sampled_records],
            "mismatches": mismatches,
        }
        report["successful"] = all(
            section.get("successful", True)
            for section in report.values()
            if isinstance(section, dict)
        )
        if mismatches:
            raise ValueError(f"Oracle validation failed: {mismatches}")
        return report

    def _write_split_plots(self, split_result: SplitResult) -> dict[str, str]:
        """Write simple SVG diagnostics for fitness and mutation-count distributions."""
        fitness_path = self.plots_dir / f"{split_result.split_id}_fitness_distribution.svg"
        mutation_path = self.plots_dir / f"{split_result.split_id}_mutation_distribution.svg"

        train_fitness = [record.fitness for record in split_result.train_records]
        candidate_fitness = [record.fitness for record in split_result.candidate_records]
        self._write_numeric_histogram_svg(
            fitness_path,
            title="Train vs Candidate Fitness Distribution",
            train_values=train_fitness,
            candidate_values=candidate_fitness,
            width=760,
            height=420,
        )

        train_mutations = self._count_categories(record.mutation_count for record in split_result.train_records)
        candidate_mutations = self._count_categories(record.mutation_count for record in split_result.candidate_records)
        self._write_category_bar_svg(
            mutation_path,
            title="Train vs Candidate Mutation Count",
            train_counts=train_mutations,
            candidate_counts=candidate_mutations,
            width=760,
            height=420,
        )
        return {
            "fitness_distribution": str(fitness_path),
            "mutation_distribution": str(mutation_path),
        }

    def _count_categories(self, values: Any) -> dict[str, int]:
        counts: dict[str, int] = {}
        for value in values:
            key = str(value)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _write_numeric_histogram_svg(
        self,
        path: Path,
        title: str,
        train_values: list[float],
        candidate_values: list[float],
        width: int,
        height: int,
    ) -> None:
        """Write a simple two-series histogram as SVG without external plotting libs."""
        values = train_values + candidate_values
        if not values:
            return
        bins = 6
        min_value = min(values)
        max_value = max(values)
        span = max(max_value - min_value, 1e-6)
        step = span / bins

        def bucketize(series: list[float]) -> list[int]:
            counts = [0 for _ in range(bins)]
            for value in series:
                index = min(bins - 1, int((value - min_value) / step))
                counts[index] += 1
            return counts

        train_counts = bucketize(train_values)
        candidate_counts = bucketize(candidate_values)
        max_count = max(train_counts + candidate_counts + [1])
        margin = 50
        chart_height = height - 2 * margin
        band_width = (width - 2 * margin) / bins
        bar_width = band_width * 0.35

        bars: list[str] = []
        labels: list[str] = []
        for index in range(bins):
            x_base = margin + index * band_width
            train_height = (train_counts[index] / max_count) * chart_height
            candidate_height = (candidate_counts[index] / max_count) * chart_height
            bars.append(
                f"<rect x='{x_base:.1f}' y='{height - margin - train_height:.1f}' width='{bar_width:.1f}' height='{train_height:.1f}' fill='#1f77b4' />"
            )
            bars.append(
                f"<rect x='{x_base + bar_width + 6:.1f}' y='{height - margin - candidate_height:.1f}' width='{bar_width:.1f}' height='{candidate_height:.1f}' fill='#ff7f0e' />"
            )
            lower = min_value + index * step
            upper = lower + step
            labels.append(
                f"<text x='{x_base + band_width / 2:.1f}' y='{height - margin + 18}' text-anchor='middle' font-size='11'>{lower:.2f}-{upper:.2f}</text>"
            )

        svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>
<rect width='100%' height='100%' fill='white' />
<text x='{width / 2:.1f}' y='28' text-anchor='middle' font-size='18' font-family='Arial'>{title}</text>
<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' stroke='black' />
<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height - margin}' stroke='black' />
{''.join(bars)}
{''.join(labels)}
<rect x='{width - 190}' y='42' width='14' height='14' fill='#1f77b4' />
<text x='{width - 168}' y='54' font-size='12' font-family='Arial'>train</text>
<rect x='{width - 110}' y='42' width='14' height='14' fill='#ff7f0e' />
<text x='{width - 88}' y='54' font-size='12' font-family='Arial'>candidate</text>
</svg>"""
        path.write_text(svg, encoding="utf-8")

    def _write_category_bar_svg(
        self,
        path: Path,
        title: str,
        train_counts: dict[str, int],
        candidate_counts: dict[str, int],
        width: int,
        height: int,
    ) -> None:
        """Write a categorical two-series bar chart as SVG."""
        categories = sorted({*train_counts.keys(), *candidate_counts.keys()}, key=lambda value: int(value))
        max_count = max(list(train_counts.values()) + list(candidate_counts.values()) + [1])
        margin = 50
        chart_height = height - 2 * margin
        band_width = (width - 2 * margin) / max(len(categories), 1)
        bar_width = band_width * 0.35

        bars: list[str] = []
        labels: list[str] = []
        for index, category in enumerate(categories):
            x_base = margin + index * band_width
            train_height = (train_counts.get(category, 0) / max_count) * chart_height
            candidate_height = (candidate_counts.get(category, 0) / max_count) * chart_height
            bars.append(
                f"<rect x='{x_base:.1f}' y='{height - margin - train_height:.1f}' width='{bar_width:.1f}' height='{train_height:.1f}' fill='#2a9d8f' />"
            )
            bars.append(
                f"<rect x='{x_base + bar_width + 6:.1f}' y='{height - margin - candidate_height:.1f}' width='{bar_width:.1f}' height='{candidate_height:.1f}' fill='#e76f51' />"
            )
            labels.append(
                f"<text x='{x_base + band_width / 2:.1f}' y='{height - margin + 18}' text-anchor='middle' font-size='11'>{category}</text>"
            )

        svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>
<rect width='100%' height='100%' fill='white' />
<text x='{width / 2:.1f}' y='28' text-anchor='middle' font-size='18' font-family='Arial'>{title}</text>
<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' stroke='black' />
<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height - margin}' stroke='black' />
{''.join(bars)}
{''.join(labels)}
<rect x='{width - 190}' y='42' width='14' height='14' fill='#2a9d8f' />
<text x='{width - 168}' y='54' font-size='12' font-family='Arial'>train</text>
<rect x='{width - 110}' y='42' width='14' height='14' fill='#e76f51' />
<text x='{width - 88}' y='54' font-size='12' font-family='Arial'>candidate</text>
</svg>"""
        path.write_text(svg, encoding="utf-8")
