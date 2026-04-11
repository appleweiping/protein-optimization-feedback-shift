"""Closed-loop system runner for the Day 7 execution milestone."""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from acquisition.registry import build_acquisition
from data.data_loader import DatasetBundle, load_dataset
from data.dataset_registry import resolve_dataset
from data.oracle import Oracle
from data.split import SplitResult, build_split
from data.validation import validate_oracle_consistency, validate_split_against_oracle
from loop.buffer import ClosedLoopBuffer
from loop.recorder import LoopRecorder
from loop.state import LoopState
from loop.stopping import LoopStopping
from models.checkpoint import CheckpointManager
from models.ensemble import DeepEnsemble
from models.trainer import EnsembleTrainer
from representation.interface import build_encoder
from utils.config import ConfigNode


class ClosedLoopRunner:
    """Single execution core that runs the full closed-loop optimization system."""

    def __init__(self, config: ConfigNode, logger: Any, context: dict[str, Any]) -> None:
        self.config = config
        self.logger = logger
        self.context = context
        self.run_dir = Path(context["paths"]["run_dir"])
        self.artifacts_dir = Path(context["paths"]["artifacts_dir"])
        self.plots_dir = Path(context["paths"]["plots_dir"])
        self.tables_dir = Path(context["paths"]["tables_dir"])
        self.checkpoints_dir = Path(context["paths"]["checkpoints_dir"])

    def run(self) -> dict[str, Any]:
        """Build the environment, validate it, and execute the closed loop."""
        self.logger.info("Closed-loop runner initialized for experiment '%s'.", self.config.experiment.name)

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
        oracle = Oracle(
            dataset_bundle.records,
            logger=self.logger,
            enable_query_logging=bool(self.config.dataset.to_dict().get("validation", {}).get("enable_query_logging", False)),
        )
        oracle_validation = self._validate_oracle(split_result, dataset_bundle, oracle)
        plot_paths = self._write_split_plots(split_result)
        loop_suite_summary = self._run_closed_loop_suite(split_result, oracle)

        dataset_summary_path = self.artifacts_dir / "dataset_summary.json"
        split_summary_path = self.artifacts_dir / "split_summary.json"
        split_suite_summary_path = self.artifacts_dir / "split_suite_summary.json"
        oracle_validation_path = self.artifacts_dir / "oracle_validation.json"
        loop_suite_summary_path = self.artifacts_dir / "loop_suite_summary.json"
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
        loop_suite_summary_path.write_text(
            json.dumps(loop_suite_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        trace_payload = {
            "run_id": self.context["run_id"],
            "stage": "day7_closed_loop",
            "status": "ok",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "message": "Closed-loop system execution completed.",
        }
        trace_path = self.artifacts_dir / "runner_trace.jsonl"
        trace_path.write_text(json.dumps(trace_payload) + "\n", encoding="utf-8")

        summary = {
            "run_id": self.context["run_id"],
            "status": "completed",
            "message": "Day 7 closed-loop foundation completed successfully.",
            "project_root": str(self.context["project_root"]),
            "config_hash": self.context["config_hash"],
            "seed_report": self.context["seed_report"],
            "device_info": self.context["device_info"],
            "dataset": dataset_bundle.to_summary_dict(),
            "split_id": split_result.split_id,
            "split_statistics": split_result.statistics,
            "split_suite_summary": split_suite_summary,
            "oracle_validation": oracle_validation,
            "loop_suite_summary": loop_suite_summary,
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

    def _run_closed_loop_suite(
        self,
        split_result: SplitResult,
        oracle: Oracle,
    ) -> dict[str, Any]:
        """Run the primary acquisition loop plus any configured Day 7 comparison loops."""
        loop_config = self.config.loop.to_dict()
        primary_name = str(self.config.acquisition.name)
        comparison_names = list(loop_config.get("comparison_acquisition_names", []) or [])
        ordered_names: list[str] = []
        for name in [primary_name, *comparison_names]:
            normalized = str(name).strip().lower()
            if normalized and normalized not in ordered_names:
                ordered_names.append(normalized)

        suite_results: dict[str, Any] = {}
        for acquisition_name in ordered_names:
            acquisition_config = dict(self.config.acquisition.to_dict())
            acquisition_config["name"] = acquisition_name
            acquisition_config["acquisition_type"] = acquisition_name
            self.logger.info("Starting closed loop for acquisition '%s'.", acquisition_name)
            suite_results[acquisition_name] = self._run_single_loop(
                split_result=split_result,
                oracle=oracle,
                acquisition_config=acquisition_config,
                label=acquisition_name,
            )

        return {
            "ordered_methods": ordered_names,
            "methods": suite_results,
            "checks": self._build_suite_checks(suite_results),
            "comparison_plot_path": str(self._write_suite_curve_svg(suite_results, split_result.split_id)),
            "successful": all(result.get("successful", False) for result in suite_results.values()),
        }

    def _run_single_loop(
        self,
        split_result: SplitResult,
        oracle: Oracle,
        acquisition_config: dict[str, Any],
        label: str,
    ) -> dict[str, Any]:
        """Execute one full train→predict→select→query→update loop."""
        state = LoopState.initialize(
            observed_pool=split_result.train_records,
            candidate_pool=split_result.candidate_records,
        )
        recorder = LoopRecorder(self.context["paths"], label=label, logger=self.logger)
        recorder.record_initial_state(state, acquisition_name=label, split_id=split_result.split_id)
        encoder = build_encoder(self.config.representation.to_dict(), logger=self.logger)
        buffer = ClosedLoopBuffer(logger=self.logger)
        stopping = LoopStopping.from_config(self.config.loop.to_dict())
        device = str(self.context["device_info"]["resolved"])
        split_id = f"{split_result.split_id}_{label}"
        checkpoint_root = self.checkpoints_dir / label

        stop_decision = stopping.decide(state)
        while not stop_decision.stop:
            observed_sequences = [record.sequence for record in state.observed_pool]
            observed_targets = np.asarray([record.fitness for record in state.observed_pool], dtype=np.float32)
            candidate_sequences = [record.sequence for record in state.candidate_pool]
            train_features = encoder.encode(observed_sequences).astype(np.float32, copy=False)
            candidate_features = encoder.encode(candidate_sequences).astype(np.float32, copy=False)

            ensemble = DeepEnsemble.from_config(
                self.config.model.to_dict(),
                input_dim=int(train_features.shape[1]),
                device=device,
                logger=self.logger,
            )
            trainer = EnsembleTrainer(
                config=self.config.model.to_dict(),
                device=device,
                logger=self.logger,
                checkpoint_manager=CheckpointManager(checkpoint_root, logger=self.logger),
            )
            training_summary = trainer.fit(
                ensemble=ensemble,
                train_features=train_features,
                train_targets=observed_targets,
                round_index=state.round_index,
                split_id=split_id,
                round_metadata={
                    "loop_label": label,
                    "observed_count": state.observed_count,
                    "candidate_count": state.candidate_count,
                },
            )

            predictions = ensemble.predict_with_uncertainty(
                candidate_features,
                batch_size=int(self.config.model.batch_size),
            )
            predictions["best_observed"] = np.asarray(float(state.best_so_far), dtype=np.float32)
            acquisition = build_acquisition(acquisition_config)
            selection = acquisition.select(
                candidates=list(state.candidate_pool),
                predictions=predictions,
                batch_size=stop_decision.next_batch_size,
            )
            selected_sequences = [
                state.candidate_pool[index].sequence
                for index in selection.selected_indices
            ]
            oracle_results = oracle.batch_query(
                selected_sequences,
                source=f"loop_{label}_round_{state.round_index}",
                log_query=False,
                record_history=True,
            )
            update = buffer.apply_selection(
                state=state,
                selection=selection,
                oracle_results=oracle_results,
            )
            recorder.record_round(
                state_before=state,
                update=update,
                predictions=predictions,
                selection=selection,
                training_summary=training_summary,
            )
            self.logger.info(
                "Loop '%s' round=%s observed=%s candidate=%s selected=%s best_so_far=%.6f",
                label,
                state.round_index,
                state.observed_count,
                state.candidate_count,
                len(selection.selected_indices),
                update.next_state.best_so_far,
            )
            state = update.next_state
            stop_decision = stopping.decide(state)

        return recorder.finalize(final_state=state, stop_decision=stop_decision)

    def _build_suite_checks(self, suite_results: dict[str, Any]) -> dict[str, Any]:
        """Summarize simple Day 7 sanity comparisons across acquisition methods."""
        checks: dict[str, Any] = {}
        if "random" in suite_results and "greedy" in suite_results:
            checks["greedy_final_best_gte_random"] = (
                suite_results["greedy"]["final_best_so_far"] >= suite_results["random"]["final_best_so_far"]
            )
        if "greedy" in suite_results and "ucb" in suite_results:
            checks["ucb_differs_from_greedy"] = (
                suite_results["greedy"]["trajectory"] != suite_results["ucb"]["trajectory"]
            )
        if "random" in suite_results and "ucb" in suite_results:
            checks["ucb_final_best_gte_random"] = (
                suite_results["ucb"]["final_best_so_far"] >= suite_results["random"]["final_best_so_far"]
            )
        return checks

    def _write_suite_curve_svg(self, suite_results: dict[str, Any], split_id: str) -> Path:
        """Write a comparison SVG for best-so-far trajectories across methods."""
        path = self.plots_dir / f"{split_id}_loop_suite_best_so_far.svg"
        width = 820
        height = 440
        margin = 56
        colors = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed"]
        all_points = [
            point
            for result in suite_results.values()
            for point in result.get("trajectory", [])
        ]
        if not all_points:
            path.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='10' height='10'></svg>", encoding="utf-8")
            return path

        xs = [float(point["step"]) for point in all_points]
        ys = [float(point["best_so_far"]) for point in all_points]
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)
        span_x = max(max_x - min_x, 1.0)
        span_y = max(max_y - min_y, 1e-6)

        def map_x(value: float) -> float:
            return margin + ((value - min_x) / span_x) * (width - 2 * margin)

        def map_y(value: float) -> float:
            return height - margin - ((value - min_y) / span_y) * (height - 2 * margin)

        lines: list[str] = []
        legend: list[str] = []
        for index, (label, result) in enumerate(suite_results.items()):
            color = colors[index % len(colors)]
            trajectory = result.get("trajectory", [])
            points = " ".join(
                f"{map_x(float(point['step'])):.1f},{map_y(float(point['best_so_far'])):.1f}"
                for point in trajectory
            )
            lines.append(f"<polyline fill='none' stroke='{color}' stroke-width='2.4' points='{points}' />")
            legend_y = 44 + index * 18
            legend.append(f"<rect x='{width - 190}' y='{legend_y}' width='12' height='12' fill='{color}' />")
            legend.append(
                f"<text x='{width - 170}' y='{legend_y + 10}' font-size='12' font-family='Arial'>{label}</text>"
            )

        svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>
<rect width='100%' height='100%' fill='white' />
<text x='{width / 2:.1f}' y='28' text-anchor='middle' font-size='18' font-family='Arial'>Day 7 loop comparison: best-so-far</text>
<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' stroke='black' />
<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height - margin}' stroke='black' />
{''.join(lines)}
{''.join(legend)}
<text x='{width - margin}' y='{height - margin + 24}' text-anchor='end' font-size='11'>round</text>
<text x='{margin - 8}' y='{margin - 10}' text-anchor='start' font-size='11'>fitness</text>
</svg>"""
        path.write_text(svg, encoding="utf-8")
        return path

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
