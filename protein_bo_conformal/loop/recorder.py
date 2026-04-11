"""Recording utilities for per-round diagnostics and artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from acquisition.registry import AcquisitionSelection
from loop.buffer import BufferUpdate
from loop.state import LoopState
from loop.stopping import StopDecision


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


class LoopRecorder:
    """Persist detailed closed-loop trajectories for later analysis."""

    def __init__(
        self,
        run_paths: dict[str, Path],
        label: str,
        logger: Any | None = None,
    ) -> None:
        self.run_paths = run_paths
        self.label = label
        self.logger = logger
        self._round_payloads: list[dict[str, Any]] = []
        self._selection_rows: list[dict[str, Any]] = []
        self._trajectory: list[dict[str, float]] = []
        self._initial_state: dict[str, Any] | None = None

    @property
    def jsonl_path(self) -> Path:
        return Path(self.run_paths["artifacts_dir"]) / f"{self.label}_loop_rounds.jsonl"

    @property
    def summary_path(self) -> Path:
        return Path(self.run_paths["artifacts_dir"]) / f"{self.label}_loop_summary.json"

    @property
    def table_path(self) -> Path:
        return Path(self.run_paths["tables_dir"]) / f"{self.label}_selected_observations.csv"

    @property
    def plot_path(self) -> Path:
        return Path(self.run_paths["plots_dir"]) / f"{self.label}_best_so_far.svg"

    def record_initial_state(
        self,
        state: LoopState,
        acquisition_name: str,
        split_id: str,
    ) -> None:
        self._initial_state = {
            "label": self.label,
            "acquisition_name": acquisition_name,
            "split_id": split_id,
            "state": state.to_dict(),
        }
        self._trajectory = [
            {
                "step": 0.0,
                "best_so_far": float(state.best_so_far),
            }
        ]

    def record_round(
        self,
        state_before: LoopState,
        update: BufferUpdate,
        predictions: dict[str, np.ndarray],
        selection: AcquisitionSelection,
        training_summary: dict[str, Any],
    ) -> None:
        selected_payloads: list[dict[str, Any]] = []
        for batch_offset, (record, oracle_result, detail) in enumerate(
            zip(update.selected_records, update.selected_oracle_results, selection.selected_details)
        ):
            payload = {
                "batch_offset": batch_offset,
                "selected_index": int(selection.selected_indices[batch_offset]),
                "sequence": record.sequence,
                "predicted_mean": float(detail.get("mean", 0.0)),
                "predicted_sigma": float(detail.get("sigma", 0.0)),
                "acquisition_score": float(detail.get("score", 0.0)),
                "true_fitness": float(oracle_result.fitness),
                "query_id": int(oracle_result.query_id),
                "is_new_query": bool(oracle_result.is_new),
                "mutation_count": int(record.mutation_count),
            }
            selected_payloads.append(payload)
            self._selection_rows.append(
                {
                    "label": self.label,
                    "round_index": state_before.round_index,
                    "selected_index": int(selection.selected_indices[batch_offset]),
                    "sequence": record.sequence,
                    "predicted_mean": payload["predicted_mean"],
                    "predicted_sigma": payload["predicted_sigma"],
                    "acquisition_score": payload["acquisition_score"],
                    "true_fitness": payload["true_fitness"],
                    "query_id": payload["query_id"],
                    "best_so_far_after": update.next_state.best_so_far,
                }
            )

        round_payload = {
            "label": self.label,
            "round_index": int(state_before.round_index),
            "observed_count_before": int(state_before.observed_count),
            "observed_count_after": int(update.next_state.observed_count),
            "candidate_count_before": int(state_before.candidate_count),
            "candidate_count_after": int(update.next_state.candidate_count),
            "best_so_far_before": float(state_before.best_so_far),
            "best_so_far_after": float(update.next_state.best_so_far),
            "selected_count": len(selected_payloads),
            "selected_candidate_indices": [int(index) for index in selection.selected_indices],
            "selected": selected_payloads,
            "acquisition_selection": selection.to_dict(),
            "candidate_prediction_summary": {
                "count": int(predictions["mean"].size),
                "mean_mu": float(predictions["mean"].mean()) if predictions["mean"].size else 0.0,
                "std_mu": float(predictions["mean"].std()) if predictions["mean"].size else 0.0,
                "max_mu": float(predictions["mean"].max()) if predictions["mean"].size else 0.0,
                "mean_sigma": float(predictions["sigma"].mean()) if predictions["sigma"].size else 0.0,
                "std_sigma": float(predictions["sigma"].std()) if predictions["sigma"].size else 0.0,
                "max_sigma": float(predictions["sigma"].max()) if predictions["sigma"].size else 0.0,
                "p90_sigma": float(np.percentile(predictions["sigma"], 90)) if predictions["sigma"].size else 0.0,
                "p95_sigma": float(np.percentile(predictions["sigma"], 95)) if predictions["sigma"].size else 0.0,
            },
            "selection_shift_hint": {
                "selected_mean_mu_gap": (
                    (sum(payload["predicted_mean"] for payload in selected_payloads) / len(selected_payloads))
                    - float(predictions["mean"].mean())
                )
                if selected_payloads and predictions["mean"].size
                else 0.0,
                "selected_mean_sigma_gap": (
                    (sum(payload["predicted_sigma"] for payload in selected_payloads) / len(selected_payloads))
                    - float(predictions["sigma"].mean())
                )
                if selected_payloads and predictions["sigma"].size
                else 0.0,
            },
            "training_summary": {
                "aggregate": dict(training_summary.get("aggregate", {})),
                "feature_dim": int(training_summary.get("feature_dim", 0)),
            },
        }
        self._round_payloads.append(round_payload)
        self._trajectory.append(
            {
                "step": float(update.next_state.round_index),
                "best_so_far": float(update.next_state.best_so_far),
            }
        )

    def finalize(
        self,
        final_state: LoopState,
        stop_decision: StopDecision,
    ) -> dict[str, Any]:
        with self.jsonl_path.open("w", encoding="utf-8") as handle:
            for payload in self._round_payloads:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")

        if self._selection_rows:
            fieldnames = list(self._selection_rows[0].keys())
            with self.table_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self._selection_rows)

        self._write_best_so_far_svg(self.plot_path, self._trajectory)
        summary = {
            "label": self.label,
            "initial_state": self._initial_state or {},
            "final_state": final_state.to_dict(),
            "stopping_reason": stop_decision.reason,
            "round_count": len(self._round_payloads),
            "total_selected": sum(payload["selected_count"] for payload in self._round_payloads),
            "initial_best_so_far": float(final_state.initial_best_so_far),
            "final_best_so_far": float(final_state.best_so_far),
            "best_improvement": float(final_state.best_improvement),
            "trajectory": list(self._trajectory),
            "paths": {
                "rounds_jsonl": str(self.jsonl_path),
                "selection_table": str(self.table_path),
                "best_so_far_plot": str(self.plot_path),
            },
            "successful": True,
        }
        _write_json(self.summary_path, summary)
        if self.logger is not None:
            self.logger.info(
                "Loop recorder finalized for '%s': rounds=%s final_best=%.6f stop=%s",
                self.label,
                summary["round_count"],
                summary["final_best_so_far"],
                summary["stopping_reason"],
            )
        return summary

    def _write_best_so_far_svg(self, path: Path, trajectory: list[dict[str, float]]) -> None:
        width = 760
        height = 420
        margin = 48
        xs = [point["step"] for point in trajectory]
        ys = [point["best_so_far"] for point in trajectory]
        min_x = min(xs) if xs else 0.0
        max_x = max(xs) if xs else 1.0
        min_y = min(ys) if ys else 0.0
        max_y = max(ys) if ys else 1.0
        span_x = max(max_x - min_x, 1.0)
        span_y = max(max_y - min_y, 1e-6)

        def map_x(value: float) -> float:
            return margin + ((value - min_x) / span_x) * (width - 2 * margin)

        def map_y(value: float) -> float:
            return height - margin - ((value - min_y) / span_y) * (height - 2 * margin)

        points = " ".join(f"{map_x(point['step']):.1f},{map_y(point['best_so_far']):.1f}" for point in trajectory)
        markers = "".join(
            f"<circle cx='{map_x(point['step']):.1f}' cy='{map_y(point['best_so_far']):.1f}' r='3.5' fill='#1d4ed8' />"
            for point in trajectory
        )
        svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>
<rect width='100%' height='100%' fill='white' />
<text x='{width / 2:.1f}' y='28' text-anchor='middle' font-size='18' font-family='Arial'>{self.label} best-so-far trajectory</text>
<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' stroke='black' />
<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height - margin}' stroke='black' />
<polyline fill='none' stroke='#1d4ed8' stroke-width='2.5' points='{points}' />
{markers}
<text x='{width - margin}' y='{height - margin + 24}' text-anchor='end' font-size='11'>round</text>
<text x='{margin - 8}' y='{margin - 10}' text-anchor='start' font-size='11'>fitness</text>
</svg>"""
        path.write_text(svg, encoding="utf-8")
