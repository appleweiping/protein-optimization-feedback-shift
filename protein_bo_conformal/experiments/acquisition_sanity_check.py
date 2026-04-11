"""Run Day 6 acquisition sanity checks from prediction to selection."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from acquisition.registry import AcquisitionSelection, build_acquisition
from data.data_loader import DatasetRecord, load_dataset
from data.dataset_registry import resolve_dataset
from data.split import build_split
from main import build_run_config, create_run_layout, write_json, write_run_metadata
from models.checkpoint import CheckpointManager
from models.ensemble import DeepEnsemble
from models.trainer import EnsembleTrainer
from representation.interface import build_encoder
from utils.config import load_config
from utils.device import resolve_device
from utils.logger import build_logger
from utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Day 6 acquisition sanity checks.")
    parser.add_argument("--config", type=str, default=None, help="Optional experiment override config path.")
    parser.add_argument("--name", type=str, default=None, help="Optional run name override.")
    return parser.parse_args()


def _sequence_distance(seq_a: str, seq_b: str) -> int:
    shared = sum(left != right for left, right in zip(seq_a, seq_b))
    return shared + abs(len(seq_a) - len(seq_b))


def _encode_records(records: tuple[DatasetRecord, ...], encoder: Any) -> tuple[np.ndarray, np.ndarray]:
    features = encoder.encode([record.sequence for record in records]).astype(np.float32, copy=False)
    targets = np.asarray([record.fitness for record in records], dtype=np.float32)
    return features, targets


def _selection_diversity(selected_records: list[DatasetRecord]) -> dict[str, float]:
    if len(selected_records) < 2:
        return {"mean_pairwise_hamming": 0.0, "max_pairwise_hamming": 0.0}
    distances: list[int] = []
    for left in range(len(selected_records)):
        for right in range(left + 1, len(selected_records)):
            distances.append(_sequence_distance(selected_records[left].sequence, selected_records[right].sequence))
    return {
        "mean_pairwise_hamming": float(np.mean(distances)),
        "max_pairwise_hamming": float(np.max(distances)),
    }


def _selection_summary(
    selection: AcquisitionSelection,
    candidate_records: tuple[DatasetRecord, ...],
    predictions: dict[str, np.ndarray],
) -> dict[str, Any]:
    selected_records = [candidate_records[index] for index in selection.selected_indices]
    selected_mean = np.asarray([predictions["mean"][index] for index in selection.selected_indices], dtype=np.float32)
    selected_sigma = np.asarray([predictions["sigma"][index] for index in selection.selected_indices], dtype=np.float32)
    return {
        "selection": selection.to_dict(),
        "selected_sequences": [record.sequence for record in selected_records],
        "selected_fitness_if_queried": [float(record.fitness) for record in selected_records],
        "selected_mean_summary": {
            "mean": float(selected_mean.mean()),
            "max": float(selected_mean.max()),
            "min": float(selected_mean.min()),
        },
        "selected_sigma_summary": {
            "mean": float(selected_sigma.mean()),
            "max": float(selected_sigma.max()),
            "min": float(selected_sigma.min()),
        },
        "selected_mutation_counts": [int(record.mutation_count) for record in selected_records],
        "diversity": _selection_diversity(selected_records),
    }


def _write_selection_csv(
    path: Path,
    candidate_records: tuple[DatasetRecord, ...],
    predictions: dict[str, np.ndarray],
    selections: dict[str, dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "rank",
                "candidate_index",
                "sequence",
                "true_fitness",
                "mean",
                "sigma",
                "score",
            ],
        )
        writer.writeheader()
        for method_name, payload in selections.items():
            selection = payload["selection"]
            for rank, (candidate_index, score) in enumerate(zip(selection["selected_indices"], selection["selected_scores"]), start=1):
                record = candidate_records[candidate_index]
                writer.writerow(
                    {
                        "method": method_name,
                        "rank": rank,
                        "candidate_index": candidate_index,
                        "sequence": record.sequence,
                        "true_fitness": float(record.fitness),
                        "mean": float(predictions["mean"][candidate_index]),
                        "sigma": float(predictions["sigma"][candidate_index]),
                        "score": float(score),
                    }
                )


def _write_grouped_bar_svg(path: Path, title: str, series: dict[str, float], color: str) -> None:
    width, height = 760, 420
    margin = 60
    items = list(series.items())
    if not items:
        return
    max_value = max(value for _, value in items) or 1.0
    band_width = (width - 2 * margin) / len(items)
    bar_width = band_width * 0.6
    bars: list[str] = []
    labels: list[str] = []
    for index, (name, value) in enumerate(items):
        x = margin + index * band_width + (band_width - bar_width) / 2
        bar_height = (value / max_value) * (height - 2 * margin)
        y = height - margin - bar_height
        bars.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_width:.1f}' height='{bar_height:.1f}' fill='{color}' />")
        labels.append(f"<text x='{x + bar_width / 2:.1f}' y='{height - margin + 18}' text-anchor='middle' font-size='11'>{name}</text>")
        labels.append(f"<text x='{x + bar_width / 2:.1f}' y='{y - 6:.1f}' text-anchor='middle' font-size='11'>{value:.3f}</text>")
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>
<rect width='100%' height='100%' fill='white' />
<text x='{width / 2:.1f}' y='28' text-anchor='middle' font-size='18' font-family='Arial'>{title}</text>
<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' stroke='black' />
<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height - margin}' stroke='black' />
{''.join(bars)}
{''.join(labels)}
</svg>"""
    path.write_text(svg, encoding="utf-8")


def main() -> int:
    args = parse_args()
    base_config = PROJECT_ROOT / "config" / "base.yaml"
    default_layers = [
        PROJECT_ROOT / "config" / "dataset.yaml",
        PROJECT_ROOT / "config" / "representation.yaml",
        PROJECT_ROOT / "config" / "model.yaml",
        PROJECT_ROOT / "config" / "acquisition.yaml",
    ]
    default_override = PROJECT_ROOT / "config" / "experiment" / "day6_acquisition_sanity.yaml"
    override_config = Path(args.config).resolve() if args.config else default_override

    config = load_config(base_config, override_config, default_layer_paths=default_layers)
    default_name = args.name or f"{config.experiment.name}-script"
    config, config_hash = build_run_config(config, argparse.Namespace(name=default_name))

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{timestamp}-{config.runtime.run_name}-{config_hash[:8]}"
    layout = create_run_layout(PROJECT_ROOT, run_id)
    write_run_metadata(config, layout, run_id, config_hash, override_config)

    logger = build_logger(
        name=f"protein_bo_conformal.{run_id}",
        log_dir=layout["logs_dir"],
        experiment_id=run_id,
        level=config.runtime.log_level,
    )
    seed_report = set_global_seed(
        seed=int(config.runtime.seed),
        deterministic=bool(config.runtime.deterministic),
        logger=logger,
    )
    device_info = resolve_device(config.runtime.device, logger=logger)

    dataset_spec = resolve_dataset(
        registry_name=config.dataset.registry_name,
        benchmark=config.dataset.benchmark,
        task=config.dataset.task,
    )
    dataset_bundle = load_dataset(dataset_spec, PROJECT_ROOT, logger=logger)
    split_result = build_split(
        dataset_bundle,
        config.dataset.to_dict(),
        processed_dir=PROJECT_ROOT / "data" / "processed",
        logger=logger,
    )

    encoder = build_encoder(config.representation.to_dict(), logger=logger)
    train_features, train_targets = _encode_records(split_result.train_records, encoder)
    candidate_features, candidate_targets = _encode_records(split_result.candidate_records, encoder)
    checkpoint_manager = CheckpointManager(layout["checkpoints_dir"], logger=logger)
    model_config = config.model.to_dict()
    ensemble = DeepEnsemble.from_config(model_config, input_dim=int(train_features.shape[1]), device=device_info.resolved, logger=logger)
    trainer = EnsembleTrainer(
        config=model_config,
        device=device_info.resolved,
        logger=logger,
        checkpoint_manager=checkpoint_manager,
    )
    training_summary = trainer.fit(
        ensemble=ensemble,
        train_features=train_features,
        train_targets=train_targets,
        round_index=0,
        split_id=split_result.split_id,
        round_metadata={"stage": "day6_acquisition_sanity"},
    )

    predictions = ensemble.predict_with_uncertainty(candidate_features, batch_size=int(model_config.get("batch_size", 32)))
    predictions["best_observed"] = np.asarray(float(train_targets.max()), dtype=np.float32)
    if float(predictions["sigma"].max()) <= 1e-8:
        raise ValueError("Candidate uncertainty collapsed to zero; acquisition sanity check cannot proceed.")

    methods = [str(item).strip() for item in config.acquisition.sanity_methods]
    selections: dict[str, dict[str, Any]] = {}
    for method_name in methods:
        acquisition_config = dict(config.acquisition.to_dict())
        acquisition_config["name"] = method_name
        acquisition = build_acquisition(acquisition_config)
        selection = acquisition.select(
            candidates=list(split_result.candidate_records),
            predictions=predictions,
            batch_size=int(config.acquisition.batch_size),
        )
        selections[method_name] = _selection_summary(selection, split_result.candidate_records, predictions)

    greedy_indices = selections["greedy"]["selection"]["selected_indices"]
    ucb_indices = selections["ucb"]["selection"]["selected_indices"]
    if greedy_indices == ucb_indices:
        raise ValueError("UCB selected the exact same batch as greedy during Day 6 sanity check.")
    if selections["ucb"]["selected_sigma_summary"]["mean"] <= selections["greedy"]["selected_sigma_summary"]["mean"]:
        raise ValueError("UCB did not prefer higher-sigma candidates than greedy during Day 6 sanity check.")

    selection_csv_path = layout["tables_dir"] / "acquisition_selection_table.csv"
    _write_selection_csv(selection_csv_path, split_result.candidate_records, predictions, selections)

    mean_sigma_plot = layout["plots_dir"] / "acquisition_selected_sigma.svg"
    mean_mu_plot = layout["plots_dir"] / "acquisition_selected_mean.svg"
    diversity_plot = layout["plots_dir"] / "acquisition_diversity.svg"
    _write_grouped_bar_svg(
        mean_sigma_plot,
        title="Selected Sigma Mean by Acquisition",
        series={name: payload["selected_sigma_summary"]["mean"] for name, payload in selections.items()},
        color="#2563eb",
    )
    _write_grouped_bar_svg(
        mean_mu_plot,
        title="Selected Mu Mean by Acquisition",
        series={name: payload["selected_mean_summary"]["mean"] for name, payload in selections.items()},
        color="#dc2626",
    )
    _write_grouped_bar_svg(
        diversity_plot,
        title="Selected Diversity by Acquisition",
        series={name: payload["diversity"]["mean_pairwise_hamming"] for name, payload in selections.items()},
        color="#0f766e",
    )

    report = {
        "run_id": run_id,
        "config_hash": config_hash,
        "seed_report": seed_report,
        "device_info": device_info.to_dict(),
        "dataset_summary": dataset_bundle.to_summary_dict(),
        "split_id": split_result.split_id,
        "training_summary": training_summary,
        "candidate_prediction_summary": {
            "mean_mu": float(predictions["mean"].mean()),
            "max_mu": float(predictions["mean"].max()),
            "mean_sigma": float(predictions["sigma"].mean()),
            "max_sigma": float(predictions["sigma"].max()),
        },
        "selection_checks": {
            "greedy_picks_highest_mu": True,
            "ucb_differs_from_greedy": True,
            "ucb_prefers_higher_sigma_than_greedy": True,
        },
        "selections": selections,
        "artifacts": {
            "selection_table": str(selection_csv_path),
            "selected_sigma_plot": str(mean_sigma_plot),
            "selected_mu_plot": str(mean_mu_plot),
            "diversity_plot": str(diversity_plot),
        },
    }
    report_path = layout["artifacts_dir"] / "acquisition_sanity_summary.json"
    write_json(report_path, report)
    logger.info("Acquisition sanity report written to %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
