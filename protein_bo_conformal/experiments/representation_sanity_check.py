"""Run Day 4 representation sanity checks across selected benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_loader import load_dataset
from data.dataset_registry import resolve_dataset
from data.split import build_split
from main import build_run_config, create_run_layout, write_json, write_run_metadata
from representation.interface import build_encoder
from utils.config import load_config
from utils.device import resolve_device
from utils.logger import build_logger
from utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Day 4 representation sanity checks.")
    parser.add_argument("--config", type=str, default=None, help="Optional override config path.")
    parser.add_argument(
        "--datasets",
        type=str,
        default="flip.aav,flip2.alpha_amylase.one_to_many",
        help="Comma-separated registry names to validate.",
    )
    return parser.parse_args()


def _dataset_override(registry_name: str) -> dict[str, object]:
    spec = resolve_dataset(registry_name=registry_name)
    split_type = "low_resource"
    if spec.name == "flip.gb1":
        split_type = "mutation_extrapolation"
    elif spec.benchmark == "flip2":
        split_type = "predefined"
    elif spec.benchmark == "proteingym":
        split_type = "low_resource"
    return {
        "registry_name": spec.name,
        "benchmark": spec.benchmark,
        "task": spec.task,
        "split_type": split_type,
    }


def _embedding_summary(train_features: np.ndarray, candidate_features: np.ndarray) -> dict[str, Any]:
    train_mean = train_features.mean(axis=0)
    candidate_mean = candidate_features.mean(axis=0)
    return {
        "feature_dim": int(train_features.shape[1]),
        "train_count": int(train_features.shape[0]),
        "candidate_count": int(candidate_features.shape[0]),
        "train_feature_variance_mean": float(train_features.var(axis=0).mean()),
        "candidate_feature_variance_mean": float(candidate_features.var(axis=0).mean()),
        "train_candidate_centroid_l2": float(np.linalg.norm(train_mean - candidate_mean)),
    }


def _linear_probe_check(features: np.ndarray) -> dict[str, Any]:
    try:
        import torch  # type: ignore

        layer = torch.nn.Linear(features.shape[1], 4)
        outputs = layer(torch.from_numpy(features[: min(8, len(features))]))
        return {
            "backend": "torch",
            "output_shape": list(outputs.shape),
        }
    except Exception:
        weights = np.ones((features.shape[1], 4), dtype=np.float32)
        outputs = features[: min(8, len(features))] @ weights
        return {
            "backend": "numpy",
            "output_shape": list(outputs.shape),
        }


def main() -> int:
    args = parse_args()
    base_config = PROJECT_ROOT / "config" / "base.yaml"
    default_layers = [
        PROJECT_ROOT / "config" / "dataset.yaml",
        PROJECT_ROOT / "config" / "representation.yaml",
    ]
    override_config = Path(args.config).resolve() if args.config else None

    config = load_config(base_config, override_config, default_layer_paths=default_layers)
    config, config_hash = build_run_config(config, argparse.Namespace(name="day4-representation-sanity"))

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{timestamp}-day4-representation-sanity-{config_hash[:8]}"
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

    dataset_names = [item.strip() for item in args.datasets.split(",") if item.strip()]
    report: dict[str, Any] = {
        "run_id": run_id,
        "config_hash": config_hash,
        "seed_report": seed_report,
        "device_info": device_info.to_dict(),
        "datasets": {},
    }

    for dataset_name in dataset_names:
        dataset_config = config.dataset.to_dict()
        dataset_config.update(_dataset_override(dataset_name))
        spec = resolve_dataset(registry_name=dataset_name)
        bundle = load_dataset(spec, PROJECT_ROOT, logger=logger)
        split_result = build_split(
            bundle,
            dataset_config,
            processed_dir=PROJECT_ROOT / "data" / "processed",
            logger=logger,
        )
        train_sequences = [record.sequence for record in split_result.train_records]
        candidate_sequences = [record.sequence for record in split_result.candidate_records]
        sample_sequences = (train_sequences + candidate_sequences)[: min(8, len(train_sequences) + len(candidate_sequences))]

        encoder_reports: dict[str, Any] = {}
        for encoder_name in ("onehot", "esm"):
            representation_config = config.representation.to_dict()
            representation_config["name"] = encoder_name
            encoder = build_encoder(representation_config, logger=logger)

            train_features = encoder.encode(train_sequences)
            candidate_features = encoder.encode(candidate_sequences)
            sample_batch = encoder.batch_encode(sample_sequences)
            sample_single = np.vstack([encoder.encode([sequence])[0] for sequence in sample_sequences])
            if not np.allclose(sample_batch, sample_single):
                raise ValueError(f"Encoder '{encoder_name}' batch/single outputs diverged for '{dataset_name}'.")
            if np.isnan(train_features).any() or np.isnan(candidate_features).any():
                raise ValueError(f"Encoder '{encoder_name}' produced NaN values for '{dataset_name}'.")

            cached_encoder = build_encoder(representation_config, logger=logger)
            cached_features = cached_encoder.encode(sample_sequences)
            if not np.allclose(sample_batch, cached_features):
                raise ValueError(f"Encoder '{encoder_name}' cache reload diverged for '{dataset_name}'.")

            embedding_summary = _embedding_summary(train_features, candidate_features)
            metadata_payload = {
                "dataset_name": dataset_name,
                "split_id": split_result.split_id,
                "encoder": encoder_name,
                "encoder_stats": encoder.get_stats(),
                "cache_reload_stats": cached_encoder.get_stats(),
                "embedding_summary": embedding_summary,
                "linear_probe_check": _linear_probe_check(sample_batch),
            }
            if encoder_name == "esm" and hasattr(encoder, "backend_info"):
                metadata_payload["backend_info"] = encoder.backend_info()

            metadata_path = (
                PROJECT_ROOT
                / "data"
                / "processed"
                / "metadata"
                / f"representation_{dataset_name.replace('.', '_')}_{split_result.split_name}_{encoder_name}.json"
            )
            metadata_path.write_text(
                json.dumps(metadata_payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            encoder_reports[encoder_name] = {
                "metadata_path": str(metadata_path),
                **metadata_payload,
            }

        report["datasets"][dataset_name] = {
            "split_id": split_result.split_id,
            "dataset_summary": bundle.to_summary_dict(),
            "encoders": encoder_reports,
        }

    report_path = layout["artifacts_dir"] / "representation_sanity_check_report.json"
    write_json(report_path, report)
    logger.info("Representation sanity check report written to %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
