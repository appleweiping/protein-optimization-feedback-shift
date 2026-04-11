"""Unit tests for Day 5 surrogate infrastructure."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.checkpoint import CheckpointManager
from models.ensemble import DeepEnsemble
from models.mlp import MLPRegressor
from models.trainer import EnsembleTrainer


class SurrogateTests(unittest.TestCase):
    def test_mlp_predict_shape(self) -> None:
        model = MLPRegressor(input_dim=6, hidden_dim=8, num_layers=2, dropout=0.0)
        features = np.random.default_rng(7).normal(size=(5, 6)).astype(np.float32)
        predictions = model.predict(features)
        self.assertEqual(predictions.shape, (5,))

    def test_ensemble_predict_with_uncertainty_shapes(self) -> None:
        config = {
            "base_model": "mlp",
            "hidden_dim": 8,
            "num_layers": 1,
            "dropout": 0.0,
            "ensemble_size": 3,
            "seed": 7,
        }
        ensemble = DeepEnsemble.from_config(config=config, input_dim=4, device="cpu")
        features = np.random.default_rng(1).normal(size=(7, 4)).astype(np.float32)
        result = ensemble.predict_with_uncertainty(features, batch_size=4)
        self.assertEqual(result["mean"].shape, (7,))
        self.assertEqual(result["sigma"].shape, (7,))
        self.assertEqual(result["member_predictions"].shape, (3, 7))
        self.assertTrue((result["sigma"] >= 0.0).all())

    def test_checkpoint_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = MLPRegressor(input_dim=4, hidden_dim=8, num_layers=1, dropout=0.0)
            manager = CheckpointManager(tmp_dir)
            path = manager.save_member(model=model, round_index=2, member_index=1, metadata={"tag": "demo"})
            reloaded = MLPRegressor(input_dim=4, hidden_dim=8, num_layers=1, dropout=0.0)
            metadata = manager.load_member(reloaded, path)
            self.assertEqual(metadata["tag"], "demo")
            for left, right in zip(model.state_dict().values(), reloaded.state_dict().values()):
                self.assertTrue(np.allclose(left.detach().cpu().numpy(), right.detach().cpu().numpy()))

    def test_trainer_fit_returns_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            rng = np.random.default_rng(11)
            features = rng.normal(size=(20, 5)).astype(np.float32)
            weights = rng.normal(size=(5,)).astype(np.float32)
            targets = (features @ weights + 0.1 * rng.normal(size=(20,))).astype(np.float32)
            config = {
                "base_model": "mlp",
                "hidden_dim": 16,
                "num_layers": 1,
                "dropout": 0.0,
                "ensemble_size": 2,
                "seed": 7,
                "learning_rate": 1e-2,
                "weight_decay": 0.0,
                "num_epochs": 5,
                "batch_size": 4,
                "validation_fraction": 0.2,
                "early_stopping_patience": 3,
                "checkpoint_enabled": True,
                "bootstrap_members": True,
            }
            ensemble = DeepEnsemble.from_config(config=config, input_dim=5, device="cpu")
            trainer = EnsembleTrainer(
                config=config,
                device="cpu",
                checkpoint_manager=CheckpointManager(tmp_dir),
            )
            summary = trainer.fit(
                ensemble=ensemble,
                train_features=features,
                train_targets=targets,
                round_index=0,
                split_id="synthetic",
            )
            self.assertEqual(len(summary["member_reports"]), 2)
            self.assertIn("aggregate", summary)
            self.assertIn("training_summary_path", summary)


if __name__ == "__main__":
    unittest.main()
