"""Training loop helpers for surrogate models."""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import torch

from models.checkpoint import CheckpointManager
from models.ensemble import DeepEnsemble


def _target_summary(targets: np.ndarray) -> dict[str, float]:
    if targets.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(targets.shape[0]),
        "mean": float(targets.mean()),
        "std": float(targets.std()),
        "min": float(targets.min()),
        "max": float(targets.max()),
    }


def _residual_summary(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    residuals = predictions - targets
    return {
        "mae": float(np.abs(residuals).mean()),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "bias": float(residuals.mean()),
        "std": float(residuals.std()),
        "max_abs": float(np.abs(residuals).max()),
    }


class EnsembleTrainer:
    """Train deep ensembles with shared configuration and logging."""

    def __init__(
        self,
        config: dict[str, Any],
        device: str = "cpu",
        logger: Any | None = None,
        checkpoint_manager: CheckpointManager | None = None,
    ) -> None:
        self.config = dict(config)
        self.device = device
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager

    def fit(
        self,
        ensemble: DeepEnsemble,
        train_features: np.ndarray,
        train_targets: np.ndarray,
        validation_features: np.ndarray | None = None,
        validation_targets: np.ndarray | None = None,
        round_index: int = 0,
        split_id: str = "unknown",
        round_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if train_features.ndim != 2:
            raise ValueError("train_features must be a rank-2 matrix.")
        if train_targets.ndim != 1:
            raise ValueError("train_targets must be a rank-1 vector.")
        if train_features.shape[0] != train_targets.shape[0]:
            raise ValueError("train_features and train_targets must have the same row count.")

        rng = np.random.default_rng(int(self.config.get("seed", 7)) + 5000 + int(round_index))
        validation_fraction = float(self.config.get("validation_fraction", 0.2))

        if validation_features is None or validation_targets is None:
            train_features, train_targets, validation_features, validation_targets = self._split_validation(
                train_features,
                train_targets,
                validation_fraction=validation_fraction,
                rng=rng,
            )

        member_reports: list[dict[str, Any]] = []
        for member_index, member in enumerate(ensemble.members):
            member_rng = np.random.default_rng(int(self.config.get("seed", 7)) + 7919 * (member_index + 1) + round_index)
            member_features, member_targets = self._bootstrap_training_data(train_features, train_targets, member_rng)
            report = self._fit_single_member(
                model=member,
                train_features=member_features,
                train_targets=member_targets,
                validation_features=validation_features,
                validation_targets=validation_targets,
                member_index=member_index,
                round_index=round_index,
            )
            if self.checkpoint_manager is not None and bool(self.config.get("checkpoint_enabled", True)):
                checkpoint_path = self.checkpoint_manager.save_member(
                    model=member,
                    round_index=round_index,
                    member_index=member_index,
                    metadata={
                        "split_id": split_id,
                        "member_seed": ensemble.member_seeds[member_index] if member_index < len(ensemble.member_seeds) else None,
                        "member_report": report,
                    },
                )
                report["checkpoint_path"] = str(checkpoint_path)
            member_reports.append(report)

        summary = {
            "round_index": int(round_index),
            "split_id": split_id,
            "train_data_summary": _target_summary(train_targets),
            "validation_data_summary": _target_summary(validation_targets),
            "feature_dim": int(train_features.shape[1]),
            "ensemble_summary": ensemble.model_summary(),
            "member_reports": member_reports,
            "aggregate": self._aggregate_member_reports(member_reports),
            "round_metadata": round_metadata or {},
        }
        ensemble.record_training_round(summary)
        if self.checkpoint_manager is not None:
            summary["training_summary_path"] = str(
                self.checkpoint_manager.save_training_summary(round_index, summary)
            )
        return summary

    def _split_validation(
        self,
        train_features: np.ndarray,
        train_targets: np.ndarray,
        validation_fraction: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if train_features.shape[0] < 4 or validation_fraction <= 0.0:
            return train_features, train_targets, train_features.copy(), train_targets.copy()

        count = train_features.shape[0]
        indices = np.arange(count)
        rng.shuffle(indices)
        validation_size = max(1, min(count - 1, int(round(count * validation_fraction))))
        validation_indices = np.sort(indices[:validation_size])
        train_indices = np.sort(indices[validation_size:])
        return (
            train_features[train_indices],
            train_targets[train_indices],
            train_features[validation_indices],
            train_targets[validation_indices],
        )

    def _bootstrap_training_data(
        self,
        train_features: np.ndarray,
        train_targets: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not bool(self.config.get("bootstrap_members", True)) or train_features.shape[0] <= 1:
            return train_features, train_targets
        indices = rng.integers(0, train_features.shape[0], size=train_features.shape[0])
        return train_features[indices], train_targets[indices]

    def _fit_single_member(
        self,
        model: torch.nn.Module,
        train_features: np.ndarray,
        train_targets: np.ndarray,
        validation_features: np.ndarray,
        validation_targets: np.ndarray,
        member_index: int,
        round_index: int,
    ) -> dict[str, Any]:
        member_seed = int(self.config.get("seed", 7)) + 1543 * (member_index + 1) + 10007 * int(round_index)
        torch.manual_seed(member_seed)
        learning_rate = float(self.config.get("learning_rate", 1e-3))
        weight_decay = float(self.config.get("weight_decay", 0.0))
        num_epochs = max(1, int(self.config.get("num_epochs", 80)))
        batch_size = max(1, int(self.config.get("batch_size", 32)))
        patience = max(1, int(self.config.get("early_stopping_patience", 10)))
        min_delta = float(self.config.get("early_stopping_min_delta", 1e-4))
        gradient_clip_norm = float(self.config.get("gradient_clip_norm", 0.0))

        model.to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        criterion = torch.nn.MSELoss()

        train_x = torch.as_tensor(train_features, dtype=torch.float32, device=self.device)
        train_y = torch.as_tensor(train_targets, dtype=torch.float32, device=self.device)
        val_x = torch.as_tensor(validation_features, dtype=torch.float32, device=self.device)
        val_y = torch.as_tensor(validation_targets, dtype=torch.float32, device=self.device)

        best_state = copy.deepcopy(model.state_dict())
        best_val_loss = math.inf
        best_epoch = 0
        epochs_without_improvement = 0
        history: list[dict[str, float]] = []

        for epoch in range(1, num_epochs + 1):
            model.train()
            permutation = torch.randperm(train_x.shape[0], device=self.device)
            epoch_losses: list[float] = []
            for start in range(0, train_x.shape[0], batch_size):
                indices = permutation[start : start + batch_size]
                batch_x = train_x[indices]
                batch_y = train_y[indices]
                optimizer.zero_grad(set_to_none=True)
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                loss.backward()
                if gradient_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))

            train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            val_loss = self._evaluate_loss(model, val_x, val_y, criterion)
            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": train_loss,
                    "validation_loss": val_loss,
                }
            )

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if self.logger is not None and (epoch == 1 or epoch == num_epochs or epoch % 10 == 0):
                self.logger.info(
                    "Round %s member %s epoch %s/%s train_loss=%.6f val_loss=%.6f",
                    round_index,
                    member_index,
                    epoch,
                    num_epochs,
                    train_loss,
                    val_loss,
                )

            if epochs_without_improvement >= patience:
                break

        model.load_state_dict(best_state)
        train_predictions = model.predict(train_features, batch_size=batch_size)
        validation_predictions = model.predict(validation_features, batch_size=batch_size)
        report = {
            "member_index": int(member_index),
            "member_seed": int(member_seed),
            "best_epoch": int(best_epoch),
            "epochs_ran": int(len(history)),
            "best_validation_loss": float(best_val_loss),
            "history": history,
            "train_error": _residual_summary(train_predictions, train_targets),
            "validation_error": _residual_summary(validation_predictions, validation_targets),
        }
        return report

    def _evaluate_loss(
        self,
        model: torch.nn.Module,
        features: torch.Tensor,
        targets: torch.Tensor,
        criterion: torch.nn.Module,
    ) -> float:
        model.eval()
        with torch.no_grad():
            predictions = model(features)
            loss = criterion(predictions, targets)
        return float(loss.detach().cpu())

    def _aggregate_member_reports(self, member_reports: list[dict[str, Any]]) -> dict[str, Any]:
        if not member_reports:
            return {}
        return {
            "mean_best_validation_loss": float(
                np.mean([report["best_validation_loss"] for report in member_reports])
            ),
            "mean_train_mae": float(
                np.mean([report["train_error"]["mae"] for report in member_reports])
            ),
            "mean_validation_mae": float(
                np.mean([report["validation_error"]["mae"] for report in member_reports])
            ),
            "mean_epochs_ran": float(np.mean([report["epochs_ran"] for report in member_reports])),
        }
