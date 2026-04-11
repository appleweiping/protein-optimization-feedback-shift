"""Deep ensemble utilities for surrogate uncertainty estimation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from models.base_model import BaseSurrogateModel, build_base_model


class DeepEnsemble:
    """Collection of independently initialized surrogate members."""

    def __init__(
        self,
        members: list[BaseSurrogateModel],
        config: dict[str, Any],
        device: str = "cpu",
        logger: Any | None = None,
        member_seeds: list[int] | None = None,
    ) -> None:
        self.members = list(members)
        self.config = dict(config)
        self.device = device
        self.logger = logger
        self.member_seeds = list(member_seeds or [])
        self.round_history: list[dict[str, Any]] = []
        for member in self.members:
            member.to(self.device)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        input_dim: int,
        device: str = "cpu",
        logger: Any | None = None,
    ) -> "DeepEnsemble":
        ensemble_size = max(1, int(config.get("ensemble_size", 5)))
        base_seed = int(config.get("seed", 7))
        members: list[BaseSurrogateModel] = []
        member_seeds: list[int] = []
        for member_index in range(ensemble_size):
            member_seed = base_seed + 1009 * member_index
            previous_state = torch.random.get_rng_state()
            try:
                torch.manual_seed(member_seed)
                members.append(build_base_model(config=config, input_dim=input_dim))
            finally:
                torch.random.set_rng_state(previous_state)
            member_seeds.append(member_seed)
        return cls(
            members=members,
            config=config,
            device=device,
            logger=logger,
            member_seeds=member_seeds,
        )

    @property
    def ensemble_size(self) -> int:
        return len(self.members)

    @property
    def input_dim(self) -> int:
        return self.members[0].input_dim if self.members else 0

    def member(self, index: int) -> BaseSurrogateModel:
        return self.members[index]

    def model_summary(self) -> dict[str, Any]:
        return {
            "ensemble_size": self.ensemble_size,
            "device": self.device,
            "member_seeds": list(self.member_seeds),
            "base_model_summary": self.members[0].model_summary() if self.members else {},
        }

    def record_training_round(self, payload: dict[str, Any]) -> None:
        self.round_history.append(dict(payload))

    def predict_members(self, features: np.ndarray, batch_size: int = 256) -> np.ndarray:
        if features.ndim != 2:
            raise ValueError(f"Expected feature matrix with rank 2, received shape {features.shape}.")
        member_predictions = [
            member.predict(features, batch_size=batch_size)
            for member in self.members
        ]
        if not member_predictions:
            return np.zeros((0, features.shape[0]), dtype=np.float32)
        return np.stack(member_predictions, axis=0).astype(np.float32, copy=False)

    def predict(self, features: np.ndarray, batch_size: int = 256) -> np.ndarray:
        member_predictions = self.predict_members(features, batch_size=batch_size)
        if member_predictions.size == 0:
            return np.zeros((features.shape[0],), dtype=np.float32)
        return member_predictions.mean(axis=0).astype(np.float32, copy=False)

    def predict_with_uncertainty(self, features: np.ndarray, batch_size: int = 256) -> dict[str, np.ndarray]:
        member_predictions = self.predict_members(features, batch_size=batch_size)
        if member_predictions.size == 0:
            zeros = np.zeros((features.shape[0],), dtype=np.float32)
            return {
                "mean": zeros,
                "sigma": zeros,
                "member_predictions": member_predictions,
            }
        mean = member_predictions.mean(axis=0)
        sigma = member_predictions.std(axis=0, ddof=0)
        return {
            "mean": mean.astype(np.float32, copy=False),
            "sigma": sigma.astype(np.float32, copy=False),
            "member_predictions": member_predictions.astype(np.float32, copy=False),
        }
