"""Base model abstractions for surrogate learning."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from torch import nn


class BaseSurrogateModel(nn.Module, ABC):
    """Common interface for all surrogate base learners."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)

    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return scalar fitness predictions for a batch of features."""

    @abstractmethod
    def model_summary(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the model architecture."""

    def _coerce_features(self, features: np.ndarray | torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        if isinstance(features, torch.Tensor):
            return features.to(device=device, dtype=torch.float32)
        return torch.as_tensor(features, dtype=torch.float32, device=device)

    def predict(
        self,
        features: np.ndarray | torch.Tensor,
        batch_size: int = 256,
    ) -> np.ndarray:
        """Return mean predictions as a NumPy array."""
        tensor = self._coerce_features(features)
        if tensor.ndim != 2:
            raise ValueError(f"Model inputs must be rank-2, received shape {tuple(tensor.shape)}.")
        outputs: list[torch.Tensor] = []
        self.eval()
        with torch.no_grad():
            for start in range(0, tensor.shape[0], max(1, int(batch_size))):
                batch = tensor[start : start + batch_size]
                outputs.append(self.forward(batch).detach().cpu())
        if not outputs:
            return np.zeros((0,), dtype=np.float32)
        return torch.cat(outputs, dim=0).numpy().astype(np.float32, copy=False)

    def predict_with_uncertainty(
        self,
        features: np.ndarray | torch.Tensor,
        batch_size: int = 256,
    ) -> dict[str, np.ndarray]:
        """Return deterministic predictions plus zero uncertainty for a single model."""
        mean = self.predict(features, batch_size=batch_size)
        return {
            "mean": mean,
            "sigma": np.zeros_like(mean, dtype=np.float32),
        }


def build_base_model(config: dict[str, Any], input_dim: int) -> BaseSurrogateModel:
    """Construct a configured base learner."""
    base_model = str(config.get("base_model", "mlp")).strip().lower()
    if base_model != "mlp":
        raise ValueError(f"Unsupported base model '{base_model}'.")

    from models.mlp import MLPRegressor

    return MLPRegressor(
        input_dim=input_dim,
        hidden_dim=int(config.get("hidden_dim", 256)),
        num_layers=int(config.get("num_layers", 3)),
        activation=str(config.get("activation", "relu")),
        dropout=float(config.get("dropout", 0.1)),
    )
