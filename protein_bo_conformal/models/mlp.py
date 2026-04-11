"""MLP surrogate model implementation."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from models.base_model import BaseSurrogateModel


def _activation_layer(name: str) -> nn.Module:
    normalized = name.strip().lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation '{name}'.")


class MLPRegressor(BaseSurrogateModel):
    """Simple configurable MLP that predicts scalar fitness."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        activation: str = "relu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__(input_dim=input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.activation = activation.strip().lower()
        self.dropout = float(dropout)

        layers: list[nn.Module] = []
        previous_dim = self.input_dim
        for _ in range(max(0, self.num_layers)):
            layers.append(nn.Linear(previous_dim, self.hidden_dim))
            layers.append(_activation_layer(self.activation))
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
            previous_dim = self.hidden_dim
        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        outputs = self.network(features)
        return outputs.squeeze(-1)

    def model_summary(self) -> dict[str, Any]:
        return {
            "name": "mlp",
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "activation": self.activation,
            "dropout": self.dropout,
        }
