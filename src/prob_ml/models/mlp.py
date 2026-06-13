"""PyTorch MLP backbones for regression and classification."""

from __future__ import annotations

import torch
import torch.nn as nn


class _MLPBackbone(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        n_hidden: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(n_hidden):
            layers.extend(
                [
                    nn.Linear(dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                ]
            )
            dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.out_dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class RegressionMLP(nn.Module):
    """Single-output MLP for regression."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        n_hidden: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = _MLPBackbone(in_dim, hidden_dim, n_hidden, dropout)
        self.head = nn.Linear(self.backbone.out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)).squeeze(-1)


class ClassificationMLP(nn.Module):
    """Binary classification MLP returning logits."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        n_hidden: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = _MLPBackbone(in_dim, hidden_dim, n_hidden, dropout)
        self.head = nn.Linear(self.backbone.out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)).squeeze(-1)
