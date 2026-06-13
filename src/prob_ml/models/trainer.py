"""Training utilities for PyTorch models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from prob_ml.models.mlp import ClassificationMLP, RegressionMLP
from prob_ml.utils.seed import set_torch_seed


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 64
    n_hidden: int = 2
    dropout: float = 0.1
    seed: int = 42


def _to_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    x_t = torch.as_tensor(X, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(x_t, y_t), batch_size=batch_size, shuffle=shuffle)


def train_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: TrainConfig | None = None,
) -> RegressionMLP:
    """Fit a regression MLP on training data."""
    cfg = config or TrainConfig()
    set_torch_seed(cfg.seed)

    model = RegressionMLP(
        in_dim=X_train.shape[1],
        hidden_dim=cfg.hidden_dim,
        n_hidden=cfg.n_hidden,
        dropout=cfg.dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.MSELoss()
    loader = _to_loader(X_train, y_train, cfg.batch_size, shuffle=True)

    model.train()
    for _ in range(cfg.epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: TrainConfig | None = None,
) -> ClassificationMLP:
    """Fit a binary classification MLP on training data."""
    cfg = config or TrainConfig()
    set_torch_seed(cfg.seed)

    model = ClassificationMLP(
        in_dim=X_train.shape[1],
        hidden_dim=cfg.hidden_dim,
        n_hidden=cfg.n_hidden,
        dropout=cfg.dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    loader = _to_loader(X_train, y_train.astype(np.float32), cfg.batch_size, shuffle=True)

    model.train()
    for _ in range(cfg.epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    return model
