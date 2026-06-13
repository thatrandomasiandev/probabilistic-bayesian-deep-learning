"""Deep ensemble uncertainty for regression."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from prob_ml.models.mlp import RegressionMLP
from prob_ml.models.trainer import TrainConfig, train_regressor


@dataclass
class DeepEnsembleResult:
    mean: np.ndarray
    std: np.ndarray
    epistemic: np.ndarray
    aleatoric: np.ndarray
    member_predictions: np.ndarray


def _predict(model: RegressionMLP, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(X, dtype=torch.float32)).cpu().numpy()


def fit_deep_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_members: int = 5,
    config: TrainConfig | None = None,
    seed: int = 42,
) -> DeepEnsembleResult:
    """
    Train a deep ensemble and decompose epistemic vs aleatoric uncertainty.

    Epistemic = std across ensemble members.
    Aleatoric = mean within-member residual std on training data.
    """
    cfg = config or TrainConfig()
    member_preds = []
    aleatoric_vals = []

    for i in range(n_members):
        member_cfg = TrainConfig(
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            hidden_dim=cfg.hidden_dim,
            n_hidden=cfg.n_hidden,
            dropout=cfg.dropout,
            seed=seed + i,
        )
        model = train_regressor(X_train, y_train, member_cfg)
        member_preds.append(_predict(model, X_test))
        residuals = y_train - _predict(model, X_train)
        aleatoric_vals.append(float(np.std(residuals)))

    member_preds_arr = np.stack(member_preds, axis=0)
    mean = member_preds_arr.mean(axis=0)
    epistemic = member_preds_arr.std(axis=0)
    aleatoric_scalar = float(np.mean(aleatoric_vals))
    aleatoric = np.full(len(X_test), aleatoric_scalar)
    total = np.sqrt(epistemic**2 + aleatoric**2)

    return DeepEnsembleResult(
        mean=mean,
        std=total,
        epistemic=epistemic,
        aleatoric=aleatoric,
        member_predictions=member_preds_arr,
    )
