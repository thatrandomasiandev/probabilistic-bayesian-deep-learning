"""Monte Carlo dropout uncertainty for regression."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from prob_ml.models.mlp import RegressionMLP
from prob_ml.models.trainer import TrainConfig, train_regressor


@dataclass
class MCDropoutResult:
    mean: np.ndarray
    std: np.ndarray
    epistemic: np.ndarray
    aleatoric: np.ndarray
    samples: np.ndarray


def _predict_samples(model: RegressionMLP, X: np.ndarray, n_samples: int) -> np.ndarray:
    model.train()
    x_t = torch.as_tensor(X, dtype=torch.float32)
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds.append(model(x_t).cpu().numpy())
    model.eval()
    return np.stack(preds, axis=0)


def mc_dropout_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_mc_samples: int = 30,
    aleatoric_std: float | None = None,
    config: TrainConfig | None = None,
) -> MCDropoutResult:
    """
    Train an MLP with dropout and estimate uncertainty via MC forward passes.

    Epistemic uncertainty = std across MC samples.
    Aleatoric uncertainty = constant or provided per-sample noise estimate.
    """
    cfg = config or TrainConfig(dropout=0.2)
    model = train_regressor(X_train, y_train, cfg)
    samples = _predict_samples(model, X_test, n_mc_samples)

    mean = samples.mean(axis=0)
    epistemic = samples.std(axis=0)
    if aleatoric_std is None:
        residuals = y_train - _predict_point(model, X_train)
        aleatoric = np.full(len(X_test), float(np.std(residuals)))
    elif isinstance(aleatoric_std, (int, float)):
        aleatoric = np.full(len(X_test), float(aleatoric_std))
    else:
        aleatoric = np.asarray(aleatoric_std, dtype=float)

    total = np.sqrt(epistemic**2 + aleatoric**2)
    return MCDropoutResult(
        mean=mean,
        std=total,
        epistemic=epistemic,
        aleatoric=aleatoric,
        samples=samples,
    )


def _predict_point(model: RegressionMLP, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(X, dtype=torch.float32)).cpu().numpy()
