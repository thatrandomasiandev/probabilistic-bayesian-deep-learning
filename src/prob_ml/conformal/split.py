"""Split conformal prediction for regression."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from prob_ml.models.trainer import TrainConfig, train_regressor


@dataclass
class SplitConformalResult:
    y_hat: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    quantile: float
    alpha: float


def split_conformal_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    alpha: float = 0.1,
    config: TrainConfig | None = None,
) -> SplitConformalResult:
    """
    Split conformal prediction with absolute residual scores.

    Uses a held-out calibration set to compute the (1-alpha) quantile
    of nonconformity scores |y - y_hat|, then forms symmetric intervals.
    """
    import torch

    model = train_regressor(X_train, y_train, config)
    model.eval()
    with torch.no_grad():
        y_hat_cal = (
            model(torch.as_tensor(X_cal, dtype=torch.float32)).cpu().numpy()
        )
        y_hat_test = (
            model(torch.as_tensor(X_test, dtype=torch.float32)).cpu().numpy()
        )

    scores = np.abs(y_cal - y_hat_cal)
    n_cal = len(scores)
    q_level = min(1.0, np.ceil((n_cal + 1) * (1 - alpha)) / n_cal)
    quantile = float(np.quantile(scores, q_level, method="higher"))

    lower = y_hat_test - quantile
    upper = y_hat_test + quantile

    return SplitConformalResult(
        y_hat=y_hat_test,
        lower=lower,
        upper=upper,
        quantile=quantile,
        alpha=alpha,
    )
