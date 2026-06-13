"""Uncertainty quantification metrics for regression."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def rmse(y_hat: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_hat - y_true) ** 2)))


def gaussian_nll(y_hat: np.ndarray, y_true: np.ndarray, sigma: np.ndarray) -> float:
    """Negative log-likelihood under Gaussian predictive distribution."""
    var = np.maximum(sigma**2, 1e-8)
    nll = 0.5 * np.log(2 * np.pi * var) + 0.5 * (y_true - y_hat) ** 2 / var
    return float(np.mean(nll))


def picp(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Prediction interval coverage probability."""
    covered = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(covered))


def mpiw(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean prediction interval width."""
    return float(np.mean(upper - lower))


def uncertainty_correlation(
    sigma_hat: np.ndarray,
    errors: np.ndarray,
) -> float:
    """Pearson correlation between predicted uncertainty and absolute errors."""
    if len(sigma_hat) < 2:
        return 0.0
    if np.std(sigma_hat) < 1e-12 or np.std(errors) < 1e-12:
        return 0.0
    return float(np.corrcoef(sigma_hat, np.abs(errors))[0, 1])


def gaussian_interval(
    mean: np.ndarray,
    std: np.ndarray,
    alpha: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric Gaussian prediction interval at 1 - alpha coverage."""
    z = norm.ppf(1 - alpha / 2)
    return mean - z * std, mean + z * std
