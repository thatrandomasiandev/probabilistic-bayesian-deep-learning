"""Calibration metrics for probabilistic classifiers."""

from __future__ import annotations

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error."""
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        if not np.any(mask):
            continue
        acc = np.mean(y_true[mask])
        conf = np.mean(probs[mask])
        total += np.abs(acc - conf) * np.sum(mask) / len(probs)
    return float(total)


def mce(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """Maximum calibration error across bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    max_err = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        if not np.any(mask):
            continue
        acc = np.mean(y_true[mask])
        conf = np.mean(probs[mask])
        max_err = max(max_err, abs(acc - conf))
    return float(max_err)


def brier_score(probs: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean((probs - y_true) ** 2))


def nll_binary(probs: np.ndarray, y_true: np.ndarray, eps: float = 1e-8) -> float:
    p = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def reliability_bins(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 15,
) -> dict[str, np.ndarray]:
    """Bin-wise accuracy and confidence for reliability diagrams."""
    bins = np.linspace(0, 1, n_bins + 1)
    confidences = []
    accuracies = []
    counts = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        if not np.any(mask):
            confidences.append(np.nan)
            accuracies.append(np.nan)
            counts.append(0)
            continue
        confidences.append(float(np.mean(probs[mask])))
        accuracies.append(float(np.mean(y_true[mask])))
        counts.append(int(np.sum(mask)))
    return {
        "confidence": np.array(confidences),
        "accuracy": np.array(accuracies),
        "counts": np.array(counts),
    }
