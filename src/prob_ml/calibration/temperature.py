"""Post-hoc calibration methods for neural classifiers.

Includes temperature scaling, vector scaling (per-class affine), and
non-parametric histogram binning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from prob_ml.calibration.metrics import brier_score, ece, nll_binary

logger = logging.getLogger(__name__)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _softmax(z: np.ndarray) -> np.ndarray:
    shifted = z - z.max(axis=-1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / exp_z.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Temperature Scaling
# ---------------------------------------------------------------------------


@dataclass
class TemperatureScalingResult:
    temperature: float
    probs_calibrated: np.ndarray
    ece_before: float
    ece_after: float
    nll_before: float
    nll_after: float
    brier_before: float
    brier_after: float


class _Temperature(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.log_temp = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temp)


def fit_temperature_scaling(
    logits_val: np.ndarray,
    y_val: np.ndarray,
    logits_eval: np.ndarray | None = None,
    y_eval: np.ndarray | None = None,
    n_bins: int = 15,
    max_iter: int = 200,
    lr: float = 0.01,
) -> TemperatureScalingResult:
    """Fit a scalar temperature T on validation logits to minimize NLL.

    Calibrated probability: sigmoid(logit / T).
    Metrics are computed on ``logits_eval`` / ``y_eval`` when provided,
    else on the validation set itself.

    Args:
        logits_val: Raw logits on the validation split ``(n_val,)``.
        y_val: Binary labels ``(n_val,)``.
        logits_eval: Optional evaluation logits ``(n_eval,)``.
        y_eval: Optional evaluation labels ``(n_eval,)``.
        n_bins: Number of bins for ECE computation.
        max_iter: Maximum L-BFGS iterations.
        lr: L-BFGS learning rate.

    Returns:
        ``TemperatureScalingResult`` with fitted temperature and before/after
        calibration metrics.
    """
    logits_val_t = torch.as_tensor(logits_val, dtype=torch.float32)
    y_val_t = torch.as_tensor(y_val, dtype=torch.float32)
    temp_module = _Temperature()
    optimizer = torch.optim.LBFGS(temp_module.parameters(), lr=lr, max_iter=max_iter)
    criterion = nn.BCEWithLogitsLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        scaled = logits_val_t / temp_module.temperature
        loss = criterion(scaled, y_val_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(temp_module.temperature.detach().cpu().item())

    logits_report = logits_eval if logits_eval is not None else logits_val
    y_report = y_eval if y_eval is not None else y_val

    probs_before = _sigmoid(logits_report)
    probs_after = _sigmoid(logits_report / temperature)

    return TemperatureScalingResult(
        temperature=temperature,
        probs_calibrated=probs_after,
        ece_before=ece(probs_before, y_report, n_bins=n_bins),
        ece_after=ece(probs_after, y_report, n_bins=n_bins),
        nll_before=nll_binary(probs_before, y_report),
        nll_after=nll_binary(probs_after, y_report),
        brier_before=brier_score(probs_before, y_report),
        brier_after=brier_score(probs_after, y_report),
    )


# ---------------------------------------------------------------------------
# Vector Scaling (per-class affine calibration)
# ---------------------------------------------------------------------------


@dataclass
class VectorScalingResult:
    """Result container for vector scaling calibration.

    Attributes:
        W: Learned per-class weight vector ``(n_classes,)``.
        b: Learned per-class bias vector ``(n_classes,)``.
        probs_calibrated: Calibrated probabilities ``(n, n_classes)``.
        nll_before: NLL before calibration.
        nll_after: NLL after calibration.
    """

    W: np.ndarray
    b: np.ndarray
    probs_calibrated: np.ndarray
    nll_before: float
    nll_after: float


class _VectorScaleModule(nn.Module):
    """Learnable per-class affine transform: z_cal = W * z + b."""

    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.ones(n_classes))
        self.b = nn.Parameter(torch.zeros(n_classes))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits * self.W + self.b


def vector_scaling(
    logits_val: np.ndarray,
    y_val: np.ndarray,
    logits_eval: np.ndarray | None = None,
    n_classes: int | None = None,
    max_iter: int = 300,
    lr: float = 0.01,
) -> VectorScalingResult:
    """Learn a per-class affine calibration map z_cal = W * z + b.

    Unlike scalar temperature scaling, vector scaling learns independent
    scale and shift parameters for each class logit, providing strictly
    more expressive recalibration at the cost of more parameters.

    Args:
        logits_val: Validation logits ``(n_val, n_classes)``.
        y_val: Integer class labels ``(n_val,)``.
        logits_eval: Optional evaluation logits. If ``None``, the validation
            logits are used for the output probabilities.
        n_classes: Number of classes. Inferred from ``logits_val`` if ``None``.
        max_iter: Maximum L-BFGS iterations.
        lr: L-BFGS learning rate.

    Returns:
        ``VectorScalingResult`` with learned parameters and calibrated probs.
    """
    if logits_val.ndim == 1:
        raise ValueError("vector_scaling requires multiclass logits (n, C)")
    if n_classes is None:
        n_classes = logits_val.shape[1]

    logits_t = torch.as_tensor(logits_val, dtype=torch.float32)
    y_t = torch.as_tensor(y_val, dtype=torch.long)

    module = _VectorScaleModule(n_classes)
    optimizer = torch.optim.LBFGS(module.parameters(), lr=lr, max_iter=max_iter)
    criterion = nn.CrossEntropyLoss()

    probs_before = _softmax(logits_val)
    nll_before = float(-np.mean(
        np.log(np.clip(probs_before[np.arange(len(y_val)), y_val.astype(int)], 1e-8, 1.0))
    ))

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = criterion(module(logits_t), y_t)
        loss.backward()
        return loss

    optimizer.step(closure)

    W_np = module.W.detach().cpu().numpy()
    b_np = module.b.detach().cpu().numpy()

    eval_logits = logits_eval if logits_eval is not None else logits_val
    calibrated = eval_logits * W_np + b_np
    probs_cal = _softmax(calibrated)

    if logits_eval is not None:
        nll_after = float("nan")
    else:
        nll_after = float(-np.mean(
            np.log(np.clip(probs_cal[np.arange(len(y_val)), y_val.astype(int)], 1e-8, 1.0))
        ))

    return VectorScalingResult(
        W=W_np,
        b=b_np,
        probs_calibrated=probs_cal,
        nll_before=nll_before,
        nll_after=nll_after,
    )


# ---------------------------------------------------------------------------
# Histogram Binning (non-parametric calibration)
# ---------------------------------------------------------------------------


@dataclass
class HistogramBinningResult:
    """Result container for histogram-binning calibration.

    Attributes:
        bin_edges: Bin boundary array ``(n_bins + 1,)``.
        bin_calibrated_probs: Calibrated probability assigned to each bin
            ``(n_bins,)``.
        probs_calibrated: Calibrated probabilities for the evaluation set.
        ece_before: ECE before calibration.
        ece_after: ECE after calibration.
    """

    bin_edges: np.ndarray
    bin_calibrated_probs: np.ndarray
    probs_calibrated: np.ndarray
    ece_before: float
    ece_after: float


def histogram_binning(
    probs_val: np.ndarray,
    y_val: np.ndarray,
    probs_eval: np.ndarray | None = None,
    n_bins: int = 15,
) -> HistogramBinningResult:
    """Non-parametric histogram-binning calibration (Zadrozny & Elkan, 2001).

    Partitions the predicted-probability range [0, 1] into equal-width bins
    and replaces each prediction with the empirical accuracy of its bin on
    the validation set.

    This is the simplest non-parametric calibration map and serves as a
    strong baseline because it cannot increase calibration error on the
    validation set by construction.

    Args:
        probs_val: Predicted probabilities on the validation set ``(n_val,)``.
        y_val: Binary labels ``(n_val,)``.
        probs_eval: Optional evaluation probabilities to calibrate.  If
            ``None``, the validation probabilities are calibrated.
        n_bins: Number of equal-width bins.

    Returns:
        ``HistogramBinningResult`` with bin edges, per-bin calibrated
        probabilities, and ECE metrics.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_cal_probs = np.zeros(n_bins, dtype=np.float64)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (probs_val >= lo) & (probs_val < hi)
        else:
            mask = (probs_val >= lo) & (probs_val <= hi)

        if np.any(mask):
            bin_cal_probs[i] = float(np.mean(y_val[mask]))
        else:
            bin_cal_probs[i] = (lo + hi) / 2.0

    target = probs_eval if probs_eval is not None else probs_val
    y_report = y_val

    calibrated = np.empty_like(target)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (target >= lo) & (target < hi)
        else:
            mask = (target >= lo) & (target <= hi)
        calibrated[mask] = bin_cal_probs[i]

    ece_before = ece(probs_val, y_val, n_bins=n_bins)
    if probs_eval is None:
        ece_after = ece(calibrated, y_report, n_bins=n_bins)
    else:
        ece_after = float("nan")

    return HistogramBinningResult(
        bin_edges=bin_edges,
        bin_calibrated_probs=bin_cal_probs,
        probs_calibrated=calibrated,
        ece_before=ece_before,
        ece_after=ece_after,
    )
