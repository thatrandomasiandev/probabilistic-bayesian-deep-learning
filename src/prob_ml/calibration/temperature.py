"""Post-hoc temperature scaling for neural classifiers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from prob_ml.calibration.metrics import brier_score, ece, nll_binary


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


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
    """
    Fit a scalar temperature T on validation logits to minimize NLL.

    Calibrated probability: sigmoid(logit / T).
    Metrics are computed on `logits_eval`/`y_eval` when provided, else validation.
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
