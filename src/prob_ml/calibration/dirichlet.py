"""Dirichlet calibration for multiclass probability estimates.

Implements the approach of Kull et al. (2019): learn a linear map in
log-probability space followed by a softmax, which is equivalent to
fitting a Dirichlet distribution to the model's predicted probabilities.
The learned map W, b parameterises:

    z_cal = W @ log(p + eps) + b
    p_cal = softmax(z_cal)

When W is constrained to be diagonal this reduces to *Beta calibration*
for each class independently.  The unconstrained version can model
inter-class dependencies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _to_log_probs(probs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Clip probabilities and take element-wise log."""
    return np.log(np.clip(probs, eps, 1.0))


def _softmax_np(z: np.ndarray) -> np.ndarray:
    shifted = z - z.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class DirichletCalibrationResult:
    """Result container for Dirichlet calibration.

    Attributes:
        W: Learned weight matrix ``(C, C)`` or diagonal ``(C,)``
            when ``diagonal=True``.
        b: Learned bias vector ``(C,)``.
        probs_calibrated: Calibrated probabilities ``(n, C)``.
        nll_before: Negative log-likelihood before calibration.
        nll_after: Negative log-likelihood after calibration.
    """

    W: np.ndarray
    b: np.ndarray
    probs_calibrated: np.ndarray
    nll_before: float
    nll_after: float


# ---------------------------------------------------------------------------
# Torch parameterisation
# ---------------------------------------------------------------------------


class _DirichletMap(nn.Module):
    """Learnable linear map in log-probability space."""

    def __init__(self, n_classes: int, *, diagonal: bool = False) -> None:
        super().__init__()
        self.diagonal = diagonal
        if diagonal:
            self.W_diag = nn.Parameter(torch.ones(n_classes))
        else:
            self.W = nn.Parameter(torch.eye(n_classes))
        self.b = nn.Parameter(torch.zeros(n_classes))

    def forward(self, log_probs: torch.Tensor) -> torch.Tensor:
        if self.diagonal:
            return log_probs * self.W_diag + self.b
        return log_probs @ self.W.T + self.b


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class DirichletCalibration:
    """Dirichlet calibration for multiclass probability estimates.

    Learns an affine map in log-probability space trained to minimise
    cross-entropy on a held-out validation set.

    Args:
        diagonal: If ``True``, constrain W to be diagonal (Beta calibration).
        lr: Optimiser learning rate.
        max_iter: Maximum L-BFGS iterations.
        l2_reg: L2 regularisation on W (prevents overfitting on small
            calibration sets).
    """

    diagonal: bool = False
    lr: float = 0.01
    max_iter: int = 300
    l2_reg: float = 1e-4

    _module: _DirichletMap | None = field(init=False, default=None, repr=False)
    _n_classes: int = field(init=False, default=0, repr=False)

    def fit(
        self,
        probs_val: np.ndarray,
        y_val: np.ndarray,
    ) -> DirichletCalibrationResult:
        """Fit the Dirichlet calibration map on validation data.

        Args:
            probs_val: Predicted probabilities ``(n_val, C)``.
            y_val: Integer class labels ``(n_val,)``.

        Returns:
            ``DirichletCalibrationResult`` with learned parameters and
            calibrated probabilities on the validation set.
        """
        if probs_val.ndim != 2:
            raise ValueError("probs_val must be 2-D (n_val, n_classes)")

        n_classes = probs_val.shape[1]
        self._n_classes = n_classes

        log_p = _to_log_probs(probs_val)
        log_p_t = torch.as_tensor(log_p, dtype=torch.float32)
        y_t = torch.as_tensor(y_val, dtype=torch.long)

        module = _DirichletMap(n_classes, diagonal=self.diagonal)
        self._module = module

        optimizer = torch.optim.LBFGS(
            module.parameters(), lr=self.lr, max_iter=self.max_iter,
        )
        criterion = nn.CrossEntropyLoss()
        l2_reg = self.l2_reg

        nll_before = self._nll(probs_val, y_val)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            logits = module(log_p_t)
            loss = criterion(logits, y_t)
            if l2_reg > 0:
                if module.diagonal:
                    loss = loss + l2_reg * module.W_diag.pow(2).sum()
                else:
                    loss = loss + l2_reg * module.W.pow(2).sum()
            loss.backward()
            return loss

        optimizer.step(closure)

        probs_cal = self.transform(probs_val)
        nll_after = self._nll(probs_cal, y_val)

        W_np = (
            module.W_diag.detach().cpu().numpy()
            if self.diagonal
            else module.W.detach().cpu().numpy()
        )
        b_np = module.b.detach().cpu().numpy()

        return DirichletCalibrationResult(
            W=W_np,
            b=b_np,
            probs_calibrated=probs_cal,
            nll_before=nll_before,
            nll_after=nll_after,
        )

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Apply the learned calibration map to new probabilities.

        Args:
            probs: Predicted probabilities ``(n, C)``.

        Returns:
            Calibrated probabilities ``(n, C)``.

        Raises:
            RuntimeError: If ``fit`` has not been called.
        """
        if self._module is None:
            raise RuntimeError("Call fit() before transform()")

        log_p = _to_log_probs(probs)
        log_p_t = torch.as_tensor(log_p, dtype=torch.float32)

        with torch.no_grad():
            logits = self._module(log_p_t).cpu().numpy()

        return _softmax_np(logits)

    @staticmethod
    def _nll(probs: np.ndarray, y: np.ndarray) -> float:
        """Compute negative log-likelihood."""
        y_int = y.astype(int)
        p_true = probs[np.arange(len(y)), y_int]
        return float(-np.mean(np.log(np.clip(p_true, 1e-8, 1.0))))
