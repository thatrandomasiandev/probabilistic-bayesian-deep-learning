"""Marginal-coverage conformal bands for regression.

Wraps point predictions with distribution-free prediction bands whose
width adapts to a local uncertainty estimate (e.g. standard deviation
from MC-Dropout or a deep ensemble).  Uses the *normalised residual*
score function so that bands are narrower in low-uncertainty regions
and wider in high-uncertainty regions while maintaining the desired
marginal coverage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConformalBandResult:
    """Container for conformal band predictions.

    Attributes:
        lower: Lower band boundary ``(n,)``.
        upper: Upper band boundary ``(n,)``.
        quantile: Calibrated quantile of normalised residuals.
        alpha: Desired miscoverage level.
    """

    lower: np.ndarray
    upper: np.ndarray
    quantile: float
    alpha: float


@dataclass
class ConformalBand:
    """Marginal-coverage conformal bands for regression.

    Unlike standard split-conformal regression which produces constant-width
    intervals, ``ConformalBand`` uses a per-sample uncertainty estimate
    sigma(x) to produce locally adaptive bands:

        C(x) = [ mu(x) - q * sigma(x),  mu(x) + q * sigma(x) ]

    where q is calibrated so that the *normalised residual* score
    ``|y - mu(x)| / sigma(x)`` satisfies the conformal coverage guarantee.

    This is particularly useful when combined with MC-Dropout or deep-ensemble
    uncertainty estimates that provide meaningful relative ordering of
    prediction difficulty.

    Args:
        alpha: Desired miscoverage level (default 0.1 for 90 % coverage).
        min_sigma: Floor on sigma values to avoid division by zero.
    """

    alpha: float = 0.1
    min_sigma: float = 1e-6
    quantile_: float = field(init=False, default=float("nan"))
    _calibrated: bool = field(init=False, default=False, repr=False)

    def calibrate(
        self,
        y_cal: np.ndarray,
        y_hat_cal: np.ndarray,
        sigma_cal: np.ndarray,
    ) -> None:
        """Calibrate the band width on a held-out calibration set.

        Computes the normalised residuals ``|y - y_hat| / sigma`` and selects
        the conformal quantile with the finite-sample correction
        ``ceil((n+1)(1-alpha)) / n``.

        Args:
            y_cal: True calibration targets ``(n_cal,)``.
            y_hat_cal: Model point predictions on calibration set ``(n_cal,)``.
            sigma_cal: Uncertainty estimates on calibration set ``(n_cal,)``.
        """
        sigma_safe = np.maximum(np.asarray(sigma_cal, dtype=np.float64), self.min_sigma)
        scores = np.abs(y_cal - y_hat_cal) / sigma_safe

        n = len(scores)
        q_level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.quantile_ = float(np.quantile(scores, q_level, method="higher"))
        self._calibrated = True

        logger.debug(
            "ConformalBand calibrated: quantile=%.4f, alpha=%.2f, n_cal=%d",
            self.quantile_,
            self.alpha,
            n,
        )

    def predict(
        self,
        y_hat: np.ndarray,
        sigma: np.ndarray,
    ) -> ConformalBandResult:
        """Produce adaptive prediction bands.

        Args:
            y_hat: Point predictions ``(n,)``.
            sigma: Uncertainty estimates ``(n,)``.

        Returns:
            ``ConformalBandResult`` with lower/upper bands.

        Raises:
            RuntimeError: If ``calibrate`` has not been called.
        """
        if not self._calibrated:
            raise RuntimeError("Call calibrate() before predict()")

        sigma_safe = np.maximum(np.asarray(sigma, dtype=np.float64), self.min_sigma)
        half_width = self.quantile_ * sigma_safe

        return ConformalBandResult(
            lower=y_hat - half_width,
            upper=y_hat + half_width,
            quantile=self.quantile_,
            alpha=self.alpha,
        )
