"""Split conformal prediction for regression and classification.

Provides distribution-free prediction intervals (regression) and prediction
sets (classification) with finite-sample coverage guarantees under the
exchangeability assumption.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from prob_ml.models.trainer import TrainConfig, train_regressor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Legacy functional interface (regression)
# ---------------------------------------------------------------------------


@dataclass
class SplitConformalResult:
    """Result from the legacy ``split_conformal_regression`` function.

    Attributes:
        y_hat: Point predictions on the test set.
        lower: Lower interval bounds.
        upper: Upper interval bounds.
        quantile: Calibrated nonconformity quantile.
        alpha: Miscoverage level.
    """

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
    """Split conformal prediction with absolute residual scores.

    Uses a held-out calibration set to compute the ``ceil((n+1)(1-alpha))/n``
    quantile of nonconformity scores ``|y - y_hat|``, then forms symmetric
    prediction intervals.

    Args:
        X_train: Training features ``(n_train, d)``.
        y_train: Training targets ``(n_train,)``.
        X_cal: Calibration features ``(n_cal, d)``.
        y_cal: Calibration targets ``(n_cal,)``.
        X_test: Test features ``(n_test, d)``.
        alpha: Desired miscoverage rate in ``(0, 1)``.
        config: Optional training configuration.

    Returns:
        ``SplitConformalResult`` with prediction intervals.
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


# ---------------------------------------------------------------------------
# OOP: SplitConformalRegressor
# ---------------------------------------------------------------------------


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Compute the conformal quantile with finite-sample correction.

    q = quantile(scores, ceil((n+1)(1-alpha)) / n)

    Args:
        scores: Nonconformity scores ``(n,)``.
        alpha: Desired miscoverage rate.

    Returns:
        The calibrated quantile threshold.
    """
    n = len(scores)
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, q_level, method="higher"))


@dataclass
class SplitConformalRegressor:
    """Split conformal regressor with calibrate / predict_interval API.

    After calling ``calibrate`` on a held-out calibration set, the
    ``predict_interval`` method produces prediction intervals with
    marginal coverage >= 1 - alpha under exchangeability.

    Args:
        alpha: Desired miscoverage level (default 0.1 for 90 % coverage).
    """

    alpha: float = 0.1
    quantile_: float = field(init=False, default=float("nan"))
    _calibrated: bool = field(init=False, default=False, repr=False)

    def calibrate(
        self,
        y_cal: np.ndarray,
        y_hat_cal: np.ndarray,
    ) -> None:
        """Calibrate using absolute-residual nonconformity scores.

        Args:
            y_cal: True calibration targets ``(n_cal,)``.
            y_hat_cal: Model predictions on the calibration set ``(n_cal,)``.
        """
        scores = np.abs(y_cal - y_hat_cal)
        self.quantile_ = _conformal_quantile(scores, self.alpha)
        self._calibrated = True
        logger.debug("Calibrated quantile=%.4f (alpha=%.2f)", self.quantile_, self.alpha)

    def predict_interval(
        self,
        y_hat_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Produce prediction intervals centred on point predictions.

        Args:
            y_hat_test: Point predictions ``(n_test,)``.

        Returns:
            Tuple ``(lower, upper)`` each of shape ``(n_test,)``.

        Raises:
            RuntimeError: If ``calibrate`` has not been called.
        """
        if not self._calibrated:
            raise RuntimeError("Call calibrate() before predict_interval()")
        lower = y_hat_test - self.quantile_
        upper = y_hat_test + self.quantile_
        return lower, upper


# ---------------------------------------------------------------------------
# OOP: SplitConformalClassifier
# ---------------------------------------------------------------------------


@dataclass
class SplitConformalClassifier:
    """Split conformal classifier producing prediction sets.

    Uses softmax-based nonconformity scores:  s(x, y) = 1 - p_hat(y | x).

    After calibration, ``predict_set`` returns the smallest set of labels
    whose cumulative score is below the calibrated threshold.

    Args:
        alpha: Desired miscoverage level.
    """

    alpha: float = 0.1
    quantile_: float = field(init=False, default=float("nan"))
    _calibrated: bool = field(init=False, default=False, repr=False)

    def calibrate(
        self,
        probs_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> None:
        """Calibrate on held-out softmax probabilities and true labels.

        Args:
            probs_cal: Softmax probabilities ``(n_cal, n_classes)``.
            y_cal: Integer class labels ``(n_cal,)``.
        """
        y_int = y_cal.astype(int)
        scores = 1.0 - probs_cal[np.arange(len(y_cal)), y_int]
        self.quantile_ = _conformal_quantile(scores, self.alpha)
        self._calibrated = True
        logger.debug("Calibrated quantile=%.4f (alpha=%.2f)", self.quantile_, self.alpha)

    def predict_set(
        self,
        probs_test: np.ndarray,
    ) -> list[np.ndarray]:
        """Return prediction sets for each test sample.

        A class ``c`` is included in the prediction set if
        ``1 - p_hat(c | x) <= quantile_``, equivalently
        ``p_hat(c | x) >= 1 - quantile_``.

        Args:
            probs_test: Softmax probabilities ``(n_test, n_classes)``.

        Returns:
            List of integer arrays, one per test sample, containing the
            labels in the prediction set.

        Raises:
            RuntimeError: If ``calibrate`` has not been called.
        """
        if not self._calibrated:
            raise RuntimeError("Call calibrate() before predict_set()")
        threshold = 1.0 - self.quantile_
        sets: list[np.ndarray] = []
        for row in probs_test:
            included = np.where(row >= threshold)[0]
            if len(included) == 0:
                included = np.array([int(np.argmax(row))])
            sets.append(included)
        return sets


# ---------------------------------------------------------------------------
# Adaptive Conformal (RAPS)
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveConformal:
    """Regularised Adaptive Prediction Sets (RAPS).

    RAPS (Angelopoulos et al., 2021) uses a modified nonconformity score
    that penalises large prediction sets via a regularisation term:

        s(x, y) = sum_{j : pi(j) <= pi(y)} p_hat(pi(j) | x)
                  + lambda * max(0, |{j : pi(j) <= pi(y)}| - k_reg)

    where pi is the descending-probability ordering, lambda controls the
    penalty strength, and k_reg is a size threshold.

    Args:
        alpha: Desired miscoverage level.
        lam: Regularisation strength penalising set sizes beyond ``k_reg``.
        k_reg: Size threshold; the penalty activates for sets larger than this.
    """

    alpha: float = 0.1
    lam: float = 0.01
    k_reg: int = 1
    quantile_: float = field(init=False, default=float("nan"))
    _calibrated: bool = field(init=False, default=False, repr=False)

    def _score(self, probs: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute RAPS nonconformity scores.

        Args:
            probs: Softmax probabilities ``(n, C)``.
            y: Integer class labels ``(n,)``.

        Returns:
            Scores ``(n,)``.
        """
        n = len(y)
        y_int = y.astype(int)
        sorted_idx = np.argsort(-probs, axis=1)
        sorted_probs = np.take_along_axis(probs, sorted_idx, axis=1)
        cumsum = np.cumsum(sorted_probs, axis=1)

        ranks = np.zeros(n, dtype=int)
        for i in range(n):
            ranks[i] = int(np.where(sorted_idx[i] == y_int[i])[0][0])

        scores = np.empty(n, dtype=np.float64)
        for i in range(n):
            r = ranks[i]
            scores[i] = cumsum[i, r] + self.lam * max(0, r + 1 - self.k_reg)
        return scores

    def calibrate(
        self,
        probs_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> None:
        """Calibrate on held-out softmax probabilities and true labels.

        Args:
            probs_cal: Softmax probabilities ``(n_cal, C)``.
            y_cal: Integer class labels ``(n_cal,)``.
        """
        scores = self._score(probs_cal, y_cal)
        self.quantile_ = _conformal_quantile(scores, self.alpha)
        self._calibrated = True

    def predict_set(
        self,
        probs_test: np.ndarray,
    ) -> list[np.ndarray]:
        """Return RAPS prediction sets for each test sample.

        Classes are included in descending probability order until the
        cumulative score exceeds the calibrated quantile.

        Args:
            probs_test: Softmax probabilities ``(n_test, C)``.

        Returns:
            List of integer arrays containing prediction-set labels.

        Raises:
            RuntimeError: If ``calibrate`` has not been called.
        """
        if not self._calibrated:
            raise RuntimeError("Call calibrate() before predict_set()")

        sets: list[np.ndarray] = []
        for row in probs_test:
            order = np.argsort(-row)
            cum = 0.0
            included: list[int] = []
            for rank, idx in enumerate(order):
                cum += row[idx]
                penalty = self.lam * max(0, rank + 1 - self.k_reg)
                included.append(int(idx))
                if cum + penalty >= self.quantile_:
                    break
            sets.append(np.array(included))
        return sets
