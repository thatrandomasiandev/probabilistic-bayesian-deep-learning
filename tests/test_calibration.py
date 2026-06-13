"""Tests for calibration and conformal coverage."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from prob_ml.calibration.metrics import ece
from prob_ml.calibration.temperature import (
    fit_temperature_scaling,
    histogram_binning,
)
from prob_ml.conformal.metrics import coverage
from prob_ml.conformal.split import split_conformal_regression
from prob_ml.data.classification_dgp import (
    ClassificationDGPConfig,
    generate_classification_data,
)
from prob_ml.data.regression_dgp import (
    RegressionDGPConfig,
    generate_regression_data,
)
from prob_ml.models.trainer import TrainConfig, train_classifier


def test_ece_low_for_simulated_perfect_calibration():
    rng = np.random.default_rng(0)
    probs = rng.uniform(0.05, 0.95, size=2000)
    y = rng.binomial(1, probs).astype(float)
    assert ece(probs, y, n_bins=10) < 0.08


def test_temperature_scaling_improves_or_maintains_nll():
    data = generate_classification_data(
        ClassificationDGPConfig(n_samples=800, label_noise=0.15, seed=20)
    )
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        data.X, data.y, test_size=0.4, random_state=20
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=20
    )

    model = train_classifier(X_train, y_train, TrainConfig(epochs=20, seed=20))
    model.eval()
    with torch.no_grad():
        logits_val = model(torch.as_tensor(X_val, dtype=torch.float32)).cpu().numpy()
        logits_test = model(torch.as_tensor(X_test, dtype=torch.float32)).cpu().numpy()

    ts = fit_temperature_scaling(
        logits_val, y_val, logits_eval=logits_test, y_eval=y_test
    )
    assert ts.temperature > 0
    assert ts.ece_after <= ts.ece_before + 0.05


# ------------------------------------------------------------------
# NEW: TemperatureScaler reduces ECE
# ------------------------------------------------------------------


def test_temperature_scaling_reduces_ece():
    """Temperature scaling should reduce (or at least not substantially
    increase) ECE on a validation set drawn from the same distribution.
    """
    data = generate_classification_data(
        ClassificationDGPConfig(n_samples=1200, label_noise=0.1, seed=55)
    )
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        data.X, data.y, test_size=0.4, random_state=55
    )
    X_val, _, y_val, _ = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=55
    )

    model = train_classifier(
        X_train, y_train, TrainConfig(epochs=30, seed=55)
    )
    model.eval()
    with torch.no_grad():
        logits_val = model(torch.as_tensor(X_val, dtype=torch.float32)).cpu().numpy()

    ts = fit_temperature_scaling(logits_val, y_val)
    assert ts.ece_after <= ts.ece_before + 0.01


# ------------------------------------------------------------------
# NEW: Temperature > 1 when model is overconfident
# ------------------------------------------------------------------


def test_temperature_greater_than_one_when_overconfident():
    """A well-trained model with low label noise tends to be over-confident;
    temperature scaling should learn T > 1 to soften the outputs.
    """
    data = generate_classification_data(
        ClassificationDGPConfig(
            n_samples=1000, label_noise=0.0, class_separation=2.0, seed=77
        )
    )
    X_train, X_val, y_train, y_val = train_test_split(
        data.X, data.y, test_size=0.3, random_state=77
    )

    model = train_classifier(
        X_train, y_train, TrainConfig(epochs=60, seed=77)
    )
    model.eval()
    with torch.no_grad():
        logits_val = model(torch.as_tensor(X_val, dtype=torch.float32)).cpu().numpy()

    ts = fit_temperature_scaling(logits_val, y_val)
    assert ts.temperature > 1.0, (
        f"Expected temperature > 1 for overconfident model, got {ts.temperature:.4f}"
    )


# ------------------------------------------------------------------
# NEW: Conformal intervals achieve >= (1 - alpha) coverage
# ------------------------------------------------------------------


def test_conformal_intervals_achieve_coverage():
    """Split conformal regression intervals should achieve at least
    (1 - alpha) marginal coverage on the test set (with a small
    tolerance for finite-sample variance).
    """
    alpha = 0.1
    data = generate_regression_data(
        RegressionDGPConfig(n_samples=800, seed=88)
    )
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        data.X, data.y, test_size=0.4, random_state=88
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=88
    )

    result = split_conformal_regression(
        X_train,
        y_train,
        X_cal,
        y_cal,
        X_test,
        alpha=alpha,
        config=TrainConfig(epochs=30, seed=88),
    )

    cov = coverage(y_test, result.lower, result.upper)
    assert cov >= (1 - alpha) - 0.05, (
        f"Coverage {cov:.3f} is below the target {1 - alpha:.2f}"
    )


# ------------------------------------------------------------------
# NEW: Histogram binning reduces ECE on validation set
# ------------------------------------------------------------------


def test_histogram_binning_reduces_ece():
    """Histogram binning should not increase ECE on its calibration set
    (by construction it assigns per-bin empirical accuracy).
    """
    rng = np.random.default_rng(42)
    probs = rng.uniform(0.0, 1.0, size=2000)
    probs = np.clip(probs ** 0.5, 0.01, 0.99)
    y = rng.binomial(1, 0.5 * np.ones_like(probs)).astype(float)

    result = histogram_binning(probs, y, n_bins=15)
    assert result.ece_after <= result.ece_before + 0.001
