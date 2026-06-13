"""Tests for conformal prediction."""

import numpy as np
from sklearn.model_selection import train_test_split

from prob_ml.conformal.metrics import coverage
from prob_ml.conformal.split import split_conformal_regression
from prob_ml.data.regression_dgp import RegressionDGPConfig, generate_regression_data
from prob_ml.models.trainer import TrainConfig


def test_split_conformal_coverage():
    data = generate_regression_data(RegressionDGPConfig(n_samples=600, seed=30))
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        data.X, data.y, test_size=0.4, random_state=30
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=30
    )

    result = split_conformal_regression(
        X_train,
        y_train,
        X_cal,
        y_cal,
        X_test,
        alpha=0.1,
        config=TrainConfig(epochs=20, seed=30),
    )
    cov = coverage(y_test, result.lower, result.upper)
    assert 0.7 <= cov <= 1.0
    assert np.all(result.lower <= result.upper)
