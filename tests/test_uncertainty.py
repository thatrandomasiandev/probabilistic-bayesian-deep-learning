"""Tests for uncertainty estimation."""

import numpy as np
from sklearn.model_selection import train_test_split

from prob_ml.data.regression_dgp import RegressionDGPConfig, generate_regression_data
from prob_ml.models.trainer import TrainConfig
from prob_ml.uncertainty.deep_ensemble import fit_deep_ensemble
from prob_ml.uncertainty.mc_dropout import mc_dropout_predict
from prob_ml.uncertainty.metrics import gaussian_interval, picp, rmse


def test_mc_dropout_runs():
    data = generate_regression_data(RegressionDGPConfig(n_samples=400, seed=10))
    X_train, X_test, y_train, y_test = train_test_split(
        data.X, data.y, test_size=0.3, random_state=10
    )
    cfg = TrainConfig(epochs=15, dropout=0.2, seed=10)
    result = mc_dropout_predict(
        X_train, y_train, X_test, n_mc_samples=10, config=cfg
    )
    assert result.mean.shape == (len(X_test),)
    assert result.std.shape == (len(X_test),)
    assert rmse(result.mean, y_test) < 2.0


def test_deep_ensemble_runs():
    data = generate_regression_data(RegressionDGPConfig(n_samples=400, seed=11))
    X_train, X_test, y_train, y_test = train_test_split(
        data.X, data.y, test_size=0.3, random_state=11
    )
    cfg = TrainConfig(epochs=15, seed=11)
    result = fit_deep_ensemble(
        X_train, y_train, X_test, n_members=3, config=cfg, seed=11
    )
    assert result.member_predictions.shape == (3, len(X_test))
    lower, upper = gaussian_interval(result.mean, result.std, alpha=0.1)
    assert 0.0 <= picp(y_test, lower, upper) <= 1.0
