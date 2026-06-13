"""Tests for synthetic DGPs."""

import numpy as np

from prob_ml.data.classification_dgp import ClassificationDGPConfig, generate_classification_data
from prob_ml.data.regression_dgp import RegressionDGPConfig, generate_regression_data


def test_regression_dgp_shapes():
    data = generate_regression_data(RegressionDGPConfig(n_samples=500, n_features=5, seed=1))
    assert data.X.shape == (500, 5)
    assert data.y.shape == (500,)
    assert data.ground_truth["mu"].shape == (500,)
    assert data.ground_truth["sigma"].shape == (500,)


def test_regression_heteroscedasticity():
    low = generate_regression_data(
        RegressionDGPConfig(n_samples=1000, heteroscedastic_strength=0.1, seed=2)
    )
    high = generate_regression_data(
        RegressionDGPConfig(n_samples=1000, heteroscedastic_strength=2.0, seed=2)
    )
    assert np.mean(high.ground_truth["sigma"]) > np.mean(low.ground_truth["sigma"])


def test_classification_dgp_shapes():
    data = generate_classification_data(ClassificationDGPConfig(n_samples=400, seed=3))
    assert data.X.shape[0] == 400
    assert set(np.unique(data.y)).issubset({0, 1})
    assert data.ground_truth["p_star"].shape == (400,)
