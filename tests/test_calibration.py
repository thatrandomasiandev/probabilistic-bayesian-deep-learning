"""Tests for calibration."""

import numpy as np
from sklearn.model_selection import train_test_split

from prob_ml.calibration.metrics import ece
from prob_ml.calibration.temperature import fit_temperature_scaling
from prob_ml.data.classification_dgp import ClassificationDGPConfig, generate_classification_data
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

    import torch

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
