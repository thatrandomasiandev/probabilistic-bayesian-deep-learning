"""Tests for Bayesian MLP components and deep-ensemble uncertainty."""

from __future__ import annotations

import numpy as np
import torch

from prob_ml.models.mlp import BayesianLinear, BayesianMLP, MCDropoutMLP
from prob_ml.uncertainty.deep_ensembles import DeepEnsemble, DeepEnsembleConfig


# ------------------------------------------------------------------
# BayesianLinear: KL divergence is positive
# ------------------------------------------------------------------


def test_bayesian_linear_kl_positive():
    """KL(q || p) must be non-negative for any variational posterior.
    After initialisation with small sigma, KL should be strictly > 0
    because the posterior differs from the N(0, 1) prior.
    """
    layer = BayesianLinear(8, 4)
    kl = layer.kl_divergence()
    assert kl.item() > 0.0, f"Expected KL > 0, got {kl.item()}"


def test_bayesian_linear_forward_stochastic():
    """Two forward passes should produce different outputs because
    weights are sampled each time.
    """
    torch.manual_seed(0)
    layer = BayesianLinear(8, 4)
    x = torch.randn(16, 8)
    out1 = layer(x).detach()
    out2 = layer(x).detach()
    assert not torch.allclose(out1, out2, atol=1e-7), (
        "BayesianLinear should produce stochastic outputs"
    )


# ------------------------------------------------------------------
# BayesianMLP: ELBO loss and KL
# ------------------------------------------------------------------


def test_bayesian_mlp_elbo_finite():
    """ELBO loss should be a finite scalar."""
    torch.manual_seed(1)
    model = BayesianMLP(in_dim=5, out_dim=1, hidden_dim=16, n_hidden=2)
    x = torch.randn(32, 5)
    y = torch.randn(32)
    loss = model.elbo_loss(x, y, n_samples=2, beta=1e-3)
    assert torch.isfinite(loss), f"ELBO loss is not finite: {loss.item()}"


def test_bayesian_mlp_kl_positive():
    """Total network KL should be positive."""
    model = BayesianMLP(in_dim=5, out_dim=1, hidden_dim=16, n_hidden=2)
    kl = model.kl_divergence()
    assert kl.item() > 0.0


# ------------------------------------------------------------------
# MCDropoutMLP: uncertainty increases on OOD data
# ------------------------------------------------------------------


def test_mc_dropout_uncertainty_increases_on_ood():
    """MC-Dropout variance should be higher on out-of-distribution
    inputs (drawn from a shifted distribution) than on in-distribution
    inputs.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    in_dim = 5
    model = MCDropoutMLP(in_dim=in_dim, hidden_dim=32, n_hidden=2, dropout=0.3)

    x_train = torch.randn(200, in_dim)
    y_train = x_train[:, 0] + 0.5 * x_train[:, 1] + 0.1 * torch.randn(200)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(80):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x_train), y_train)
        loss.backward()
        optimizer.step()

    x_id = torch.randn(100, in_dim)
    x_ood = torch.randn(100, in_dim) * 5.0 + 10.0

    _, var_id = model.predict_with_uncertainty(x_id, n_samples=50)
    _, var_ood = model.predict_with_uncertainty(x_ood, n_samples=50)

    mean_var_id = var_id.mean().item()
    mean_var_ood = var_ood.mean().item()

    assert mean_var_ood > mean_var_id, (
        f"OOD variance ({mean_var_ood:.6f}) should exceed "
        f"in-distribution variance ({mean_var_id:.6f})"
    )


# ------------------------------------------------------------------
# DeepEnsemble: epistemic > aleatoric on low-data regime
# ------------------------------------------------------------------


def test_deep_ensemble_epistemic_exceeds_aleatoric_low_data():
    """With very few training samples and OOD test inputs, model
    uncertainty (epistemic) should dominate data noise (aleatoric)
    because ensemble members disagree substantially.
    """
    rng = np.random.default_rng(99)
    n_train = 10
    in_dim = 5
    X_train = rng.standard_normal((n_train, in_dim)).astype(np.float32)
    y_train = (
        np.sin(X_train[:, 0]) * X_train[:, 1]
        + 0.05 * rng.standard_normal(n_train)
    ).astype(np.float32)

    X_test = (rng.standard_normal((50, in_dim)) * 3.0 + 5.0).astype(np.float32)

    config = DeepEnsembleConfig(
        n_models=5,
        epochs=20,
        hidden_dim=32,
        n_hidden=2,
        lr=5e-3,
        seed=99,
    )
    ensemble = DeepEnsemble(config)
    ensemble.fit(X_train, y_train)
    pred = ensemble.predict(X_test)

    mean_epi = float(pred.epistemic_var.mean())
    mean_ale = float(pred.aleatoric_var.mean())

    assert mean_epi > mean_ale, (
        f"Epistemic variance ({mean_epi:.6f}) should exceed "
        f"aleatoric variance ({mean_ale:.6f}) in the low-data regime"
    )


# ------------------------------------------------------------------
# DeepEnsemble: predictions have correct shapes
# ------------------------------------------------------------------


def test_deep_ensemble_predict_shapes():
    """Verify output array shapes from DeepEnsemble.predict."""
    rng = np.random.default_rng(10)
    X = rng.standard_normal((60, 4)).astype(np.float32)
    y = rng.standard_normal(60).astype(np.float32)
    X_test = rng.standard_normal((20, 4)).astype(np.float32)

    ensemble = DeepEnsemble(DeepEnsembleConfig(n_models=3, epochs=5, seed=10))
    ensemble.fit(X, y)
    pred = ensemble.predict(X_test)

    assert pred.mean.shape == (20,)
    assert pred.epistemic_var.shape == (20,)
    assert pred.aleatoric_var.shape == (20,)
    assert pred.total_var.shape == (20,)
    assert np.allclose(
        pred.total_var,
        pred.epistemic_var + pred.aleatoric_var,
        atol=1e-6,
    )
