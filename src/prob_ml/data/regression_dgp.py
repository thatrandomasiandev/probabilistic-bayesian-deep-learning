"""Synthetic heteroscedastic regression DGP with optional covariate shift."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from prob_ml.data.base import RegressionDataset
from prob_ml.utils.seed import set_seed


@dataclass
class RegressionDGPConfig:
    n_samples: int = 5000
    n_features: int = 10
    noise_base: float = 0.3
    heteroscedastic_strength: float = 1.0
    ood_fraction: float = 0.0
    seed: int = 42


def _f(x: np.ndarray) -> np.ndarray:
    """Nonlinear mean function with known structure."""
    return (
        0.8 * x[:, 0]
        + 0.5 * np.sin(2 * np.pi * x[:, 1])
        + 0.3 * x[:, 2] * x[:, 3]
    )


def _sigma(x: np.ndarray, base: float, strength: float) -> np.ndarray:
    """Heteroscedastic aleatoric noise: higher variance in feature tails."""
    return base + strength * np.abs(x[:, 0]) + 0.1 * strength * np.abs(x[:, 1])


def generate_regression_data(config: RegressionDGPConfig | None = None) -> RegressionDataset:
    """
    Generate regression data with known mean and aleatoric variance.

    DGP:
      X ~ N(0, I)
      mu(x) = f(x)
      sigma(x) = noise_base + heteroscedastic_strength * |x0| + ...
      y = mu(x) + sigma(x) * eps,  eps ~ N(0, 1)

    Optional OOD: last `ood_fraction` of samples drawn from N(2, I).
    """
    cfg = config or RegressionDGPConfig()
    rng = set_seed(cfg.seed)

    n, d = cfg.n_samples, cfg.n_features
    X = rng.standard_normal((n, d))

    n_ood = int(n * cfg.ood_fraction)
    if n_ood > 0:
        X[-n_ood:] = rng.normal(2.0, 1.0, size=(n_ood, d))

    mu = _f(X)
    sigma = _sigma(X, cfg.noise_base, cfg.heteroscedastic_strength)
    y = mu + sigma * rng.standard_normal(n)

    in_domain = np.ones(n, dtype=bool)
    if n_ood > 0:
        in_domain[-n_ood:] = False

    return RegressionDataset(
        X=X,
        y=y,
        metadata={
            "dgp": "heteroscedastic_regression",
            "n_samples": n,
            "n_features": d,
            "noise_base": cfg.noise_base,
            "heteroscedastic_strength": cfg.heteroscedastic_strength,
            "ood_fraction": cfg.ood_fraction,
            "seed": cfg.seed,
        },
        ground_truth={
            "mu": mu,
            "sigma": sigma,
            "in_domain": in_domain,
        },
    )
