"""Synthetic binary classification DGP with controllable miscalibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from prob_ml.data.base import ClassificationDataset
from prob_ml.utils.seed import set_seed


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


@dataclass
class ClassificationDGPConfig:
    n_samples: int = 5000
    n_features: int = 10
    class_separation: float = 1.5
    label_noise: float = 0.0
    seed: int = 42


def generate_classification_data(
    config: ClassificationDGPConfig | None = None,
) -> ClassificationDataset:
    """
    Generate binary classification data with known Bayes-optimal probabilities.

    DGP:
      X ~ N(0, I)
      logits(x) = class_separation * (w·x)
      p*(x) = sigmoid(logits(x))
      y ~ Bernoulli(p*(x)) with optional label noise flip
    """
    cfg = config or ClassificationDGPConfig()
    rng = set_seed(cfg.seed)

    n, d = cfg.n_samples, cfg.n_features
    X = rng.standard_normal((n, d)).astype(np.float64)

    w = rng.standard_normal(d).astype(np.float64)
    w = w / (np.linalg.norm(w) + 1e-8)
    logits = cfg.class_separation * np.dot(X, w)
    p_star = _sigmoid(logits)
    y = rng.binomial(1, p_star).astype(np.int64)

    if cfg.label_noise > 0:
        flip = rng.random(n) < cfg.label_noise
        y = np.where(flip, 1 - y, y)

    return ClassificationDataset(
        X=X,
        y=y,
        metadata={
            "dgp": "binary_classification",
            "n_samples": n,
            "n_features": d,
            "class_separation": cfg.class_separation,
            "label_noise": cfg.label_noise,
            "seed": cfg.seed,
        },
        ground_truth={
            "logits": logits,
            "p_star": p_star,
            "weights": w,
        },
    )
