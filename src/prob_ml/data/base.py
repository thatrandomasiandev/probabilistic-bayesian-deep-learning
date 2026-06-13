"""Dataset containers with ground-truth accessors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RegressionDataset:
    """Regression data with optional heteroscedastic noise ground truth."""

    X: np.ndarray
    y: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    ground_truth: dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1] if self.X.ndim > 1 else 1


@dataclass
class ClassificationDataset:
    """Binary classification data with logits and calibration metadata."""

    X: np.ndarray
    y: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    ground_truth: dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1] if self.X.ndim > 1 else 1
