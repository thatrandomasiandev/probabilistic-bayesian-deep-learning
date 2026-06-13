"""Deep ensemble with heteroscedastic variance heads.

Implements the approach of Lakshminarayanan et al. (2017): each ensemble
member predicts both a mean and a log-variance, trained with the Gaussian
negative log-likelihood.  The ensemble's predictive uncertainty naturally
decomposes into epistemic (inter-model disagreement) and aleatoric
(mean predicted variance) components.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from prob_ml.utils.seed import set_torch_seed

logger = logging.getLogger(__name__)


class VarianceHead(nn.Module):
    """MLP head that predicts both mean and log-variance.

    The network outputs two values per sample: the predictive mean mu
    and log(sigma^2).  Training uses the heteroscedastic Gaussian NLL:

        L = 0.5 * [ log_var + (y - mu)^2 / exp(log_var) ]

    Predicting log-variance (instead of variance directly) avoids
    constrained optimisation and is numerically more stable.

    Args:
        in_dim: Input feature dimensionality.
        hidden_dim: Width of hidden layers.
        n_hidden: Number of hidden layers.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        n_hidden: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(n_hidden):
            layers.extend([
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            ])
            dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(dim, 1)
        self.logvar_head = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and log-variance.

        Args:
            x: Input tensor ``(batch, in_dim)``.

        Returns:
            Tuple ``(mean, log_var)`` each of shape ``(batch,)``.
        """
        h = self.backbone(x)
        mu = self.mean_head(h).squeeze(-1)
        log_var = self.logvar_head(h).squeeze(-1)
        return mu, log_var


def gaussian_nll_loss(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Heteroscedastic Gaussian negative log-likelihood.

    L = 0.5 * mean[ log_var + (y - mu)^2 / exp(log_var) ]

    A small floor is applied to exp(log_var) for numerical stability.

    Args:
        mu: Predicted means ``(batch,)``.
        log_var: Predicted log-variances ``(batch,)``.
        y: True targets ``(batch,)``.

    Returns:
        Scalar loss.
    """
    var = torch.exp(log_var).clamp(min=1e-6)
    return 0.5 * torch.mean(log_var + (y - mu) ** 2 / var)


# ---------------------------------------------------------------------------
# DeepEnsemble class
# ---------------------------------------------------------------------------


@dataclass
class DeepEnsembleConfig:
    """Configuration for ``DeepEnsemble`` training.

    Attributes:
        n_models: Number of ensemble members.
        epochs: Training epochs per member.
        batch_size: Mini-batch size.
        lr: Learning rate.
        weight_decay: L2 regularisation.
        hidden_dim: Hidden layer width.
        n_hidden: Number of hidden layers.
        dropout: Dropout probability.
        seed: Base random seed (member *i* uses ``seed + i``).
    """

    n_models: int = 5
    epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 64
    n_hidden: int = 2
    dropout: float = 0.1
    seed: int = 42


@dataclass
class DeepEnsemblePrediction:
    """Prediction container for ``DeepEnsemble``.

    Attributes:
        mean: Ensemble predictive mean ``(n,)``.
        epistemic_var: Inter-model variance (epistemic) ``(n,)``.
        aleatoric_var: Mean predicted variance (aleatoric) ``(n,)``.
        total_var: Sum of epistemic and aleatoric variance ``(n,)``.
    """

    mean: np.ndarray
    epistemic_var: np.ndarray
    aleatoric_var: np.ndarray
    total_var: np.ndarray


class DeepEnsemble:
    """Heteroscedastic deep ensemble (Lakshminarayanan et al., 2017).

    Each member is a ``VarianceHead`` network trained with Gaussian NLL.
    Uncertainty decomposes as:

        epistemic  = Var_m[ mu_m(x) ]          (model disagreement)
        aleatoric  = (1/M) sum_m sigma^2_m(x)  (average predicted noise)
        total      = epistemic + aleatoric

    Args:
        config: Ensemble training configuration.
    """

    def __init__(self, config: DeepEnsembleConfig | None = None) -> None:
        self.config = config or DeepEnsembleConfig()
        self.models: list[VarianceHead] = []

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        """Train all ensemble members from scratch.

        Each member is initialised with a different random seed to promote
        diversity.

        Args:
            X_train: Training features ``(n_train, d)``.
            y_train: Training targets ``(n_train,)``.
        """
        cfg = self.config
        self.models = []

        x_t = torch.as_tensor(X_train, dtype=torch.float32)
        y_t = torch.as_tensor(y_train, dtype=torch.float32)
        dataset = TensorDataset(x_t, y_t)

        for m in range(cfg.n_models):
            set_torch_seed(cfg.seed + m)
            model = VarianceHead(
                in_dim=X_train.shape[1],
                hidden_dim=cfg.hidden_dim,
                n_hidden=cfg.n_hidden,
                dropout=cfg.dropout,
            )
            optimizer = torch.optim.Adam(
                model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
            )
            loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

            model.train()
            for epoch in range(cfg.epochs):
                for xb, yb in loader:
                    optimizer.zero_grad()
                    mu, log_var = model(xb)
                    loss = gaussian_nll_loss(mu, log_var, yb)
                    loss.backward()
                    optimizer.step()

            model.eval()
            self.models.append(model)
            logger.debug("Trained ensemble member %d/%d", m + 1, cfg.n_models)

    @torch.no_grad()
    def predict(
        self,
        X: np.ndarray,
    ) -> DeepEnsemblePrediction:
        """Generate predictions with epistemic / aleatoric decomposition.

        Args:
            X: Input features ``(n, d)``.

        Returns:
            ``DeepEnsemblePrediction`` with mean and variance arrays.

        Raises:
            RuntimeError: If ``fit`` has not been called.
        """
        if not self.models:
            raise RuntimeError("Call fit() before predict()")

        x_t = torch.as_tensor(X, dtype=torch.float32)
        means: list[np.ndarray] = []
        variances: list[np.ndarray] = []

        for model in self.models:
            model.eval()
            mu, log_var = model(x_t)
            means.append(mu.cpu().numpy())
            variances.append(torch.exp(log_var).cpu().numpy())

        means_arr = np.stack(means, axis=0)
        vars_arr = np.stack(variances, axis=0)

        ensemble_mean = means_arr.mean(axis=0)
        epistemic_var = means_arr.var(axis=0)
        aleatoric_var = vars_arr.mean(axis=0)
        total_var = epistemic_var + aleatoric_var

        return DeepEnsemblePrediction(
            mean=ensemble_mean,
            epistemic_var=epistemic_var,
            aleatoric_var=aleatoric_var,
            total_var=total_var,
        )
