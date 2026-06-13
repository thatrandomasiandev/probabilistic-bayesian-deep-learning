"""PyTorch MLP backbones for regression, classification, and Bayesian inference."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class _MLPBackbone(nn.Module):
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
            layers.extend(
                [
                    nn.Linear(dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                ]
            )
            dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.out_dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class RegressionMLP(nn.Module):
    """Single-output MLP for regression."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        n_hidden: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = _MLPBackbone(in_dim, hidden_dim, n_hidden, dropout)
        self.head = nn.Linear(self.backbone.out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)).squeeze(-1)


class ClassificationMLP(nn.Module):
    """Binary classification MLP returning logits."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        n_hidden: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = _MLPBackbone(in_dim, hidden_dim, n_hidden, dropout)
        self.head = nn.Linear(self.backbone.out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)).squeeze(-1)


# ---------------------------------------------------------------------------
# Bayesian layers and models
# ---------------------------------------------------------------------------


class BayesianLinear(nn.Module):
    """Fully-connected layer with mean-field variational weight uncertainty.

    Each weight and bias is parameterised by a Gaussian q(w) = N(mu, sigma^2)
    where sigma = log(1 + exp(rho))  (softplus reparameterisation).

    During the forward pass a single Monte-Carlo sample is drawn via the
    local reparameterisation trick:

        w = mu + sigma * epsilon,   epsilon ~ N(0, 1)

    The KL divergence KL(q(w) || p(w)) is available via ``kl_divergence()``
    where the prior p(w) = N(0, 1).

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        prior_sigma: Standard deviation of the isotropic Gaussian prior.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_sigma: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_rho = nn.Parameter(torch.empty(out_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialise mu from Kaiming-uniform and rho so sigma starts small."""
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        nn.init.constant_(self.weight_rho, -5.0)
        fan_in = self.in_features
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.bias_rho, -5.0)

    @staticmethod
    def _softplus(x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sample weights from q and compute the linear transform.

        Args:
            x: Input tensor of shape ``(*, in_features)``.

        Returns:
            Output tensor of shape ``(*, out_features)``.
        """
        weight_sigma = self._softplus(self.weight_rho)
        bias_sigma = self._softplus(self.bias_rho)

        weight_eps = torch.randn_like(self.weight_mu)
        bias_eps = torch.randn_like(self.bias_mu)

        weight = self.weight_mu + weight_sigma * weight_eps
        bias = self.bias_mu + bias_sigma * bias_eps

        return F.linear(x, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        """Analytic KL(q(w) || p(w)) for Gaussian q and Gaussian prior.

        Uses the closed-form KL between two univariate Gaussians summed
        over all weight and bias parameters:

            KL = sum_i [ log(sigma_p / sigma_q)
                         + (sigma_q^2 + (mu_q - mu_p)^2) / (2 sigma_p^2)
                         - 0.5 ]

        Returns:
            Scalar KL divergence.
        """
        prior_var = self.prior_sigma ** 2
        log_prior = math.log(self.prior_sigma)

        def _kl_term(mu: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
            sigma = self._softplus(rho)
            return (
                log_prior
                - torch.log(sigma)
                + (sigma ** 2 + mu ** 2) / (2.0 * prior_var)
                - 0.5
            )

        kl_w = _kl_term(self.weight_mu, self.weight_rho).sum()
        kl_b = _kl_term(self.bias_mu, self.bias_rho).sum()
        return kl_w + kl_b


class BayesianMLP(nn.Module):
    """Multi-layer Bayesian neural network built from ``BayesianLinear`` layers.

    Trained via the evidence lower bound (ELBO):

        ELBO = E_q[log p(y|x,w)] - beta * KL(q(w) || p(w))

    where beta controls the complexity cost weighting (set to 1/N_train for
    standard Bayesian treatment, or tuned as a hyperparameter).

    Args:
        in_dim: Input feature dimensionality.
        out_dim: Output dimensionality.
        hidden_dim: Width of hidden layers.
        n_hidden: Number of hidden layers.
        prior_sigma: Prior standard deviation for all ``BayesianLinear`` layers.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        hidden_dim: int = 64,
        n_hidden: int = 2,
        prior_sigma: float = 1.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(n_hidden):
            layers.append(BayesianLinear(dim, hidden_dim, prior_sigma=prior_sigma))
            layers.append(nn.ReLU())
            dim = hidden_dim
        layers.append(BayesianLinear(dim, out_dim, prior_sigma=prior_sigma))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with a single weight sample.

        Args:
            x: Input tensor ``(batch, in_dim)``.

        Returns:
            Predictions ``(batch, out_dim)`` or ``(batch,)`` when ``out_dim=1``.
        """
        out = self.net(x)
        if out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out

    def kl_divergence(self) -> torch.Tensor:
        """Sum KL contributions from every ``BayesianLinear`` in the network.

        Returns:
            Total KL divergence scalar.
        """
        kl = torch.tensor(0.0, device=next(self.parameters()).device)
        for module in self.modules():
            if isinstance(module, BayesianLinear):
                kl = kl + module.kl_divergence()
        return kl

    def elbo_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        n_samples: int = 3,
        beta: float = 1.0,
    ) -> torch.Tensor:
        """Compute the negative ELBO for a mini-batch.

        ELBO = (1/S) sum_{s=1}^{S} log p(y | x, w_s) - beta * KL(q || p)

        For regression the likelihood is Gaussian with unit variance so
        log p(y|x,w) proportional to -MSE.

        Args:
            x: Input batch ``(batch, in_dim)``.
            y: Target batch ``(batch,)`` or ``(batch, out_dim)``.
            n_samples: Number of MC samples for the expected log-likelihood.
            beta: KL weighting coefficient.

        Returns:
            Scalar loss (lower is better).
        """
        nll_sum = torch.tensor(0.0, device=x.device)
        for _ in range(n_samples):
            y_hat = self.forward(x)
            nll_sum = nll_sum + F.mse_loss(y_hat, y, reduction="mean")
        nll = nll_sum / n_samples
        kl = self.kl_divergence()
        return nll + beta * kl


# ---------------------------------------------------------------------------
# MC-Dropout MLP (keeps dropout on at test time)
# ---------------------------------------------------------------------------


@dataclass
class MCDropoutPrediction:
    """Container for MC-Dropout predictive statistics.

    Attributes:
        mean: Predictive mean ``(n,)``.
        variance: Total predictive variance ``(n,)``.
        samples: Raw stochastic forward-pass samples ``(n_samples, n)``.
    """

    mean: torch.Tensor
    variance: torch.Tensor
    samples: torch.Tensor


class MCDropoutMLP(nn.Module):
    """MLP that keeps dropout active at test time for MC-Dropout inference.

    At prediction time, ``n_samples`` stochastic forward passes are performed
    and the predictive mean and variance are estimated empirically:

        mu(x)  = (1/S) sum_s f(x; mask_s)
        var(x) = (1/S) sum_s (f(x; mask_s) - mu(x))^2

    Args:
        in_dim: Input feature dimensionality.
        hidden_dim: Width of hidden layers.
        n_hidden: Number of hidden layers.
        dropout: Dropout probability applied after each hidden activation.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        n_hidden: int = 2,
        dropout: float = 0.2,
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
        self.head = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single forward pass (dropout is applied if module is in train mode).

        Args:
            x: Input ``(batch, in_dim)``.

        Returns:
            Predictions ``(batch,)``.
        """
        return self.head(self.backbone(x)).squeeze(-1)

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 50,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run stochastic forward passes and return mean and variance.

        Dropout is enabled regardless of the module's current mode so that
        each forward pass produces a different mask realisation.

        Args:
            x: Input tensor ``(batch, in_dim)``.
            n_samples: Number of stochastic forward passes.

        Returns:
            Tuple of ``(mean, variance)`` each of shape ``(batch,)``.
        """
        was_training = self.training
        self.train()

        preds = torch.stack([self.forward(x) for _ in range(n_samples)], dim=0)

        if not was_training:
            self.eval()

        mean = preds.mean(dim=0)
        variance = preds.var(dim=0)
        return mean, variance
