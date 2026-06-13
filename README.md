# Probabilistic & Bayesian Deep Learning

A research benchmark suite for **predictive uncertainty quantification**, **post-hoc calibration**, and **distribution-free conformal prediction** — three complementary approaches to reliable probabilistic inference from neural networks. All experiments use synthetic DGPs with known noise structure, enabling exact evaluation of coverage, calibration, and uncertainty quality.

The central research question: *when should we trust a model's predictions, and how do different uncertainty methods behave under heteroscedastic noise, label noise, and distribution shift?*

---

## Research scope

| Module | Problem | Methods | Primary metrics |
|--------|---------|---------|-----------------|
| **Uncertainty** | Predictive distributions for regression | MC Dropout, deep ensembles | RMSE, NLL, PICP, MPIW, uncertainty correlation |
| **Calibration** | Align predicted probabilities with empirical frequencies | Temperature scaling | ECE, NLL, Brier score |
| **Conformal** | Finite-sample prediction intervals with coverage guarantees | Split conformal prediction | Coverage, interval width |

---

## Module 1: Uncertainty quantification

### Problem formulation

Given training data, produce not just point predictions ŷ but **predictive distributions** p(y|x) that capture both **aleatoric uncertainty** (irreducible noise) and **epistemic uncertainty** (model ignorance). Gal & Ghahramani (2016) and Lakshminarayanan et al. (2017) provide the two most widely used deep learning approaches.

### Implemented methods

| Method | Mechanism | Reference |
|--------|-----------|-----------|
| **MC Dropout** | Run T forward passes with dropout active; variance ≈ epistemic uncertainty | Gal & Ghahramani (2016) |
| **Deep ensembles** | Train M independent networks; disagreement ≈ epistemic uncertainty | Lakshminarayanan et al. (2017) |

MC Dropout treats dropout as approximate Bayesian inference over network weights (Gal & Ghahramani, 2016). Deep ensembles take a frequentist route: independently trained models capture different modes of the loss landscape, and their disagreement signals epistemic uncertainty (Lakshminarayanan et al., 2017).

### Synthetic DGP (`data/regression_dgp.py`)

- Nonlinear mean μ(x) with **heteroscedastic noise** σ(x) growing with |x₀|, |x₁|
- Optional OOD test region for evaluating uncertainty extrapolation
- Ground-truth σ(x) available for calibration assessment

### Evaluation metrics

- **RMSE:** Point prediction accuracy
- **NLL:** Negative log-likelihood under Gaussian predictive distribution
- **PICP** (prediction interval coverage probability): Fraction of targets within predicted interval
- **MPIW** (mean prediction interval width): Average interval width — coverage-width tradeoff
- **Uncertainty correlation:** Rank correlation between predicted uncertainty and absolute error

---

## Module 2: Calibration

### Problem formulation

Modern neural classifiers are systematically **overconfident** (Guo et al., 2017). A perfectly calibrated classifier satisfies:

$$\mathbb{P}(\hat{Y} = Y \mid \hat{p} = p) = p \quad \forall p \in [0,1]$$

### Implemented method

**Temperature scaling** (Guo et al., 2017): fit a single scalar T on a held-out validation set:

$$\hat{p}_i = \text{softmax}(z_i / T)$$

This is a post-hoc method that does not require retraining and preserves accuracy while improving calibration.

### Synthetic DGP (`data/classification_dgp.py`)

Binary classification with tunable **label noise** to induce miscalibration in the base model, providing a controlled setting where temperature scaling's benefit is measurable.

### Evaluation metrics

- **ECE** (expected calibration error): Weighted average of |accuracy − confidence| across confidence bins (Naeini et al., 2015)
- **NLL / Brier score:** Proper scoring rules for probabilistic predictions

---

## Module 3: Conformal prediction

### Problem formulation

Conformal prediction (Vovk et al., 2005) provides **distribution-free, finite-sample coverage guarantees** without assuming a parametric noise model. Given a calibration set, construct prediction intervals C(x) such that:

$$\mathbb{P}(Y \in C(X)) \geq 1 - \alpha$$

under the sole assumption of **exchangeability** between calibration and test data.

### Implemented method

**Split conformal prediction** (Papadopoulos et al., 2002; Lei et al., 2018):

1. Train model on proper training set
2. Compute nonconformity scores |yᵢ − ŷᵢ| on calibration set
3. At test time, interval = [ŷ − q, ŷ + q] where q is the (1−α) quantile of calibration scores

### Evaluation metrics

- **Coverage:** Empirical fraction of test targets within predicted intervals (should ≥ 1−α)
- **Interval width:** Average width — tighter is better at fixed coverage

---

## Benchmark protocol

```bash
pip install -e ".[dev]"

python scripts/run_benchmark.py --config configs/uncertainty_benchmark.yaml --module all
python scripts/run_benchmark.py --config configs/uncertainty_benchmark.yaml --module uncertainty
python scripts/run_benchmark.py --config configs/calibration_benchmark.yaml --module calibration
python scripts/run_benchmark.py --config configs/conformal_benchmark.yaml --module conformal

pytest
```

Configs sweep heteroscedasticity levels, label noise rates, and significance levels α ∈ {0.05, 0.1, 0.2}.

---

## Project layout

```
src/prob_ml/
├── data/          # Heteroscedastic regression and noisy classification DGPs
├── models/        # PyTorch MLP backbones and trainers
├── uncertainty/   # MC Dropout, deep ensembles, uncertainty metrics
├── calibration/   # Temperature scaling, ECE, reliability diagrams
├── conformal/     # Split conformal regression
└── evaluation/    # Benchmark runner and reporting
```

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_synthetic_dgp_walkthrough.ipynb` | Validate noise structure and OOD regions |
| `02_uncertainty_estimation.ipynb` | MC Dropout vs. deep ensemble comparison |
| `03_calibration.ipynb` | Reliability diagrams and temperature scaling |
| `04_conformal_prediction.ipynb` | Coverage guarantees across α levels |

---

## Implementation notes

- MC Dropout requires dropout layers active at **both** train and test time
- Conformal intervals use **symmetric absolute-residual** scores; asymmetric or adaptive methods (Romano et al., 2019) are not included
- Deep ensembles use M independent training runs with different seeds

---

## References

- Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. *ICML*. [arXiv](https://arxiv.org/abs/1506.02142)
- Guo, C., et al. (2017). On calibration of modern neural networks. *ICML*. [arXiv](https://arxiv.org/abs/1706.04599)
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS*. [arXiv](https://arxiv.org/abs/1612.01474)
- Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018). Distribution-free predictive inference for regression. *JASA*, 113(523), 1094–1111. [DOI](https://doi.org/10.1080/01621459.2017.1307116)
- Naeini, M. P., Cooper, G., & Hauskrecht, M. (2015). Obtaining well calibrated probabilities using Bayesian binning. *AAAI*. [arXiv](https://arxiv.org/abs/1412.2475)
- Papadopoulos, H., Proedrou, K., Vovk, V., & Gammerman, A. (2002). Inductive confidence machines for regression. *ECML*. [DOI](https://doi.org/10.1007/3-540-36755-1_30)
- Romano, Y., Patterson, E., & Candès, E. (2019). Conformalized quantile regression. *NeurIPS*. [arXiv](https://arxiv.org/abs/1905.12929)
- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer. [DOI](https://doi.org/10.1007/3-540-31800-0)

---

## Future work

- Variational inference (Blundell et al., 2015; Bayes-by-Backprop)
- Conformal classification: APS, RAPS (Romano et al., 2020)
- OOD detection with covariate shift DGPs (Ovadia et al., 2019)
