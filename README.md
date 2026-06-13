# Probabilistic & Bayesian Deep Learning

PhD-level probabilistic deep learning suite covering **uncertainty quantification**, **calibration**, and **conformal prediction** — all evaluated on synthetic data with known ground truth.

## Modules

| Module | Description | Key metrics |
|--------|-------------|-------------|
| **Uncertainty** | MC Dropout + Deep Ensembles on heteroscedastic regression | RMSE, NLL, PICP, MPIW, uncertainty correlation |
| **Calibration** | Post-hoc temperature scaling for neural classifiers | ECE, NLL, Brier score |
| **Conformal** | Split conformal prediction with finite-sample guarantees | Coverage, interval width |

## Assumptions

- **Uncertainty:** Gaussian predictive approximation; MC dropout requires dropout at train and test time
- **Calibration:** Binary classification; temperature scaling assumes i.i.d. validation set
- **Conformal:** Exchangeability between calibration and test data; symmetric absolute-residual scores

## Setup

```bash
cd 04-probabilistic-bayesian-deep-learning
pip install -e ".[dev]"
```

## Run benchmarks

```bash
# All modules
python scripts/run_benchmark.py --config configs/uncertainty_benchmark.yaml --module all

# Individual modules
python scripts/run_benchmark.py --config configs/uncertainty_benchmark.yaml --module uncertainty
python scripts/run_benchmark.py --config configs/calibration_benchmark.yaml --module calibration
python scripts/run_benchmark.py --config configs/conformal_benchmark.yaml --module conformal
```

Results are written to `results/{timestamp}/metrics.json` and `summary.md`.

## Run tests

```bash
pytest
```

## Project layout

```
src/prob_ml/
├── data/          # Synthetic regression & classification DGPs
├── models/        # PyTorch MLP backbones and trainers
├── uncertainty/   # MC Dropout, deep ensembles, metrics
├── calibration/   # Temperature scaling and calibration metrics
├── conformal/     # Split conformal regression
└── evaluation/    # Benchmark runner and reporting
```

## Notebooks

- `notebooks/01_synthetic_dgp_walkthrough.ipynb` — validate DGPs
- `notebooks/02_uncertainty_estimation.ipynb` — MC Dropout vs deep ensemble
- `notebooks/03_calibration.ipynb` — reliability diagrams and temperature scaling
- `notebooks/04_conformal_prediction.ipynb` — coverage guarantees

## Future work

- Variational inference (Bayes-by-Backprop) and SWAG
- Conformal classification (APS, RAPS)
- OOD detection benchmarks with covariate shift DGP
- Real datasets (UCI, CIFAR-10) via the same dataset interface
