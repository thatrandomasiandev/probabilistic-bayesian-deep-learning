"""Benchmark runner for uncertainty, calibration, and conformal modules."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split

from prob_ml.calibration.temperature import fit_temperature_scaling
from prob_ml.conformal.metrics import coverage, interval_width
from prob_ml.conformal.split import split_conformal_regression
from prob_ml.data.classification_dgp import ClassificationDGPConfig, generate_classification_data
from prob_ml.data.regression_dgp import RegressionDGPConfig, generate_regression_data
from prob_ml.models.trainer import TrainConfig, train_classifier
from prob_ml.uncertainty.deep_ensemble import fit_deep_ensemble
from prob_ml.uncertainty.mc_dropout import mc_dropout_predict
from prob_ml.uncertainty.metrics import (
    gaussian_interval,
    gaussian_nll,
    mpiw,
    picp,
    rmse,
    uncertainty_correlation,
)
from prob_ml.utils.seed import config_hash


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _aggregate(results: list[dict]) -> dict[str, float]:
    if not results:
        return {}
    keys = results[0].keys()
    return {
        k: float(np.mean([r[k] for r in results]))
        for k in keys
        if isinstance(results[0][k], (int, float))
    }


def _aggregate_std(results: list[dict]) -> dict[str, float]:
    if not results:
        return {}
    keys = results[0].keys()
    return {
        k: float(np.std([r[k] for r in results]))
        for k in keys
        if isinstance(results[0][k], (int, float))
    }


def _train_config(config: dict[str, Any], seed: int) -> TrainConfig:
    return TrainConfig(
        epochs=config.get("epochs", 40),
        batch_size=config.get("batch_size", 128),
        lr=config.get("lr", 1e-3),
        hidden_dim=config.get("hidden_dim", 64),
        n_hidden=config.get("n_hidden", 2),
        dropout=config.get("dropout", 0.1),
        seed=seed,
    )


def _logits(model, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(X, dtype=torch.float32)).cpu().numpy()


def run_uncertainty_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    """Compare MC dropout vs deep ensemble on heteroscedastic regression."""
    seeds = config.get("seeds", [42])
    methods = config.get("methods", ["mc_dropout", "deep_ensemble"])
    hetero_levels = config.get("heteroscedastic_levels", [0.5, 1.0, 2.0])
    n_samples = config.get("n_samples", 3000)
    alpha = config.get("alpha", 0.1)

    all_results = []
    for hetero in hetero_levels:
        for method in methods:
            seed_results = []
            for seed in seeds:
                data = generate_regression_data(
                    RegressionDGPConfig(
                        n_samples=n_samples,
                        heteroscedastic_strength=hetero,
                        seed=seed,
                    )
                )
                X_train, X_test, y_train, y_test = train_test_split(
                    data.X, data.y, test_size=0.3, random_state=seed
                )
                train_cfg = _train_config(config, seed)

                if method == "mc_dropout":
                    result = mc_dropout_predict(
                        X_train,
                        y_train,
                        X_test,
                        n_mc_samples=config.get("n_mc_samples", 25),
                        config=TrainConfig(**{**train_cfg.__dict__, "dropout": 0.2}),
                    )
                else:
                    result = fit_deep_ensemble(
                        X_train,
                        y_train,
                        X_test,
                        n_members=config.get("n_members", 5),
                        config=train_cfg,
                        seed=seed,
                    )

                lower, upper = gaussian_interval(result.mean, result.std, alpha=alpha)
                seed_results.append(
                    {
                        "rmse": rmse(result.mean, y_test),
                        "nll": gaussian_nll(result.mean, y_test, result.std),
                        "picp": picp(y_test, lower, upper),
                        "mpiw": mpiw(lower, upper),
                        "unc_corr": uncertainty_correlation(
                            result.std, y_test - result.mean
                        ),
                    }
                )

            mean = _aggregate(seed_results)
            std = _aggregate_std(seed_results)
            all_results.append(
                {
                    "method": method,
                    "heteroscedastic_strength": hetero,
                    "n_samples": n_samples,
                    **{f"{k}_mean": v for k, v in mean.items()},
                    **{f"{k}_std": v for k, v in std.items()},
                }
            )

    return {"module": "uncertainty", "results": all_results}


def run_calibration_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate temperature scaling on miscalibrated classifiers."""
    seeds = config.get("seeds", [42])
    label_noise_levels = config.get("label_noise_levels", [0.0, 0.1, 0.2])
    n_samples = config.get("n_samples", 4000)

    all_results = []
    for noise in label_noise_levels:
        seed_results = []
        for seed in seeds:
            data = generate_classification_data(
                ClassificationDGPConfig(
                    n_samples=n_samples,
                    label_noise=noise,
                    seed=seed,
                )
            )
            X_train, X_tmp, y_train, y_tmp = train_test_split(
                data.X, data.y, test_size=0.4, random_state=seed
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_tmp, y_tmp, test_size=0.5, random_state=seed
            )

            model = train_classifier(X_train, y_train, _train_config(config, seed))
            logits_val = _logits(model, X_val)
            logits_test = _logits(model, X_test)

            ts = fit_temperature_scaling(
                logits_val,
                y_val,
                logits_eval=logits_test,
                y_eval=y_test,
            )

            seed_results.append(
                {
                    "ece_before": ts.ece_before,
                    "ece_after": ts.ece_after,
                    "nll_before": ts.nll_before,
                    "nll_after": ts.nll_after,
                    "temperature": ts.temperature,
                }
            )

        mean = _aggregate(seed_results)
        std = _aggregate_std(seed_results)
        all_results.append(
            {
                "label_noise": noise,
                "n_samples": n_samples,
                **{f"{k}_mean": v for k, v in mean.items()},
                **{f"{k}_std": v for k, v in std.items()},
            }
        )

    return {"module": "calibration", "results": all_results}


def run_conformal_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate split conformal coverage and interval width."""
    seeds = config.get("seeds", [42])
    alphas = config.get("alphas", [0.05, 0.1, 0.2])
    n_samples = config.get("n_samples", 3000)

    all_results = []
    for alpha in alphas:
        seed_results = []
        for seed in seeds:
            data = generate_regression_data(
                RegressionDGPConfig(n_samples=n_samples, seed=seed)
            )
            X_train, X_tmp, y_train, y_tmp = train_test_split(
                data.X, data.y, test_size=0.4, random_state=seed
            )
            X_cal, X_test, y_cal, y_test = train_test_split(
                X_tmp, y_tmp, test_size=0.5, random_state=seed
            )

            result = split_conformal_regression(
                X_train,
                y_train,
                X_cal,
                y_cal,
                X_test,
                alpha=alpha,
                config=_train_config(config, seed),
            )

            seed_results.append(
                {
                    "coverage": coverage(y_test, result.lower, result.upper),
                    "interval_width": interval_width(result.lower, result.upper),
                    "target_coverage": 1 - alpha,
                }
            )

        mean = _aggregate(seed_results)
        std = _aggregate_std(seed_results)
        all_results.append(
            {
                "alpha": alpha,
                "n_samples": n_samples,
                **{f"{k}_mean": v for k, v in mean.items()},
                **{f"{k}_std": v for k, v in std.items()},
            }
        )

    return {"module": "conformal", "results": all_results}


def run_benchmark(
    config_path: str | Path,
    module: str = "all",
    output_dir: str | Path | None = None,
) -> Path:
    """Run benchmark(s) and write results."""
    config = load_config(config_path)
    merged = {**load_config(Path(config_path).parent / "default.yaml"), **config}

    results: dict[str, Any] = {
        "config_hash": config_hash(merged),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "modules": {},
    }

    if module in ("uncertainty", "all"):
        results["modules"]["uncertainty"] = run_uncertainty_benchmark(merged)
    if module in ("calibration", "all"):
        results["modules"]["calibration"] = run_calibration_benchmark(merged)
    if module in ("conformal", "all"):
        results["modules"]["conformal"] = run_conformal_benchmark(merged)

    out = Path(output_dir or "results")
    run_dir = out / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    from prob_ml.evaluation.report import write_report

    write_report(results, run_dir / "summary.md")

    return run_dir
