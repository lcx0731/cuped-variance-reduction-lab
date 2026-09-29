"""Monte Carlo evaluation and artifact generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .estimation import estimate_all_methods
from .reporting import create_figures
from .simulation import METHODS, simulate_experiment

CORRELATIONS = (0.0, 0.3, 0.6, 0.8)


def run_monte_carlo(
    *,
    n_replicates: int = 4000,
    n_units: int = 800,
    treatment_effect: float = 0.20,
    post_treatment_shift: float = 0.50,
    seed: int = 20260929,
) -> pd.DataFrame:
    """Generate paired estimator results over pre-outcome correlations."""

    if n_replicates <= 0:
        raise ValueError("n_replicates must be positive")
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    for correlation in CORRELATIONS:
        for replicate in range(n_replicates):
            experiment = simulate_experiment(
                n_units=n_units,
                pre_outcome_correlation=correlation,
                treatment_effect=treatment_effect,
                post_treatment_shift=post_treatment_shift,
                rng=rng,
            )
            estimates = estimate_all_methods(experiment)
            for method in METHODS:
                result = estimates[method]
                records.append(
                    {
                        "correlation": correlation,
                        "replicate": replicate,
                        "method": method,
                        **result,
                        "covers": result["ci_lower"] <= treatment_effect <= result["ci_upper"],
                        "rejects_zero": result["p_value"] < 0.05,
                    }
                )
    return pd.DataFrame.from_records(records)


def summarize_replications(
    replications: pd.DataFrame, treatment_effect: float = 0.20
) -> pd.DataFrame:
    """Summarize bias, precision, coverage, and power."""

    rows: list[dict[str, object]] = []
    grouped = replications.groupby(["correlation", "method"], sort=False)
    for (correlation, method), group in grouped:
        estimates = group["estimate"].to_numpy(dtype=float)
        empirical_sd = float(np.std(estimates, ddof=1))
        rows.append(
            {
                "correlation": correlation,
                "method": method,
                "n_replicates": len(group),
                "mean_estimate": float(np.mean(estimates)),
                "bias": float(np.mean(estimates) - treatment_effect),
                "bias_mcse": empirical_sd / np.sqrt(len(estimates)),
                "empirical_sd": empirical_sd,
                "mean_standard_error": float(group["standard_error"].mean()),
                "coverage_95": float(group["covers"].mean()),
                "coverage_mcse": float(
                    np.sqrt(group["covers"].mean() * (1.0 - group["covers"].mean()) / len(group))
                ),
                "power": float(group["rejects_zero"].mean()),
                "power_mcse": float(
                    np.sqrt(
                        group["rejects_zero"].mean()
                        * (1.0 - group["rejects_zero"].mean())
                        / len(group)
                    )
                ),
                "rmse": float(np.sqrt(np.mean(np.square(estimates - treatment_effect)))),
            }
        )
    summary = pd.DataFrame.from_records(rows)
    baseline_variances = (
        summary.loc[summary["method"] == "unadjusted", ["correlation", "empirical_sd"]]
        .assign(baseline_variance=lambda frame: frame["empirical_sd"] ** 2)
        .drop(columns="empirical_sd")
    )
    summary = summary.merge(baseline_variances, on="correlation", validate="many_to_one")
    summary["variance_ratio"] = summary["empirical_sd"] ** 2 / summary["baseline_variance"]
    summary["theoretical_cuped_ratio"] = 1.0 - summary["correlation"] ** 2
    return summary.drop(columns="baseline_variance")


def run_experiment(
    output_dir: str | Path,
    *,
    n_replicates: int = 4000,
    n_units: int = 800,
    treatment_effect: float = 0.20,
    post_treatment_shift: float = 0.50,
    seed: int = 20260929,
) -> dict[str, object]:
    """Run the audit and write tables, figures, and a JSON summary."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    replications = run_monte_carlo(
        n_replicates=n_replicates,
        n_units=n_units,
        treatment_effect=treatment_effect,
        post_treatment_shift=post_treatment_shift,
        seed=seed,
    )
    summary = summarize_replications(replications, treatment_effect=treatment_effect)
    summary.to_csv(destination / "simulation_results.csv", index=False, float_format="%.10g")
    create_figures(summary, destination)

    result: dict[str, object] = {
        "design": {
            "source": "deterministic synthetic randomized experiments",
            "seed": seed,
            "n_replicates_per_correlation": n_replicates,
            "n_units_per_experiment": n_units,
            "true_treatment_effect": treatment_effect,
            "post_treatment_shift": post_treatment_shift,
            "pre_outcome_correlations": list(CORRELATIONS),
        },
        "results": [
            {
                "correlation": float(row.correlation),
                "method": str(row.method),
                "bias": float(row.bias),
                "empirical_sd": float(row.empirical_sd),
                "variance_ratio": float(row.variance_ratio),
                "coverage_95": float(row.coverage_95),
                "power": float(row.power),
            }
            for row in summary.itertuples(index=False)
        ],
    }
    (destination / "run_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result
