"""Treatment-effect estimation with heteroskedasticity-robust uncertainty."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def estimate_treatment_effect(
    outcome: np.ndarray,
    treatment: np.ndarray,
    covariate: np.ndarray | None = None,
    confidence_level: float = 0.95,
) -> dict[str, float]:
    """Estimate the treatment coefficient by OLS with an HC1 standard error."""

    y = np.asarray(outcome, dtype=float)
    assignment = np.asarray(treatment, dtype=float)
    if y.ndim != 1 or assignment.shape != y.shape:
        raise ValueError("outcome and treatment must be aligned vectors")
    columns = [np.ones(len(y)), assignment]
    if covariate is not None:
        values = np.asarray(covariate, dtype=float)
        if values.shape != y.shape:
            raise ValueError("covariate must align with the outcome")
        columns.append(values - values.mean())
    design = np.column_stack(columns)

    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank != design.shape[1]:
        raise ValueError("design matrix is rank deficient")
    residuals = y - design @ coefficients
    bread = np.linalg.inv(design.T @ design)
    score_rows = design * residuals[:, None]
    meat = score_rows.T @ score_rows
    n_observations, n_parameters = design.shape
    covariance = (n_observations / (n_observations - n_parameters)) * bread @ meat @ bread
    standard_error = float(np.sqrt(max(covariance[1, 1], 0.0)))

    estimate = float(coefficients[1])
    critical_value = float(norm.ppf(0.5 + confidence_level / 2.0))
    z_statistic = estimate / standard_error if standard_error > 0.0 else np.inf
    return {
        "estimate": estimate,
        "standard_error": standard_error,
        "ci_lower": estimate - critical_value * standard_error,
        "ci_upper": estimate + critical_value * standard_error,
        "p_value": float(2.0 * norm.sf(abs(z_statistic))),
    }


def estimate_all_methods(experiment: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Fit unadjusted, valid pre-period, and invalid post-period specifications."""

    common = {
        "outcome": experiment["outcome"],
        "treatment": experiment["treatment"],
    }
    return {
        "unadjusted": estimate_treatment_effect(**common),
        "cuped_pre": estimate_treatment_effect(**common, covariate=experiment["pre_metric"]),
        "bad_control_post": estimate_treatment_effect(
            **common, covariate=experiment["post_metric"]
        ),
    }
