"""Randomized experiment generator with pre- and post-treatment covariates."""

from __future__ import annotations

import numpy as np

METHODS = ("unadjusted", "cuped_pre", "bad_control_post")


def simulate_experiment(
    *,
    n_units: int,
    pre_outcome_correlation: float,
    treatment_effect: float,
    post_treatment_shift: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Simulate an individually randomized experiment.

    The pre metric is fixed before assignment. The post metric is affected by
    treatment and is therefore an invalid control for estimating the total effect.
    """

    if n_units < 10:
        raise ValueError("n_units must be at least 10")
    if not 0.0 <= pre_outcome_correlation < 1.0:
        raise ValueError("pre_outcome_correlation must lie in [0, 1)")

    pre_metric = rng.normal(size=n_units)
    treatment = rng.binomial(1, 0.5, size=n_units).astype(float)
    outcome_noise = rng.normal(size=n_units)
    outcome = (
        treatment_effect * treatment
        + pre_outcome_correlation * pre_metric
        + np.sqrt(1.0 - pre_outcome_correlation**2) * outcome_noise
    )
    post_metric = pre_metric + post_treatment_shift * treatment + rng.normal(
        scale=0.5, size=n_units
    )
    return {
        "treatment": treatment,
        "pre_metric": pre_metric,
        "post_metric": post_metric,
        "outcome": outcome,
    }
