import numpy as np

from cuped_variance_reduction_lab.estimation import (
    estimate_all_methods,
    estimate_treatment_effect,
)
from cuped_variance_reduction_lab.simulation import simulate_experiment


def test_adjusted_ols_recovers_exact_coefficient_in_noiseless_data() -> None:
    treatment = np.tile([0.0, 1.0], 50)
    covariate = np.linspace(-2.0, 2.0, 100)
    outcome = 1.0 + 0.4 * treatment + 2.0 * covariate
    result = estimate_treatment_effect(outcome, treatment, covariate)
    assert abs(result["estimate"] - 0.4) < 1e-10


def test_bad_control_is_biased_in_large_sample() -> None:
    experiment = simulate_experiment(
        n_units=100000,
        pre_outcome_correlation=0.8,
        treatment_effect=0.2,
        post_treatment_shift=0.5,
        rng=np.random.default_rng(23),
    )
    results = estimate_all_methods(experiment)
    assert abs(results["cuped_pre"]["estimate"] - 0.2) < 0.02
    assert results["bad_control_post"]["estimate"] < 0.05


def test_robust_standard_error_and_interval_are_finite() -> None:
    rng = np.random.default_rng(29)
    treatment = rng.binomial(1, 0.5, size=500)
    outcome = 0.2 * treatment + rng.normal(size=500)
    result = estimate_treatment_effect(outcome, treatment)
    assert result["standard_error"] > 0.0
    assert result["ci_lower"] < result["estimate"] < result["ci_upper"]
    assert 0.0 <= result["p_value"] <= 1.0
