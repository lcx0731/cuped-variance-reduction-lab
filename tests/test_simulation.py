import numpy as np

from cuped_variance_reduction_lab.simulation import simulate_experiment


def test_simulation_is_reproducible_and_randomized() -> None:
    arguments = dict(
        n_units=5000,
        pre_outcome_correlation=0.6,
        treatment_effect=0.2,
        post_treatment_shift=0.5,
    )
    first = simulate_experiment(**arguments, rng=np.random.default_rng(7))
    second = simulate_experiment(**arguments, rng=np.random.default_rng(7))
    for key in first:
        np.testing.assert_allclose(first[key], second[key])
    treated = first["treatment"] == 1
    assert abs(first["pre_metric"][treated].mean() - first["pre_metric"][~treated].mean()) < 0.08


def test_post_metric_is_treatment_affected_but_pre_metric_is_not() -> None:
    experiment = simulate_experiment(
        n_units=50000,
        pre_outcome_correlation=0.6,
        treatment_effect=0.2,
        post_treatment_shift=0.5,
        rng=np.random.default_rng(13),
    )
    treated = experiment["treatment"] == 1
    pre_gap = experiment["pre_metric"][treated].mean() - experiment["pre_metric"][~treated].mean()
    post_gap = experiment["post_metric"][treated].mean() - experiment["post_metric"][~treated].mean()
    assert abs(pre_gap) < 0.03
    assert abs(post_gap - 0.5) < 0.03
