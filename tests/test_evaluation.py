import json

from cuped_variance_reduction_lab.evaluation import (
    run_experiment,
    run_monte_carlo,
    summarize_replications,
)


def test_high_correlation_reduces_cuped_variance() -> None:
    replications = run_monte_carlo(n_replicates=300, n_units=500, seed=31)
    summary = summarize_replications(replications)
    high = summary[summary["correlation"] == 0.8].set_index("method")
    assert high.loc["unadjusted", "variance_ratio"] == 1.0
    assert high.loc["cuped_pre", "variance_ratio"] < 0.5
    assert abs(high.loc["cuped_pre", "bias"]) < 0.02


def test_small_end_to_end_run_writes_outputs(tmp_path) -> None:
    summary = run_experiment(
        output_dir=tmp_path,
        n_replicates=20,
        n_units=300,
        seed=37,
    )
    assert summary["design"]["n_replicates_per_correlation"] == 20
    json.loads((tmp_path / "run_summary.json").read_text())
    for name in [
        "simulation_results.csv",
        "run_summary.json",
        "variance_ratio.png",
        "cuped_theory_check.png",
        "bias.png",
        "coverage.png",
    ]:
        assert (tmp_path / name).is_file()
