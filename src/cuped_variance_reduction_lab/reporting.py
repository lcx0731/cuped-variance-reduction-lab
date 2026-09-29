"""Static figures for estimator precision and validity."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

LABELS = {
    "unadjusted": "Unadjusted",
    "cuped_pre": "CUPED (pre metric)",
    "bad_control_post": "Bad control (post metric)",
}
COLORS = {
    "unadjusted": "#718096",
    "cuped_pre": "#2b6cb0",
    "bad_control_post": "#c05621",
}


def _line_chart(
    summary: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    path: Path,
    reference: float | None = None,
) -> None:
    figure, axis = plt.subplots(figsize=(8.7, 4.8), layout="constrained")
    for method in LABELS:
        subset = summary.loc[summary["method"] == method].sort_values("correlation")
        axis.plot(
            subset["correlation"],
            subset[metric],
            marker="o",
            linewidth=2.1,
            color=COLORS[method],
            label=LABELS[method],
        )
    if reference is not None:
        axis.axhline(reference, color="#1a202c", linestyle="--", linewidth=1.4, label="Reference")
    axis.set(
        title=title,
        xlabel="Pre-period metric / outcome correlation",
        ylabel=ylabel,
        xticks=[0.0, 0.3, 0.6, 0.8],
    )
    axis.legend(frameon=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def create_figures(summary: pd.DataFrame, destination: Path) -> None:
    """Save variance, bias, and confidence-interval coverage charts."""

    plt.style.use("seaborn-v0_8-whitegrid")
    _line_chart(
        summary,
        "variance_ratio",
        "Estimator variance relative to unadjusted analysis",
        "Variance ratio",
        destination / "variance_ratio.png",
        reference=1.0,
    )
    cuped = summary.loc[summary["method"] == "cuped_pre"].sort_values("correlation")
    figure, axis = plt.subplots(figsize=(8.7, 4.8), layout="constrained")
    axis.plot(
        cuped["correlation"],
        cuped["variance_ratio"],
        marker="o",
        linewidth=2.1,
        color=COLORS["cuped_pre"],
        label="Empirical CUPED ratio",
    )
    axis.plot(
        cuped["correlation"],
        cuped["theoretical_cuped_ratio"],
        linestyle="--",
        linewidth=2.0,
        color="#2f855a",
        label="1 - correlation²",
    )
    axis.set(
        title="CUPED variance reduction matches the linear benchmark",
        xlabel="Pre-period metric / outcome correlation",
        ylabel="Variance ratio",
        xticks=[0.0, 0.3, 0.6, 0.8],
    )
    axis.legend(frameon=True)
    figure.savefig(destination / "cuped_theory_check.png", dpi=170)
    plt.close(figure)

    _line_chart(
        summary,
        "bias",
        "Post-treatment adjustment introduces bias",
        "Mean estimate minus true effect",
        destination / "bias.png",
        reference=0.0,
    )
    _line_chart(
        summary,
        "coverage_95",
        "Coverage of nominal 95% confidence intervals",
        "Empirical coverage",
        destination / "coverage.png",
        reference=0.95,
    )
