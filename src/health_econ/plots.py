"""Research-facing, accessible visualizations using actual archived observations."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#15324b"
BLUE = "#2563a6"
TEAL = "#14877c"
ORANGE = "#c76724"
LIGHT = "#f3f6fa"


def _save(fig, path: Path):
    fig.patch.set_facecolor("white")
    fig.tight_layout(pad=2.1)
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_descriptive(df: pd.DataFrame, out: Path):
    visits = df.mdvis.clip(upper=16)
    counts = np.array([int((visits == i).sum()) for i in range(17)])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(17), counts, color=TEAL, width=.76)
    ax.set_xticks(
        range(0, 17, 2),
        [str(i) if i < 16 else "16+" for i in range(0, 17, 2)]
    )
    ax.set_ylabel("Number of archived observations")
    ax.set_xlabel("Reported physician outpatient visits (16+ pooled)")
    ax.set_title(
        "Physician utilization is zero-heavy and right-skewed",
        color=INK, fontweight="bold"
    )
    ax.text(
        .97, .97,
        f"n = {len(df):,} observations\nZero visits: {(df.mdvis.eq(0).mean()):.1%}",
        transform=ax.transAxes, ha="right", va="top",
        bbox=dict(facecolor=LIGHT, edgecolor="none")
    )
    _save(fig, out / "figure_1_visits_distribution.png")


def plot_heterogeneity(desc: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(10, 5.8))
    palette = {"Low": BLUE, "Middle": TEAL, "High": ORANGE}
    for name, sub in desc.groupby("disease_tertile", observed=True):
        sub = sub.sort_values("coinsurance_pct_approx")
        ax.plot(
            sub.coinsurance_pct_approx, sub.mean_visits, marker="o",
            linewidth=2.4, label=f"{name} disease-burden tertile",
            color=palette[str(name)]
        )
        for row in sub.itertuples():
            if row.n < 100:
                ax.annotate(
                    f"n={row.n}", (row.coinsurance_pct_approx, row.mean_visits),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=8
                )
    ax.set_title(
        "Observed physician visits by cost-sharing and disease burden",
        color=INK, fontweight="bold"
    )
    ax.set_xlabel("Recorded insurance coinsurance level (%)")
    ax.set_ylabel("Mean outpatient physician visits per observation")
    ax.legend(frameon=False)
    ax.grid(alpha=.13)
    _save(fig, out / "figure_2_costshare_disease_heterogeneity.png")


def plot_margins(margins: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    x = margins.disease_percentile.astype(float)
    y = margins.conditional_count_ratio
    low = y - margins.ci_low_hc3
    high = margins.ci_high_hc3 - y
    ax.errorbar(
        x, y, yerr=[low, high], marker="o", capsize=5,
        elinewidth=1.7, color=BLUE, linewidth=2.5
    )
    ax.axhline(1, ls="--", c=ORANGE, lw=1.5)
    ax.set_xticks(x, [f"P{int(v)}" for v in x])
    ax.set_xlabel("Percentile of archived chronic-disease measure")
    ax.set_ylabel("Conditional count ratio per +1 ln(coinsurance + 1)")
    ax.set_title(
        "Exploratory heterogeneity: Poisson interaction model",
        color=INK, fontweight="bold"
    )
    ax.text(
        .01, .01, "HC3 intervals assume independent observations; IDs unavailable.",
        transform=ax.transAxes, fontsize=9, color="#475569"
    )
    ax.grid(alpha=.13)
    _save(fig, out / "figure_3_interaction_margins.png")


def plot_ml(metrics: pd.DataFrame, importance: pd.DataFrame,
            cal: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ordered = metrics.sort_values("mean_poisson_deviance")
    ax.barh(ordered.model, ordered.mean_poisson_deviance,
            color=[TEAL, BLUE, ORANGE])
    ax.invert_yaxis()
    ax.set_xlabel("Held-out mean Poisson deviance (lower is better)")
    ax.set_title(
        "Model comparison: covariate-pattern-group holdout",
        color=INK, fontweight="bold"
    )
    for i, v in enumerate(ordered.mean_poisson_deviance):
        ax.text(v + .03, i, f"{v:.3f}", va="center")
    ax.set_xlim(0, ordered.mean_poisson_deviance.max() * 1.22)
    _save(fig, out / "figure_4_ml_comparison.png")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    top = importance.sort_values("increase_in_poisson_deviance_mean").tail(9)
    ax.barh(top.feature, top.increase_in_poisson_deviance_mean, color=BLUE)
    ax.axvline(0, c=ORANGE, ls="--")
    ax.set_xlabel("Increase in held-out Poisson deviance after shuffling feature")
    ax.set_title(
        "Model-reliance analysis, NOT causal importance",
        color=INK, fontweight="bold"
    )
    _save(fig, out / "figure_5_permutation_importance.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(cal.predicted_mean, cal.observed_mean, s=60, color=TEAL)
    for row in cal.itertuples():
        ax.annotate(
            str(row.bin), (row.predicted_mean, row.observed_mean),
            xytext=(6, 4), textcoords="offset points"
        )
    lo = min(cal.predicted_mean.min(), cal.observed_mean.min())
    hi = max(cal.predicted_mean.max(), cal.observed_mean.max())
    ax.plot([lo, hi], [lo, hi], "--", color=ORANGE)
    ax.set_xlabel("Mean predicted visits in prediction quintile")
    ax.set_ylabel("Mean observed visits in prediction quintile")
    ax.set_title("Out-of-pattern-sample calibration", color=INK, fontweight="bold")
    _save(fig, out / "figure_6_prediction_calibration.png")
