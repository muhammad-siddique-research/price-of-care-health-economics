"""A real-data two-part hurdle count model: logit(any care) + zero-truncated NB2.

The historical RAND research extract lacks participant IDs, assignment records,
and observation dates: output is descriptive, not a randomized causal estimate.
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.truncated_model import TruncatedLFNegativeBinomialP

TERMS = "costshare_c * disease_z + idp + lpi + fmde + physlm + hlthg + hlthf + hlthp"


def fit_hurdle(df: pd.DataFrame):
    """Maximum likelihood on the same complete sample: logit + truncated NB2."""
    if not df.mdvis.ge(0).all() or not np.isfinite(df.mdvis).all():
        raise ValueError("Hurdle requires finite nonnegative claim counts")
    if not df.mdvis.gt(0).any() or not df.mdvis.eq(0).any():
        raise ValueError("Both positive and zero outcomes are required")
    binary = smf.glm(
        f"any_visit ~ {TERMS}", df, family=sm.families.Binomial()
    ).fit(cov_type="HC3")
    positive = df.loc[df.mdvis.gt(0)]
    design = patsy.dmatrix(TERMS, positive, return_type="dataframe")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        truncated = TruncatedLFNegativeBinomialP(
            positive.mdvis.to_numpy(), design.to_numpy(dtype=float),
            truncation=0, p=2
        ).fit(method="bfgs", disp=0, maxiter=200)
    if not truncated.mle_retvals.get("converged", False):
        raise RuntimeError("Truncated NB2 did not converge")
    if not np.isfinite(truncated.params).all() or truncated.params[-1] <= 0:
        raise RuntimeError("Invalid positive-part dispersion")
    return binary, truncated, design.design_info


def hurdle_predict(binary, truncated, design_info, frame: pd.DataFrame):
    """Combine Pr(Y>0) with E(Y|Y>0) for the full hurdle mean."""
    design = np.asarray(patsy.build_design_matrices([design_info], frame)[0])
    probability = np.asarray(binary.predict(frame), dtype=float)
    positive_mean = np.asarray(
        truncated.model.predict(truncated.params, exog=design, which="mean"),
        dtype=float,
    )
    expectation = probability * positive_mean
    if not np.isfinite(expectation).all() or not (positive_mean >= 1).all():
        raise ValueError("Invalid hurdle expected counts")
    return probability, positive_mean, expectation


def estimate_hurdle_and_margins(df: pd.DataFrame, output_dir: Path):
    """Fit full likelihood, export provenance and supported non-IDP margins."""
    binary, truncated, info = fit_hurdle(df)
    probability, positive, expected = hurdle_predict(binary, truncated, info, df)
    summary = {
        "sample_n": int(len(df)),
        "positive_n": int(df.any_visit.sum()),
        "zero_n": int((df.any_visit == 0).sum()),
        "logit_converged": bool(binary.converged),
        "truncated_nb2_converged": bool(truncated.mle_retvals.get("converged", False)),
        "truncated_nb2_alpha": float(truncated.params[-1]),
        "log_likelihood_logit": float(binary.llf),
        "log_likelihood_zero_truncated_nb2": float(truncated.llf),
        "combined_log_likelihood": float(binary.llf + truncated.llf),
        "combined_aic": float(
            -2 * (binary.llf + truncated.llf)
            + 2 * (len(binary.params) + len(truncated.params))
        ),
        "observed_zero_fraction": float((df.any_visit == 0).mean()),
        "hurdle_predicted_zero_fraction": float(np.mean(1 - probability)),
        "observed_average_visits": float(df.mdvis.mean()),
        "hurdle_predicted_average_visits": float(np.mean(expected)),
        "limitations": (
            "In-sample independent-row likelihood, unknown participant-level clustering. "
            "No randomization or time markers in the extract. No causal effects. "
            "An exactly fitted zero share is not out-of-sample validation."
        ),
    }
    nonidp = df.loc[df.idp.eq(0)].copy()
    if nonidp.empty:
        raise ValueError("No non-IDP sample")
    records = []
    for label, frame in [
        ("Lower disease-burden third", nonidp.loc[nonidp.disease_tertile.eq("Low")]),
        ("Middle disease-burden third", nonidp.loc[nonidp.disease_tertile.eq("Middle")]),
        ("Higher disease-burden third", nonidp.loc[nonidp.disease_tertile.eq("High")]),
    ]:
        if len(frame) < 30:
            continue
        for price in (0, 25, 50, 95):
            altered = frame.copy()
            altered["costshare_c"] = np.log1p(price) - float(df.lncoins.mean())
            chance, conditional, average = hurdle_predict(
                binary, truncated, info, altered
            )
            records.append({
                "disease_burden_group": label,
                "coinsurance_pct": price,
                "standardization_group_n": len(frame),
                "probability_any_visit": float(chance.mean()),
                "positive_count_mean_among_users": float(conditional.mean()),
                "expected_visits_mean": float(average.mean()),
                "inference_scope": "adjusted historical associations; not causal",
            })
    margins = pd.DataFrame(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    margins.to_csv(output_dir / "hurdle_standardized_margins.csv", index=False)
    (output_dir / "hurdle_model_diagnostics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary, margins


def plot_hurdle_margins(margins: pd.DataFrame, figures: Path):
    """Plot probability, positive-use intensity and combined expected visits."""
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharex=True)
    colors = ["#1d4ed8", "#0d9488", "#c2410c"]
    parts = [
        ("probability_any_visit", "Probability of any visits", "Extensive margin"),
        ("positive_count_mean_among_users", "Mean visits if any", "Intensity margin"),
        ("expected_visits_mean", "Mean expected visits", "Combined hurdle mean"),
    ]
    for ax, (col, ylabel, title) in zip(axes, parts):
        for color, (label, frame) in zip(
            colors, margins.groupby("disease_burden_group", sort=False)
        ):
            frame = frame.sort_values("coinsurance_pct")
            ax.plot(frame.coinsurance_pct, frame[col], marker="o", lw=2.4,
                    label=label, color=color)
        ax.set_title(title, fontweight="semibold")
        ax.set_xticks([0, 25, 50, 95])
        ax.set_xlabel("Recorded non-IDP coinsurance (%)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Historical RAND HIE: model-standardized descriptive patterns",
                 fontsize=14, fontweight="bold")
    fig.text(.5, -.02,
             "Not a causal price-response curve. Person IDs and assignment unavailable.",
             ha="center", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(figures / f"figure_7_hurdle_decomposition.{ext}",
                    dpi=180, bbox_inches="tight")
    plt.close(fig)
