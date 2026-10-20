"""Price functional-form sensitivity with observed non-IDP plan overlap.

0/25/50/95 coinsurance levels appear under the non-individual-deductible plan.
The 100% category exists only in a separate plan; its naive inclusion would
confound structural benefit design with the price parameter.
"""
from __future__ import annotations
from pathlib import Path
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

COVARIATES = "lpi + fmde + physlm + hlthg + hlthf + hlthp"
LINEAR = "mdvis ~ costshare_c * disease_z + " + COVARIATES
CATEGORICAL = (
    "mdvis ~ C(coinsurance_pct_approx, Treatment(reference=0)) * disease_z + "
    + COVARIATES
)


def analyze_functional_form(df: pd.DataFrame, results_dir: Path, figures_dir: Path):
    """Compare log-price versus categories using Poisson HC3 and NB2 likelihood."""
    sub = df.loc[df.idp.eq(0)].copy()
    observed = set(sub.coinsurance_pct_approx.unique().tolist())
    if observed != {0, 25, 50, 95}:
        raise ValueError(f"Unexpected non-IDP coinsurance categories: {observed}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        po_log = smf.glm(
            LINEAR, sub, family=sm.families.Poisson()
        ).fit(cov_type="HC3")
        po_cat = smf.glm(
            CATEGORICAL, sub, family=sm.families.Poisson()
        ).fit(cov_type="HC3")
        nb_log = smf.negativebinomial(LINEAR, sub).fit(
            method="bfgs", maxiter=180, disp=False
        )
        nb_cat = smf.negativebinomial(CATEGORICAL, sub).fit(
            method="bfgs", maxiter=180, disp=False
        )
    rows = []
    for family, price, model in [
        ("Poisson HC3", "continuous log costshare", po_log),
        ("Poisson HC3", "categorical coinsurance", po_cat),
        ("NB2 MLE", "continuous log costshare", nb_log),
        ("NB2 MLE", "categorical coinsurance", nb_cat),
    ]:
        converged = (
            family == "Poisson HC3"
            or bool(model.mle_retvals.get("converged", False))
        )
        n = len(sub)
        k = len(model.params)
        rows.append({
            "model": family,
            "price_functional_form": price,
            "observations": n,
            "converged": bool(converged),
            "n_parameters": k,
            "aic": float(model.aic) if converged else np.nan,
            "bic_information": float(-2 * model.llf + np.log(n) * k)
            if converged else np.nan,
            "interpretive_scope": "same-sample descriptive fit; missing trial IDs",
        })
    comparison = pd.DataFrame(rows)
    records = []
    for burden, frame in sub.groupby("disease_tertile", observed=True):
        for coins in (0, 25, 50, 95):
            altered = frame.copy()
            altered["coinsurance_pct_approx"] = coins
            altered["lncoins"] = np.log1p(coins)
            altered["costshare_c"] = altered.lncoins - float(df.lncoins.mean())
            log_means = np.asarray(nb_log.predict(altered), dtype=float)
            cat_means = np.asarray(nb_cat.predict(altered), dtype=float)
            records.append({
                "burden_third": str(burden),
                "coinsurance_pct": coins,
                "standardization_n": len(frame),
                "nb2_log_price_adjusted_mean": float(log_means.mean()),
                "nb2_categorical_price_adjusted_mean": float(cat_means.mean()),
                "scope": "descriptive standardized model outputs; not causal",
            })
    margins = pd.DataFrame(records)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(
        results_dir / "price_functional_form_comparisons.csv", index=False
    )
    margins.to_csv(
        results_dir / "price_support_sensitivity_margins.csv", index=False
    )
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    colors = {"Low": "#1d4ed8", "Middle": "#0d9488", "High": "#c2410c"}
    for label, grp in margins.groupby("burden_third", sort=False):
        grp = grp.sort_values("coinsurance_pct")
        ax.plot(
            grp.coinsurance_pct, grp.nb2_categorical_price_adjusted_mean,
            marker="o", lw=2.4, color=colors[label], label=f"{label} burden"
        )
        ax.plot(
            grp.coinsurance_pct, grp.nb2_log_price_adjusted_mean,
            lw=1.4, linestyle="--", alpha=.6, color=colors[label]
        )
    ax.set_xticks([0, 25, 50, 95])
    ax.set_xlabel("Recorded non-IDP coinsurance (%)")
    ax.set_ylabel("NB2 expected visits (model standardized)")
    ax.set_title(
        "Price form sensitivity: solid categorical vs dashed log-price",
        fontsize=12, fontweight="semibold"
    )
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=.18)
    fig.text(
        .5, .01,
        "Historical observational associations; no patient IDs; not causal",
        ha="center", fontsize=10
    )
    fig.tight_layout(rect=(0, .035, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(
            figures_dir / f"figure_8_price_functional_form.{ext}",
            dpi=180, bbox_inches="tight"
        )
    plt.close(fig)
    return comparison, margins
