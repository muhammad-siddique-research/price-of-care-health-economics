"""Prespecified sensitivity checks of the exploratory interaction.

These are comparisons across analytic choices, not participant-clustered causal
inference. Treatment randomization cannot be reconstructed from this subset.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from .econometrics import FORMULA_INTERACT


def check_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    high = float(np.quantile(df.mdvis, .99))
    designs = {
        "full_archived_sample": df.copy(),
        "omit_top_1pct_utilization": df.loc[df.mdvis <= high].copy(),
        "non_idp_plans_only": df.loc[df.idp == 0].copy(),
    }
    rows = []
    key = "costshare_c:disease_z"
    for name, sample in designs.items():
        formula = (
            FORMULA_INTERACT.replace(" + idp", "")
            if name == "non_idp_plans_only" else FORMULA_INTERACT
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            po = smf.glm(
                formula, data=sample, family=sm.families.Poisson()
            ).fit(cov_type="HC3")
            nb = smf.negativebinomial(formula, data=sample).fit(
                method="bfgs", maxiter=160, disp=False
            )
        for model_name, fit in (("poisson_hc3", po), ("nb2_mle", nb)):
            conv = (
                model_name == "poisson_hc3"
                or bool(fit.mle_retvals.get("converged"))
            )
            rows.append({
                "design": name, "model": model_name, "n": int(fit.nobs),
                "interaction_beta": float(fit.params[key]) if conv else np.nan,
                "interaction_se": float(fit.bse[key]) if conv else np.nan,
                "interaction_p_exploratory": (
                    float(fit.pvalues[key]) if conv else np.nan
                ),
                "aic": float(fit.aic) if conv else np.nan,
                "converged": bool(conv),
                "note": (
                    "Independent-observation standard errors; unknown original "
                    "participant clustering and assignment; not causal."
                ),
            })
    return pd.DataFrame(rows)
