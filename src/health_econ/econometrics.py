"""Exploratory econometric models on historical RAND HIE observations.

CAUTION: The original experiment was randomized, but this reduced public dataset
lacks participant IDs, timing and full assignment metadata. Associations here
are NOT identified randomized treatment effects; standard errors may be too
narrow if individuals contribute multiple correlated observations.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm

FORMULA_BASE = (
    "mdvis ~ costshare_c + disease_z + idp + lpi + fmde + "
    "physlm + hlthg + hlthf + hlthp"
)
FORMULA_INTERACT = (
    "mdvis ~ costshare_c * disease_z + idp + lpi + fmde + "
    "physlm + hlthg + hlthf + hlthp"
)


def estimate_models(df: pd.DataFrame):
    null = smf.glm(
        FORMULA_BASE, data=df, family=sm.families.Poisson()
    ).fit(cov_type="HC3")
    interaction = smf.glm(
        FORMULA_INTERACT, data=df, family=sm.families.Poisson()
    ).fit(cov_type="HC3")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            nb = smf.negativebinomial(FORMULA_INTERACT, data=df).fit(
                method="bfgs", disp=False, maxiter=200
            )
            if not nb.mle_retvals.get("converged", False):
                nb = None
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            nb = None
    logit = smf.glm(
        FORMULA_INTERACT.replace("mdvis ~", "any_visit ~"),
        data=df, family=sm.families.Binomial()
    ).fit(cov_type="HC3")
    # This is a separate binary response, not a formal hurdle count model.
    return {
        "poisson_base": null,
        "poisson_interaction": interaction,
        "nb2_interaction": nb,
        "any_visit_logit": logit,
    }


def coefficient_table(models: dict) -> pd.DataFrame:
    rec = []
    for name, fit in models.items():
        if fit is None:
            continue
        for term in fit.params.index:
            beta = float(fit.params[term])
            se = float(fit.bse[term])
            rec.append({
                "model": name, "term": term, "beta": beta, "std_error": se,
                "95ci_low": beta - 1.96 * se,
                "95ci_high": beta + 1.96 * se,
                "exp_beta": float(np.exp(beta)) if term != "alpha" else np.nan,
                "p_value_exploratory": float(fit.pvalues[term]),
                "aic": float(fit.aic), "n": int(fit.nobs),
            })
    return pd.DataFrame.from_records(rec)


def model_diagnostics(df: pd.DataFrame, models: dict) -> dict:
    out = {}
    for key, fit in models.items():
        if fit is None:
            out[key] = {"converged": False}
            continue
        fitted = np.maximum(np.asarray(fit.predict(df), dtype=float), 1e-9)
        summary = {
            "n": int(fit.nobs), "aic": float(fit.aic), "converged": True
        }
        if key != "any_visit_logit":
            if key == "nb2_interaction":
                alpha = float(fit.params["alpha"])
                variance = fitted + alpha * fitted**2
                zero_prob = (1 + alpha * fitted)**(-1 / alpha)
                variance_label = "NB2: mean + alpha * mean_squared"
            else:
                variance = fitted
                zero_prob = np.exp(-fitted)
                variance_label = "Poisson: variance equals conditional mean"
            summary.update({
                "pearson_dispersion": float(
                    np.sum((df.mdvis - fitted)**2 / variance)
                    / (len(df) - len(fit.params))
                ),
                "variance_assumption": variance_label,
                "mean_predicted_count": float(fitted.mean()),
                "observed_zero_fraction": float((df.mdvis == 0).mean()),
                "expected_model_zero_fraction": float(zero_prob.mean()),
            })
        out[key] = summary
    return out


def heterogeneity_table(df: pd.DataFrame, model) -> pd.DataFrame:
    """HC3 delta-method changes per one-unit ln(coinsurance+1) difference."""
    beta = model.params
    cov = model.cov_params()
    key = "costshare_c:disease_z"
    rows = []
    for pct in (10, 25, 50, 75, 90):
        burden = float(np.percentile(df.disease_z, pct))
        slope = float(beta["costshare_c"] + burden * beta[key])
        var = (
            cov.loc["costshare_c", "costshare_c"]
            + burden**2 * cov.loc[key, key]
            + 2 * burden * cov.loc["costshare_c", key]
        )
        se = np.sqrt(max(float(var), 0.0))
        rows.append({
            "disease_percentile": pct,
            "disease_index": float(np.percentile(df.disea, pct)),
            "ln_costshare_slope": slope,
            "conditional_count_ratio": float(np.exp(slope)),
            "ci_low_hc3": float(np.exp(slope - norm.ppf(.975) * se)),
            "ci_high_hc3": float(np.exp(slope + norm.ppf(.975) * se)),
        })
    return pd.DataFrame(rows)


def descriptive_table(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(
        ["coinsurance_pct_approx", "disease_tertile"], observed=True
    ).agg(
        n=("mdvis", "size"), mean_visits=("mdvis", "mean"),
        any_visit_fraction=("any_visit", "mean"),
        disease_index_mean=("disea", "mean")
    ).reset_index()
    return grouped
