"""Leakage-aware predictive benchmarking, not identification of effects."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_poisson_deviance, root_mean_squared_error
)
from sklearn.inspection import permutation_importance

FEATURES = [
    "lncoins", "idp", "lpi", "fmde", "physlm", "disea", "hlthg", "hlthf", "hlthp"
]


def group_split(df: pd.DataFrame, seed: int = 2026):
    """Keep identical covariate fingerprints in one split.

    This is a record-overlap check, NOT valid patient-level separation,
    because the archive has no participant IDs.
    """
    groups = pd.util.hash_pandas_object(df[FEATURES], index=False).to_numpy()
    train_idx, test_idx = next(GroupShuffleSplit(
        n_splits=1, test_size=.25, random_state=seed
    ).split(df[FEATURES], df.mdvis, groups=groups))
    assert not set(groups[train_idx]).intersection(set(groups[test_idx]))
    return train_idx, test_idx


def evaluate_ml(df: pd.DataFrame, seed: int = 2026):
    train, test = group_split(df, seed)
    x_train = df.iloc[train][FEATURES].copy()
    x_test = df.iloc[test][FEATURES].copy()
    y_train = df.iloc[train]["mdvis"].to_numpy()
    y_test = df.iloc[test]["mdvis"].to_numpy()
    models = {
        "Training-mean baseline": DummyRegressor(strategy="mean"),
        "Regularized Poisson GLM": make_pipeline(
            StandardScaler(), PoissonRegressor(alpha=.05, max_iter=1200)
        ),
        "Gradient-boosted Poisson trees": HistGradientBoostingRegressor(
            loss="poisson", learning_rate=.06, max_iter=180, max_leaf_nodes=15,
            min_samples_leaf=45, l2_regularization=4.0, random_state=seed,
            early_stopping=False
        ),
    }
    metrics = []
    preds = {}
    for name, estimator in models.items():
        estimator.fit(x_train, y_train)
        yhat = np.clip(estimator.predict(x_test), 1e-8, None)
        preds[name] = yhat
        metrics.append({
            "model": name, "train_n": len(train), "test_n": len(test),
            "mean_absolute_error": float(mean_absolute_error(y_test, yhat)),
            "root_mean_squared_error": float(root_mean_squared_error(y_test, yhat)),
            "mean_poisson_deviance": float(mean_poisson_deviance(y_test, yhat)),
            "mean_predicted_count": float(yhat.mean()),
            "mean_observed_count": float(y_test.mean()),
        })
    boosted = models["Gradient-boosted Poisson trees"]
    # Permutation-based predictive reliance, not causal feature effects.
    pi = permutation_importance(
        boosted, x_test, y_test, n_repeats=4, random_state=seed,
        scoring="neg_mean_poisson_deviance", n_jobs=1
    )
    importance = pd.DataFrame({
        "feature": FEATURES,
        "increase_in_poisson_deviance_mean": pi.importances_mean,
        "increase_in_poisson_deviance_sd": pi.importances_std
    }).sort_values("increase_in_poisson_deviance_mean", ascending=False)
    calibration = pd.DataFrame({
        "observed": y_test, "predicted": preds["Gradient-boosted Poisson trees"]
    })
    calibration["bin"] = pd.qcut(
        calibration.predicted.rank(method="first"), 5,
        labels=[f"Q{i}" for i in range(1, 6)]
    )
    calibration = calibration.groupby("bin", observed=True).agg(
        n=("observed", "size"), observed_mean=("observed", "mean"),
        predicted_mean=("predicted", "mean")
    ).reset_index()
    split = {
        "seed": seed, "train_n": int(len(train)), "test_n": int(len(test)),
        "split_unit": "Identical-covariate-pattern groups",
        "participant_id_available": False,
        "caution": (
            "Group split is NOT verified patient-level holdout, and patterns "
            "may be correlated across groups. No randomization identifiers "
            "or experiment time markers were available."
        ),
        "n_unique_fingerprints": int(
            pd.util.hash_pandas_object(df[FEATURES], index=False).nunique()
        ),
    }
    return pd.DataFrame(metrics), importance, calibration, split
