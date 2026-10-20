"""Read and validate the HISTORICAL, real RAND HIE research sample."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

REQUIRED = (
    "mdvis", "lncoins", "idp", "lpi", "fmde", "physlm", "disea",
    "hlthg", "hlthf", "hlthp",
)


def load_data(path: Path, manifest: Path | None = None) -> pd.DataFrame:
    """Load real archived sample. Check integrity if provenance manifest supplied."""
    if manifest is not None:
        record = json.loads(manifest.read_text(encoding="utf-8"))
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != record["sha256"]:
            raise ValueError("Data SHA-256 differs from documented research sample")
    df = pd.read_csv(path)
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError(f"Missing RAND HIE columns: {missing}")
    if df[list(REQUIRED)].isna().any().any():
        raise ValueError("Study variables contain missing values; do not impute silently")
    if not np.isfinite(df[list(REQUIRED)].to_numpy(dtype=float)).all():
        raise ValueError("Study variables must be finite")
    if (df.mdvis < 0).any() or not np.allclose(df.mdvis, np.round(df.mdvis)):
        raise ValueError("mdvis should be a nonnegative integer visit count")
    if not df.idp.isin((0, 1)).all():
        raise ValueError("Insurance deductible plan indicator must be binary")
    if df.lncoins.lt(0).any() or (df.disea < 0).any():
        raise ValueError("Negative cost-share transformation or disease score")
    if (df[["hlthg", "hlthf", "hlthp"]].sum(axis=1) > 1).any():
        raise ValueError("Health category indicators must be mutually exclusive")
    if not df[["hlthg", "hlthf", "hlthp"]].isin((0, 1)).all().all():
        raise ValueError("Health category indicators must be binary")
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # In the archived data, 'disea' is sometimes non-integer despite the
    # documentation describing a chronic-disease count; treat it as a score.
    mean, sd = float(out.disea.mean()), float(out.disea.std(ddof=0))
    if sd <= 0:
        raise ValueError("Chronic disease measure has no variation")
    out["disease_z"] = (out.disea - mean) / sd
    out["costshare_c"] = out.lncoins - float(out.lncoins.mean())
    out["any_visit"] = (out.mdvis > 0).astype(int)
    out["coinsurance_pct_approx"] = np.rint(np.expm1(out.lncoins)).astype(int)
    out["disease_tertile"] = pd.qcut(
        out.disea.rank(method="first"), q=3, labels=["Low", "Middle", "High"]
    )
    return out


def dataset_diagnostics(df: pd.DataFrame) -> dict:
    return {
        "observations": int(len(df)),
        "variables_original": list(REQUIRED),
        "missing_values_total": int(df[list(REQUIRED)].isna().sum().sum()),
        "identical_full_rows_retained": int(df[list(REQUIRED)].duplicated().sum()),
        "visit_mean": float(df.mdvis.mean()),
        "visit_variance": float(df.mdvis.var(ddof=1)),
        "zero_visit_share": float((df.mdvis == 0).mean()),
        "positive_visit_share": float((df.mdvis > 0).mean()),
        "max_visits": int(df.mdvis.max()),
        "unique_costsharing_levels": sorted(df.coinsurance_pct_approx.unique().tolist()),
        "disea_noninteger_records": int((~np.isclose(df.disea, np.round(df.disea))).sum()),
        "participant_identifiers_available": False,
        "experimental_assignment_identifiers_available": False,
    }
