"""Tests of real-data hurdle estimation, prediction and restricted support."""
from pathlib import Path
import numpy as np
from health_econ.data import load_data, prepare_data
from health_econ.hurdle import fit_hurdle, hurdle_predict, estimate_hurdle_and_margins

ROOT = Path(__file__).resolve().parents[1]


def test_hurdle_fits_and_predicts_real_historical_observations():
    df = prepare_data(load_data(
        ROOT / "data/rand_hie_statsmodels_sample.csv",
        ROOT / "data/PROVENANCE.json"
    ))
    binary, positive, info = fit_hurdle(df)
    p, m, total = hurdle_predict(binary, positive, info, df)
    assert len(p) == 20190
    assert (p >= 0).all() and (p <= 1).all()
    assert (m >= 1).all()
    assert np.allclose(total, p*m)
    assert positive.mle_retvals.get("converged")


def test_hurdle_standardization_has_only_supported_nonidp_levels(tmp_path):
    df = prepare_data(load_data(
        ROOT / "data/rand_hie_statsmodels_sample.csv",
        ROOT / "data/PROVENANCE.json"
    ))
    summary, margins = estimate_hurdle_and_margins(df, tmp_path)
    assert summary["sample_n"] == 20190
    assert summary["combined_aic"] > 0
    assert set(margins.coinsurance_pct) == {0, 25, 50, 95}
    assert len(margins) == 12
    assert margins.standardization_group_n.sum() == 4 * (df.idp == 0).sum()
    assert margins.expected_visits_mean.gt(0).all()
    assert (tmp_path / "hurdle_model_diagnostics.json").exists()
