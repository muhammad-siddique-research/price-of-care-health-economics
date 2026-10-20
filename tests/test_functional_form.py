"""Same-sample log-price versus category models: no fabricated plan overlap."""
from pathlib import Path
import numpy as np
from health_econ.data import load_data, prepare_data
from health_econ.functional_form import analyze_functional_form

ROOT = Path(__file__).resolve().parents[1]


def test_model_comparison_and_support_aware_margins(tmp_path):
    df = prepare_data(load_data(
        ROOT / "data/rand_hie_statsmodels_sample.csv",
        ROOT / "data/PROVENANCE.json"
    ))
    models, margins = analyze_functional_form(df, tmp_path, tmp_path)
    assert len(models) == 4
    assert models.converged.all()
    assert models.observations.eq(14941).all()
    assert len(margins) == 12
    assert set(margins.coinsurance_pct) == {0, 25, 50, 95}
    assert np.isfinite(margins.nb2_categorical_price_adjusted_mean).all()
    assert (tmp_path / "figure_8_price_functional_form.png").exists()
