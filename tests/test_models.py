from pathlib import Path
import numpy as np
import pandas as pd
from health_econ.data import load_data, prepare_data
from health_econ.econometrics import descriptive_table
from health_econ.ml import group_split

ROOT = Path(__file__).resolve().parents[1]


def test_group_split_disjoint_fingerprints():
    d = prepare_data(load_data(ROOT / "data/rand_hie_statsmodels_sample.csv"))
    a, b = group_split(d)
    cols = [
        "lncoins", "idp", "lpi", "fmde", "physlm", "disea",
        "hlthg", "hlthf", "hlthp"
    ]
    ga = set(pd.util.hash_pandas_object(d.iloc[a][cols], index=False))
    gb = set(pd.util.hash_pandas_object(d.iloc[b][cols], index=False))
    assert not ga.intersection(gb)
    assert len(a) + len(b) == len(d)


def test_descriptive_n_sum():
    d = prepare_data(load_data(ROOT / "data/rand_hie_statsmodels_sample.csv"))
    table = descriptive_table(d)
    assert table.n.sum() == len(d)
    assert np.isfinite(table.mean_visits).all()


def test_real_data_permissible_costsharing_levels():
    d = prepare_data(load_data(ROOT / "data/rand_hie_statsmodels_sample.csv"))
    assert set(d.coinsurance_pct_approx.unique()) == {0, 25, 50, 95, 100}
    assert d["physlm"].between(0, 1).all()
