from pathlib import Path
import pytest
from health_econ.data import load_data, prepare_data, dataset_diagnostics

ROOT = Path(__file__).resolve().parents[1]


def test_real_data_provenance_and_size():
    d = load_data(
        ROOT / "data/rand_hie_statsmodels_sample.csv",
        ROOT / "data/PROVENANCE.json"
    )
    assert d.shape == (20190, 10)
    assert d.mdvis.max() >= 20
    assert d.duplicated().sum() > 0  # Preserve authentic repeated records


def test_detect_dataset_tampering(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("fake\n1\n")
    with pytest.raises(ValueError, match="SHA-256"):
        load_data(f, ROOT / "data/PROVENANCE.json")


def test_disease_measure_not_fabricated():
    d = load_data(ROOT / "data/rand_hie_statsmodels_sample.csv")
    p = prepare_data(d)
    assert (p["disea"] == d.disea).all()
    assert dataset_diagnostics(p)["disea_noninteger_records"] > 0
    assert not (p.disea == p.disea.round()).all()


def test_reject_invalid_counts(tmp_path):
    d = load_data(ROOT / "data/rand_hie_statsmodels_sample.csv")
    d.loc[0, "mdvis"] = -1
    x = tmp_path / "bad.csv"
    d.to_csv(x, index=False)
    with pytest.raises(ValueError, match="nonnegative"):
        load_data(x)
