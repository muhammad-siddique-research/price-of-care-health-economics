"""One-command, fully reproducible empirical Health Economics study."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .data import load_data, prepare_data, dataset_diagnostics
from .econometrics import (
    estimate_models, coefficient_table, model_diagnostics, heterogeneity_table, descriptive_table
)
from .ml import evaluate_ml
from .plots import plot_descriptive, plot_heterogeneity, plot_margins, plot_ml
from .robustness import check_sensitivity


def run(data_path: Path, provenance_path: Path, results: Path, figs: Path):
    results.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    df = prepare_data(load_data(data_path, provenance_path))
    audit = dataset_diagnostics(df)
    models = estimate_models(df)
    coefs = coefficient_table(models)
    diagnostics = model_diagnostics(df, models)
    margins = heterogeneity_table(df, models["poisson_interaction"])
    strata = descriptive_table(df)
    comparison, importance, calibration, split = evaluate_ml(df)
    sensitivity = check_sensitivity(df)
    (results / "data_diagnostics.json").write_text(json.dumps(audit, indent=2))
    (results / "econometric_model_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2)
    )
    (results / "predictive_split_audit.json").write_text(json.dumps(split, indent=2))
    coefs.to_csv(results / "econometric_coefficients.csv", index=False)
    sensitivity.to_csv(results / "interaction_sensitivity_checks.csv", index=False)
    margins.to_csv(results / "heterogeneous_count_ratios.csv", index=False)
    strata.to_csv(results / "observed_costsharing_strata.csv", index=False)
    comparison.to_csv(results / "ml_holdout_metrics.csv", index=False)
    importance.to_csv(results / "ml_permutation_importance.csv", index=False)
    calibration.to_csv(results / "ml_calibration.csv", index=False)
    plot_descriptive(df, figs)
    plot_heterogeneity(strata, figs)
    plot_margins(margins, figs)
    plot_ml(comparison, importance, calibration, figs)
    return (
        audit, coefs, diagnostics, margins, strata, comparison,
        importance, calibration, split, sensitivity
    )


def main():
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="Analyze actual archived RAND HIE records")
    p.add_argument(
        "--data", type=Path, default=root / "data" / "rand_hie_statsmodels_sample.csv"
    )
    p.add_argument("--manifest", type=Path, default=root / "data" / "PROVENANCE.json")
    p.add_argument("--results", type=Path, default=root / "results")
    p.add_argument("--figures", type=Path, default=root / "figures")
    a = p.parse_args()
    audit, *_ = run(a.data, a.manifest, a.results, a.figures)
    print(f"Processed {audit['observations']:,} REAL historical RAND sample observations")
    print("CRITICAL: No individual identifiers or full experimental assignment data;")
    print("regression results are associations and held-out predictions, NOT causal effects.")


if __name__ == "__main__":
    main()
