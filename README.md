# THE PRICE OF CARE
### Cost-Sharing, Chronic Disease Burden & Outpatient Physician Utilization

**Real-data Health Economics research** · Applied Econometrics · Count Models · Explainable Machine Learning · Reproducible Python

![Data](https://img.shields.io/badge/DATA-REAL%20historical%20RAND%20HIE-12796f) ![N](https://img.shields.io/badge/N-20%2C190-284b70) ![Scope](https://img.shields.io/badge/Scope-Exploratory%20%26%20Noncausal-c6752b) ![License](https://img.shields.io/badge/Code-MIT-blue)

> **Research question:** Does the recorded cost-sharing–physician utilization relationship vary with chronic disease burden—and do these findings hold up under Negative Binomial Econometrics and nonlinear Machine Learning?

[![Real-data research CI](https://github.com/muhammad-siddique-research/price-of-care-health-economics/actions/workflows/reproduce.yml/badge.svg)](https://github.com/muhammad-siddique-research/price-of-care-health-economics/actions/workflows/reproduce.yml)

**Read the complete research:** [APA 7th-style Manuscript](docs/MANUSCRIPT_APA7.md) · [Data Provenance](data/README.md) · [Research Methodology](docs/METHODS.md) · [Research Agenda](docs/RESEARCH_AGENDA.md).

## Why this project matters

The RAND Health Insurance Experiment is a landmark study in Health Economics. This independent project analyzes **real archived research observations**, unlike a synthetic Medicare methods demonstration. The provided file is a **reduced 10-variable historical sample**, not the entire experiment; critically it has **no participant IDs, experimental assignment, or time stamps**. We therefore do not claim randomized treatment effects, independent-patient validation, or current clinical recommendations.

The intellectual contribution here is **model-robustness triangulation**: a seemingly precise cost-sharing × health-burden interaction in a Poisson model becomes weaker under a Negative Binomial model that better captures dispersion. The entire analysis is reproducible and the limits of inference are explicit.

## Verified empirical findings (20,190 real observations)

| Research question / statistic | Calculated result |
|---|---:|
| Mean physician visits per archived observation | **2.86** |
| Observations with zero visits | **31.2%** |
| Visit count variance | **20.29** |
| Poisson-interaction implied zero fraction | 8.4% |
| NB2-interaction implied zero fraction | **32.0%** |
| Cost-sharing × chronic burden, Poisson HC3 | beta = 0.0141; exploratory *p* = .003 |
| Cost-sharing × chronic burden, NB2 | beta = 0.0079; exploratory *p* = .094 |
| Test Poisson deviance, regularized Poisson | 4.589 |
| Test Poisson deviance, Gradient-Boosted Poisson Trees | **4.435** |

**Interpretation:** Poisson may overstate certainty when severe overdispersion and zero counts are inadequately modeled; however, NB2 and HC3 nominal tests also **cannot** correct unknown within-person clustering. The exploratory H2 moderation is **not robustly established**. The ML uplift is modest and predictive—not causal.

## Six real-data, reproducible research figures

### Figure 1 — Why visit counts require appropriate Econometrics

![Real empirical distribution of doctor visits](figures/figure_1_visits_distribution.png)

### Figure 2 — Cost-sharing patterns by chronic disease burden

![Observed mean physician visits by coinsurance and disease-burden tertile](figures/figure_2_costshare_disease_heterogeneity.png)

### Figure 3 — Model-based disease heterogeneity with HC3 intervals

![Count-ratio estimates at chronic-disease percentiles](figures/figure_3_interaction_margins.png)

### Figure 4 — Out-of-pattern predictive accuracy

![Poisson regression versus nonlinear ML on held-out data](figures/figure_4_ml_comparison.png)

### Figure 5 — Explainable ML: predictive reliance, not causality

![Permutation feature importance measured by held-out Poisson deviance](figures/figure_5_permutation_importance.png)

### Figure 6 — Calibration across prediction quintiles

![Observed versus model-predicted visit counts](figures/figure_6_prediction_calibration.png)

All six charts were generated from **real archived observations**; each is available as a PNG and editable, resolution-independent SVG. The dataset is historical, excludes patient identifiers, and cannot establish present-day medical or insurance effects.

## Analytical research workflow

```mermaid
flowchart LR
    A[Real archived RAND sample] --> B[SHA-256 and data validation]
    B --> C[Poisson HC3 and interaction]
    B --> D[NB2 and occurrence logit]
    C --> E[Estimator and sample sensitivity]
    D --> E
    B --> F[Covariate-pattern holdout]
    F --> G[Regularized Poisson and boosted trees]
    G --> H[Deviance, calibration, permutation importance]
    E --> I[APA-style research manuscript]
    H --> I
```

## Scientific competencies demonstrated

**Applied Econometrics:** Poisson GLM, HC3 sandwich standard errors, NB2 dispersion estimation, interaction coefficients, count ratios, likelihood/AIC model comparisons, binomial logit, zero-mass adequacy, and sensitivity to sample composition.

**Machine Learning:** Baseline versus regularized Poisson and gradient-boosted trees, a stringent exact-covariate-pattern holdout, Poisson deviance, MAE/RMSE, test-set permutation feature reliance, and prediction calibration.

**Research Engineering:** Authentic data with a SHA-256 manifest, source attribution, programmatic validation, non-invention of fractional disease-score values, deterministic seeds, reproducible outputs, automated tests, and CI for Python 3.10 and 3.12.

**Research ethics:** Complete publication-style methodology, reproducibility statement, inconclusive and sensitivity-dependent findings, historical-study limitations, and literature references in APA 7th style.

## One-command reproducibility

Python 3.10+ required:

```bash
python -m venv .venv
# On Windows PowerShell: .\.venv\Scripts\Activate.ps1
# On macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
python -m health_econ.cli
```

The bundled CSV, `data/rand_hie_statsmodels_sample.csv`, was copied from the **public-domain historical dataset** provided in `statsmodels.datasets.randhie`; it is **not generated or simulated**. The command verifies its stored checksum, computes Econometric and ML results, and regenerates the six figures. Small platform-specific floating-point differences in tree learning may occur; `results/ml_holdout_metrics.csv` is the source of exact values for each verified run. No secret keys, API access, network downloads, or restricted patient information are required.

## Repository architecture

```text
data/                         Real historical RAND subset and source SHA-256
src/health_econ/data.py         Input provenance, validation, constructed indicators
src/health_econ/econometrics.py  Poisson, NB2, logistic, interactions
src/health_econ/robustness.py    Distributional and sample checks
src/health_econ/ml.py            Pattern-group split and interpretable ML
src/health_econ/plots.py         Publication-quality PNG and SVG figures
src/health_econ/cli.py           Reproducible research runner
docs/MANUSCRIPT_APA7.md         Article with empirical results and APA references
docs/METHODS.md                 Equations, modeling and identification limits
docs/RESEARCH_AGENDA.md         Hypotheses, value addition, further research
results/                        Actual model outputs, sensitivity, audit, ML scores
figures/                        Six empirical figures in PNG and SVG
tests/                          Dataset provenance and reproducibility checks
.github/workflows/ci.yml        Full rerun on pushes and pull requests
```

## Limitations that safeguard credibility

1. **Historical, real but incomplete sample.** No participant ID, experimental assignment, observation date, age, or income is available, so this is **not an identified randomized causal study**.
2. **Participant clustering unknown.** HC3 standard errors assume independent rows and may be optimistic. The covariate-fingerprint split is not a verified person-level holdout.
3. **Variable documentation imperfect.** Chronic disease and physical limitation fields contain fractional values; we preserve original observations rather than fabricating clinical events.
4. **Nominal significance is exploratory.** Interactions differ by model, plan structure and selection of high utilizers.
5. **No new-literature-priority claim.** This is a reproducible robustness and research training contribution; novelty requires a systematic literature review and additional identifiers.

## Research conclusion

The result worth highlighting is methodological rather than sensational: *overdispersion changes how strongly we should believe observed cost-sharing interactions*. NB2 fits the real count distribution substantially better, while nonlinear Poisson boosting improves the predictive baseline without resolving causal identification. This distinction is central to credible Health Economics and Applied Econometrics research.

## Selected references (APA 7th edition)

Brot-Goldberg, Z. C., Chandra, A., Handel, B. R., & Kolstad, J. T. (2017). What does a deductible do? The impact of cost-sharing on health care prices, quantities, and spending dynamics. *The Quarterly Journal of Economics, 132*(3), 1261–1318. https://doi.org/10.1093/qje/qjx013

Cameron, A. C., & Trivedi, P. K. (2005). *Microeconometrics: Methods and applications*. Cambridge University Press. https://cameron.econ.ucdavis.edu/mmabook/mma.html

Cameron, A. C., & Trivedi, P. K. (2013). *Regression analysis of count data* (2nd ed.). Cambridge University Press. https://doi.org/10.1017/CBO9781139013567

Manning, W. G., Newhouse, J. P., Duan, N., Keeler, E. B., Leibowitz, A., & Marquis, M. S. (1987). Health insurance and the demand for medical care: Evidence from a randomized experiment. *The American Economic Review, 77*(3), 251–277. https://pubmed.ncbi.nlm.nih.gov/10284091/

Newhouse, J. P., & the Insurance Experiment Group. (1993). *Free for all? Lessons from the RAND Health Insurance Experiment*. Harvard University Press. https://books.google.com/books?id=SVUJ4W9Lk5IC

Statsmodels Developers. (n.d.). *RAND Health Insurance Experiment Data* [Data set and documentation]. Statsmodels. https://www.statsmodels.org/stable/datasets/generated/randhie.html

## License

Code and original research documentation: MIT. The archived RAND subset is distributed as public-domain data in statsmodels, with full provenance and source credits. These independent project results should not be attributed to RAND Corporation or the original experimental research team.