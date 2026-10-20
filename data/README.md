# Dataset provenance and statistical interpretation

The included RAND Health Insurance Experiment sample contains **20,190 actual historical observations**, distributed with `statsmodels.datasets.randhie` version 0.14.6. These are **not synthetic**. The downloaded dataset is generated reproducibly by `scripts/materialize_rand.py` from the public-domain statsmodels sample, and every research run validates its SHA-256 against `PROVENANCE.json`.

The sample has ten variables and is **not the full original experiment**. Participant IDs, experimental assignment, treatment dates, and complete randomization metadata are unavailable; **do not claim causal treatment effects** or a confirmed independent-person predictive holdout. Some archived rows are identical. These rows are retained because unique participant identifiers are missing and dropping them would alter real observations.

The `disea` variable has fractional values in this archived extract, notwithstanding a documentation description as a count of chronic diseases. We treat it as a **continuous disease burden measure**, never as an invented integer count. The `physlm` variable is also preserved as distributed.

| Column | Original published description | This study |
|---|---|---|
| `mdvis` | Outpatient physician visits | Nonnegative count response |
| `lncoins` | Log(coinsurance percent + 1) | Cost sharing; exploratory, noncausal |
| `idp` | Individual deductible plan | Plan indicator |
| `lpi` | Log annual participation incentive | Covariate |
| `fmde` | Financial plan cost-sharing transformation | Covariate |
| `physlm` | Physical limitation | Preserve recorded numeric values |
| `disea` | Number of chronic diseases | Continuous burden index in this extract |
| `hlthg` | Good self-reported health | Indicator |
| `hlthf` | Fair self-reported health | Indicator |
| `hlthp` | Poor self-reported health | Indicator |

**Source:** [statsmodels RAND HIE documentation](https://www.statsmodels.org/stable/datasets/generated/randhie.html), with background in [RAND's experiment description](https://www.rand.org/health-care/projects/hie.html).
