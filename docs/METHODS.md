# Detailed Research Methods, Identification Audit, and Reproducibility
## The Price of Care: Computational Health Economics, Robust Count Econometrics, and Interpretable Machine Learning

**Study type:** Independent, exploratory computational reanalysis of authentic **historical RAND Health Insurance Experiment (HIE)** observations distributed in a reduced `statsmodels` extract.  
**Available sample:** **20,190 rows × 10 original variables**; no missing cells in the documented fields.  
**Primary outcome:** `mdvis`, a nonnegative outpatient physician-visit count.  
**Inference:** **Descriptive associations, model diagnostics, and pattern-held-out prediction—not identified randomized treatment effects or current clinical recommendations.**  
**Implementation:** Python ≥3.10; `statsmodels`, `scikit-learn`, `pandas`, `NumPy`, `SciPy`, and `Matplotlib`.  

[Research overview](../README.md) · [Manuscript](MANUSCRIPT_APA7.md) · [Data provenance](../data/README.md) · [Research agenda](RESEARCH_AGENDA.md) · [Verified outputs](../results/)

---

## 1. Research objectives, questions, and the limits of the estimand

The original RAND HIE randomized insurance arrangements. **This 10-variable analysis extract does not contain the participant/family identifiers, allocation records, timing, assignment strata, or baseline covariate history required to reproduce that experimental design.** The analysis therefore does not estimate an intention-to-treat (ITT) effect, treatment-on-the-treated effect, identified insurance-price elasticity, or effects of present-day health policies.

For archived row $i$, denote physician-visit counts by $Y_i$, logged cost-sharing by $C_i$, recorded disease-burden measure by $D_i$, and observed controls by $X_i$. The primary descriptive target is the conditional utilization mean:

```math
m(c,d,x) = \mathrm{E}[Y_i\mid C_i=c,\ D_i=d,\ X_i=x].
```

Regression coefficients summarize **model-conditional associations** in this reduced archive. Predictions and model-standardized contrasts must not be interpreted as outcomes under interventions that change insurance coverage.

| Research question | Operational comparison | Interpretation permitted |
|---|---|---|
| **RQ1.** Does chronic-disease burden modify the fitted cost-sharing association? | `costshare_c × disease_z` in Poisson HC3, NB2, and any-visit logit | Exploratory, specification-sensitive interaction |
| **RQ2.** Are non-use and use intensity modeled differently? | Occurrence logit + zero-truncated NB2 hurdle likelihood | Descriptive two-part count fit |
| **RQ3.** Does the insurance-price functional form alter fitted utilization? | Logged continuous vs. categorical coinsurance on non-IDP records | Within-subset fit and specification sensitivity |
| **RQ4.** Does nonlinear count-aware ML improve prediction? | Training mean, regularized Poisson GLM, Poisson gradient-boosted trees | Fixed held-out covariate-pattern prediction |
| **RQ5.** Can this extract support an insurance-design policy effect? | Identification audit of omitted assignment/IDs/time variables | **No identified policy effect** |

**Research-status declaration:** These are exploratory post hoc hypotheses and analyses—not preregistered confirmatory tests. The researcher should not represent nominal p-values as multiplicity-controlled discoveries.

## 2. Source data, reproducibility, and variable construction

### 2.1 Historical provenance

The original historical observations are distributed in `statsmodels.datasets.randhie`. The repository's [provenance manifest](../data/PROVENANCE.json) records `statsmodels` source version **0.14.6**, the original ten fields, and SHA-256:

```text
6c58ff02c7a568c267e93c3fffc517daba34239da2fed6287e94098b08471a5d
```

The bundled dataset is **not synthetic**. The automated workflow rematerializes the data using [`scripts/materialize_rand.py`](../scripts/materialize_rand.py), verifies its SHA-256 fingerprint, runs tests, and regenerates artifacts. The ordinary CLI uses the local bundled file; it does not silently download fresh observations. A matching fingerprint establishes file integrity relative to the repository manifest, **not** validity of the original trial design or correctness of all data documentation.

The loader (`src/health_econ/data.py`) requires all ten original fields, rejects missing or non-finite values, checks that visits are nonnegative integers, checks binary plan/health indicators, and rejects negative transformed coinsurance or disease values. Exact duplicate *records* are retained because there is no participant identifier proving they are erroneous duplicate measurements.

### 2.2 Variables and transformations

| Source field | Meaning / analysis role | Handling |
|---|---|---|
| `mdvis` | Outpatient physician-visit count | Outcome, integer ≥0 |
| `lncoins` | `log(1 + coinsurance percentage)` | Main continuous recorded cost-sharing covariate |
| `idp` | Individual deductible plan indicator | Benefit-design control; omitted as constant in `idp=0` subsample |
| `disea` | Archived chronic-disease measure | Continuous observed burden index; **not** repaired into an integer count |
| `lpi` | Logged participation incentive | Control, potentially linked to plan arrangements |
| `fmde` | Plan financial transformation | Control, potentially mechanically linked to cost sharing |
| `physlm` | Archived physical limitation measure | Preserved numerically without invented diagnoses |
| `hlthg`, `hlthf`, `hlthp` | Self-reported good, fair, poor health indicators | Binary controls (omitted reference category implicit) |

The record includes **18,396 noninteger values of `disea`**; treating these as discrete diagnoses would alter the observed data. The implementation uses the recorded continuous measure.

Let $C_i=\mathtt{lncoins}_i$, $D_i=\mathtt{disea}_i$, and define *full-extract* preprocessing constants:

```math
C_i^c = C_i-\overline{C},\qquad
D_i^z = \frac{D_i-\overline{D}}{s_D},\qquad
s_D=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(D_i-\overline D)^2}.
```

The standard deviation uses `ddof=0`. Centering and standardization are calculated **once on the entire archived dataset** by `prepare_data`; they are reused in model-subset sensitivity analyses. This is an in-sample descriptive modeling convention, not leakage-free preprocessing for the ML holdout (which separately uses the source features).

An approximate percentage is reconstructed as `round(exp(lncoins)-1)`. Observed categories are **0%, 25%, 50%, 95%, and 100%**. Disease-burden thirds are constructed with `qcut(disea.rank(method="first"), 3)`: rank-breaking of ties creates approximately equal-sized groups and is a descriptive visualization device, not clinical disease categorization.

### 2.3 Verified descriptive diagnostics

| Diagnostic | Saved result |
|---|---:|
| Archived rows | 20,190 |
| Original variables | 10 |
| Missing values | 0 |
| Mean physician visits | 2.8604 |
| Sample variance of visits | 20.2893 |
| Rows reporting zero visits | 6,308 (**31.243%**) |
| Positive-visit rows | 13,882 |
| Largest observed visit count | 77 |
| Identical ten-field rows retained | 11,065 |

The large variance relative to the mean and substantial zero share motivate count-distribution diagnostics. They do **not** establish the existence of an additional structural-zero process.

Source: [`results/data_diagnostics.json`](../results/data_diagnostics.json) and [`observed_costsharing_strata.csv`](../results/observed_costsharing_strata.csv).

## 3. Core Count Econometrics: Poisson, NB2, and occurrence logit

### 3.1 Conditional mean, interaction, and covariates

The implemented full-sample baseline is:

```math
\log \mu_i
=\beta_0+\beta_1 C_i^c+\beta_2 D_i^z + X_i^\top \gamma,
\qquad \mu_i=\mathrm{E}[Y_i\mid C_i^c,D_i^z,X_i].
```

The main interaction specification is:

```math
\log \mu_i
=\beta_0+\beta_1 C_i^c+\beta_2 D_i^z
+\beta_3(C_i^cD_i^z)+X_i^\top \gamma.
```

Here $X_i$ contains `idp`, `lpi`, `fmde`, `physlm`, `hlthg`, `hlthf`, and `hlthp`. The interaction coefficient $\beta_3$ measures the change in the **fitted log-count association** with a one-standard-deviation shift in recorded disease burden, **per one-unit increase in `log(1 + coinsurance percent)`**. It is **not** a direct percentage-point coinsurance interaction, treatment effect, or elasticity.

The discrete model-conditional count ratio at standardized burden $d$ for a one-unit change in logged cost-sharing is:

```math
\mathrm{CR}(d)=\exp\left(\widehat{\beta}_1+d\widehat{\beta}_3\right).
```

For any actual change from price $p_0$ to $p_1$ (both expressed as percentages), the fitted ratio under the log-linear specification is:

```math
\exp\left[
\lbrace\widehat{\beta}_1+d\widehat{\beta}_3\rbrace
\lbrace\log(1+p_1)-\log(1+p_0)\rbrace
\right].
```

Both are **conditional model contrasts**, not observed causal responses to changing a person's insurance plan. All estimated interactions are exploratory.

### 3.2 Poisson GLM and HC3 uncertainty

The Poisson GLM assumes the working conditional variance $\mathrm{Var}(Y_i\mid X_i)=\mu_i$. The code fits the baseline and interaction models using the log link and row-level **HC3 sandwich covariance**:

```python
smf.glm(formula, data=df, family=sm.families.Poisson()).fit(cov_type="HC3")
```

Poisson quasi-maximum-likelihood interpretation may remain useful under overdispersion if the conditional mean is correctly specified and the appropriate independent observational units are available. **HC3 protects neither against unobserved within-person/family correlation nor against unmeasured confounding or incorrect conditional mean specification.** Since IDs are absent, reported HC3 intervals assume row-level independence that cannot be verified.

For $g(d)=\beta_1+d\beta_3$, the implemented delta-method uncertainty is:

```math
\widehat{\mathrm{Var}}[\widehat g(d)]
=\widehat V_{11}+d^2\widehat V_{33}+2d\widehat V_{13},
```

where $V$ is the fitted HC3 covariance matrix for the two coefficients. The displayed approximate 95% conditional count-ratio interval is:

```math
\left[
\exp\lbrace\widehat g(d)-1.96\thinspace \mathrm{SE}(\widehat g(d))\rbrace,
\exp\lbrace\widehat g(d)+1.96\thinspace \mathrm{SE}(\widehat g(d))\rbrace
\right].
```

The archived burden percentiles **10, 25, 50, 75, 90** are evaluated in [`heterogeneous_count_ratios.csv`](../results/heterogeneous_count_ratios.csv) and Figure 3. These intervals are **nominal**, not participant-clustered or family-clustered confidence intervals.

### 3.3 NB2 maximum likelihood and count variance

To diagnose sensitivity to overdispersion, the implementation fits a **Negative Binomial NB2** regression by BFGS maximum likelihood with the same covariates and interaction:

```math
\mathrm{Var}(Y_i\mid X_i)=\mu_i+\alpha\mu_i^2,\qquad \alpha>0.
```

The NB2 conditional zero probability is:

```math
\Pr(Y_i=0\mid X_i)=(1+\alpha\mu_i)^{-1/\alpha},
```

versus $\exp(-\mu_i)$ for Poisson. For each model the reported zero fraction is the **average model-implied zero probability**, not simply the probability at the mean covariate pattern. Pearson dispersion is computed from $\sum_i (Y_i-\mu_i)^2/V_i\thinspace /\thinspace (N-k)$ using the model-specific variance.

The **NB2 MLE standard errors use the model's default independent-observation covariance**, not HC3. Therefore comparisons between Poisson HC3 and NB2 nominal p-values reflect changes in distributional assumptions **and variance estimation conventions**; they should not be interpreted as a formal test of whether the interaction truly exists.

### 3.4 Separate any-visit logit

The project additionally fits a binomial-logit GLM (with HC3 covariance) for $A_i=\mathbf{1}\lbraceY_i>0\rbrace$ using the same interaction terms:

```math
\Pr(A_i=1\mid X_i)=\mathrm{logit}^{-1}
(\delta_0+\delta_1 C_i^c+\delta_2 D_i^z+\delta_3 C_i^cD_i^z+X_i^\top\eta).
```

This standalone logit is **not, by itself, a count hurdle model**. Its coefficients concern log-odds of any use, and are not directly comparable in scale with Poisson/NB2 log-count coefficients. A complete hurdle likelihood is implemented separately in `hurdle.py` (`Section 4` below).

### 3.5 Verified primary model results

| Full-extract measure | Poisson with HC3 | NB2 MLE |
|---|---:|---:|
| Interaction coefficient | 0.014064 | 0.007885 |
| Interaction standard error | 0.004806 | 0.004713 |
| Exploratory nominal p | .00343 | .09434 |
| AIC | 124,798.92 | 86,788.52 |
| Pearson dispersion | 6.289 | 1.421 |
| Predicted zero share | 8.417% | 31.969% |
| Actual zero share | 31.243% | 31.243% |

**Interpretive finding:** The apparently precise Poisson moderation estimate weakens under NB2. This is evidence of **specification-dependent inference**, not a conclusive causal or clinically actionable interaction. The NB2 model fits the in-sample zero share more closely but cannot resolve unidentified clustering or plan-selection mechanisms.

Sources: [coefficients](../results/econometric_coefficients.csv) · [model diagnostics](../results/econometric_model_diagnostics.json).

## 4. Full two-part hurdle count likelihood

### 4.1 Why a hurdle specification?

The project distinguishes the *extensive margin* (any physician visits) from the *intensive positive-count distribution* (how many visits appear among rows with at least one visit). This distinction is **statistical**, not a demonstrated sequential behavioral decision by patients.

Let $\pi_i=\Pr(Y_i>0\mid X_i)$. The code estimates a binomial occurrence logit using the complete sample and a **zero-truncated Negative Binomial (NB2, p=2)** likelihood using $Y_i>0$ rows. The latter is fit through `statsmodels.discrete.truncated_model.TruncatedLFNegativeBinomialP` with `truncation=0`.

For positive integer $y$, let $f_{\mathrm{NB2}}(y;\mu_i,\alpha)$ denote an untruncated NB2 probability, and let $p_{0i}=(1+\alpha\mu_i)^{-1/\alpha}$. The hurdle probability mass is:

```math
\Pr(Y_i=y\mid X_i)=
\begin{cases}
1-\pi_i, & y=0,\\[3pt]
\pi_i\thinspace \dfrac{f_{\mathrm{NB2}}(y;\mu_i,\alpha)}{1-p_{0i}},
 & y=1,2,\ldots
\end{cases}
```

with $\pi_i=\mathrm{logit}^{-1}(X_i^\top\delta)$ and $\mu_i=\exp(X_i^\top\theta)$. The two components have distinct parameter vectors and are estimated on the applicable complete and positive-count subsets:

```math
\ell_{\mathrm{hurdle}}
=\sum_i\left[
1(Y_i=0)\log(1-\pi_i)
+1(Y_i>0)\left\lbrace\log\pi_i+\log f_{\mathrm{NB2}}(Y_i;\mu_i,\alpha)-\log(1-p_{0i})\right\rbrace
\right].
```

The software's combined log-likelihood is the sum of the fitted **occurrence** and **positive-count truncated** log-likelihoods. Its reported information criterion counts parameters in both components:

```math
\mathrm{AIC}_{\mathrm{hurdle}}
=-2\left(\ell_{\mathrm{logit}}+\ell_{\mathrm{truncNB2}}\right)
+2(k_{\mathrm{logit}}+k_{\mathrm{truncated\ NB2}}).
```

The conditional positive-use expectation and total hurdle expectation are:

```math
\mathrm{E}[Y_i\mid Y_i>0,X_i]
=\frac{\mu_i}{1-(1+\alpha\mu_i)^{-1/\alpha}},
\qquad
\mathrm{E}[Y_i\mid X_i]
=\pi_i\thinspace \mathrm{E}[Y_i\mid Y_i>0,X_i].
```

The implementation obtains the truncated conditional positive mean through the statsmodels prediction interface and multiplies it by the occurrence probability. **In-sample reproduction of the observed zero fraction follows largely from the fitted occurrence intercept; it is not out-of-sample validation or proof of a structural two-decision process.**

### 4.2 Fit and standardized descriptive margins

| Full-extract count model | AIC | Average implied zero share |
|---|---:|---:|
| Poisson interaction | 124,798.92 | 8.417% |
| NB2 interaction | 86,788.52 | 31.969% |
| **Occurrence logit + zero-truncated NB2** | **86,246.96** | **31.243%** |

The fitted positive-count dispersion parameter is **1.7725** and both hurdle components report convergence. The model-standardized mean of **2.8632** is close to the observed mean of **2.8604**.

**Margins are not randomized contrasts.** The code takes the non-IDP rows within each *global* disease-burden third, substitutes each observed non-IDP price argument (0, 25, 50, or 95), holds other covariates as recorded, and averages fitted occurrence, conditional positive count and combined count. Even though these levels appear in the non-IDP subset, the substitution may create **unsupported joint combinations** of price with `lpi`, `fmde` or other plan fields. Treat Figure 7 as a visual diagnostic of fitted specifications, **not** a policy simulation or causal standardization.

Source: [hurdle diagnostics](../results/hurdle_model_diagnostics.json) · [standardized margins](../results/hurdle_standardized_margins.csv) · [Figure 7](../figures/figure_7_hurdle_decomposition.png).

## 5. Benefit-design support and insurance-price functional form

The 100% coinsurance category in this extract is structurally tied to the **individual deductible plan** (`idp=1`), whereas the **14,941 `idp=0` rows** include observed rounded coinsurance levels **0, 25, 50, and 95**. This motivates a restricted **between-model same-sample comparison** that removes the constant `idp` covariate and checks how conclusions depend on price functional form.

Two conditional-mean models with identical non-IDP records and remaining controls are compared:

**Continuous logged cost sharing:**

```math
\log\mu_i=a+b_1C_i^c+b_2D_i^z
+b_3 C_i^cD_i^z+W_i^\top\lambda.
```

**Categorical insurance-price levels (reference: 0%):**

```math
\log\mu_i=a+b_DD_i^z+
\sum_{p\in\lbrace25,50,95\rbrace}
\left[b_p\thinspace 1(P_i=p)+h_p\thinspace 1(P_i=p)D_i^z\right]
+W_i^\top\lambda.
```

Here $W_i$ is the original control list **without `idp`**. Both specifications are estimated using Poisson HC3 and NB2 MLE. The categorical interactions are not interpreted as effects of randomized insurance transitions.

For $k$ fitted coefficients, full-sample model likelihood $\ell$ and sample size $n$, the code computes:

```math
\mathrm{AIC}=-2\ell+2k,\qquad
\mathrm{BIC}=-2\ell+k\log n.
```

| Same **14,941** non-IDP rows | AIC | BIC |
|---|---:|---:|
| Poisson, continuous log price | 93,316.84 | 93,392.96 |
| Poisson, categorical price | 93,277.04 | 93,383.61 |
| NB2, continuous log price | 65,508.93 | **65,592.66** |
| NB2, categorical price | **65,501.65** | 65,615.83 |

For NB2, **AIC favors the categorical model while BIC favors the more parsimonious continuous model**. These are relative **in-sample likelihood penalties**, sensitive to the likelihood and independence assumptions. They do not establish model truth, causal identification or out-of-sample improvement.

A further diagnostic calculates model-standardized expected visits for each global disease-burden third and the four observed non-IDP price arguments, averaging predictions over recorded subgroup covariates. Again, **marginal support for the price category does not imply joint covariate positivity or feasible individual-level insurance switches**.

Source: [price-form comparisons](../results/price_functional_form_comparisons.csv) · [support sensitivity margins](../results/price_support_sensitivity_margins.csv) · [Figure 8](../figures/figure_8_price_functional_form.png).

## 6. Sensitivity analysis and inferential reliability

The implementation refits the primary interaction specification across three analysis samples:

1. **Full extract:** $n=20{,}190$.
2. **Exclude rows above the empirical 99th percentile of utilization:** $n=20{,}007$; observations at or below the threshold are retained. This is not verified exclusion of a fixed percentage of unique people.
3. **Non-IDP subset:** $n=14{,}941$, dropping the now-constant `idp` regressor.

| Sample | Estimator | N | Interaction β | Exploratory p |
|---|---|---:|---:|---:|
| All rows | Poisson HC3 | 20,190 | 0.014064 | .00343 |
| All rows | NB2 MLE | 20,190 | 0.007885 | .09434 |
| Trim upper utilization | Poisson HC3 | 20,007 | 0.013130 | .00075 |
| Trim upper utilization | NB2 MLE | 20,007 | 0.008425 | .06381 |
| Non-IDP only | Poisson HC3 | 14,941 | 0.021356 | .00012 |
| Non-IDP only | NB2 MLE | 14,941 | 0.016416 | .00216 |

Sample changes, overdispersion and model functional form alter numerical conclusions. The stronger non-IDP interaction may reflect plan composition and selection rather than a robust heterogeneity mechanism.

**Unresolved inferential limitations:** No subject/household ID for clustered covariance; no experiment allocation for randomized design reconstruction; no longitudinal time or baseline disease measurement; no multiplicity correction for exploratory testing. A small nominal p-value under these conditions is not a definitive hypothesis test. Excluding high-visit observations also changes the target population represented in the sample.

Saved file: [interaction sensitivity checks](../results/interaction_sensitivity_checks.csv).

## 7. Predictive Machine Learning, grouping, and validation

### 7.1 Prediction target and split design

The predictive task is `mdvis` from **nine original non-outcome variables**: `lncoins`, `idp`, `lpi`, `fmde`, `physlm`, `disea`, `hlthg`, `hlthf` and `hlthp`. Models are trained to predict counts; **they do not estimate causal effects**.

The code hashes each complete predictor vector using `pandas.util.hash_pandas_object(..., index=False)` and applies a **single `GroupShuffleSplit`**, with `test_size=0.25` and random seed **2026**, grouping identical covariate fingerprints. This prevents *exact* predictor patterns from crossing train/test sets. It is **not a person-level, family-level, temporal, or external holdout**: one patient could contribute different covariate patterns to both partitions; hash fingerprints do not reveal patient identity.

| Split diagnostic | Verified count |
|---|---:|
| Training observations | **14,881** |
| Test observations | **5,309** |
| Unique full-dataset predictor patterns | **2,760** |
| Split count | **1** |
| Model random seed | **2026** |

`GroupShuffleSplit` samples groups rather than enforcing an exact row-level 25% proportion; the realized test share is approximately 26.3%. Outcomes were not used to construct the grouping fingerprint, but no verified independent-patient generalization is possible.

### 7.2 Implemented estimators

- **Training-mean baseline:** `DummyRegressor(strategy="mean")`.
- **Regularized Poisson:** `StandardScaler` followed by `PoissonRegressor(alpha=0.05, max_iter=1200)`. The scaler is fitted on training data inside the pipeline.
- **Nonlinear count model:** `HistGradientBoostingRegressor(loss="poisson", learning_rate=0.06, max_iter=180, max_leaf_nodes=15, min_samples_leaf=45, l2_regularization=4.0, early_stopping=False, random_state=2026)`.

No nested cross-validation, random hyperparameter search, external validation cohort, or patient-grouped cross-validation is implemented. The settings should be reported as fixed specifications rather than a globally optimized model.

### 7.3 Evaluation metrics

Held-out primary loss is the mean Poisson deviance for positive predictions:

```math
D_{\mathrm{P}}(y,\widehat\mu)
=\frac{2}{n_{\mathrm{test}}}\sum_{i\in\mathrm{test}}
\left[
y_i\log\left(\frac{y_i}{\widehat\mu_i}\right)-y_i+\widehat\mu_i
\right],
```

with $0\log(0/\widehat\mu_i)=0$ by convention. Code clips predicted means below $10^{-8}$ for numerical stability. Supplementary metrics:

```math
\mathrm{MAE}=\frac{1}{n}\sum_i|y_i-\widehat\mu_i|,
\qquad
\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_i(y_i-\widehat\mu_i)^2}.
```

### 7.4 Verified fixed-holdout results

| Model | Mean Poisson deviance ↓ | MAE ↓ | RMSE ↓ |
|---|---:|---:|---:|
| Training mean | 5.0403 | 2.8581 | 4.8375 |
| Regularized Poisson | 4.5892 | 2.6925 | 4.6860 |
| **Boosted Poisson trees** | **4.4354** | **2.6219** | **4.6377** |

On this single held-out split, boosted trees reduced Poisson deviance by approximately **3.35%** versus regularized Poisson and **12.0%** versus the training-mean baseline. No uncertainty interval across repeated splits was estimated, and these gains do not establish modern clinical risk prediction, transportability, or causal feature influence.

### 7.5 Predictive reliance and calibration

After fitting the boosted model, the code computes permutation importance on the held-out subset using **four feature shuffles each**, seed 2026, and negative mean Poisson deviance as the scoring convention. The reported importance is the resulting **increase in deviance** under disruption of a feature. Correlated insurance variables may exchange or suppress apparent importance; this is **not SHAP**, causal attribution, or a stable ranking of policy levers.

Predictions are sorted and divided into **five equally sized rank-based prediction groups** using a `qcut` of predicted-value ranks. For each bin, mean observed and predicted visit counts are compared. This is one-split group calibration **within the archived data**, not validation across hospitals, years or individuals.

Sources: [holdout metrics](../results/ml_holdout_metrics.csv) · [permutation reliance](../results/ml_permutation_importance.csv) · [calibration](../results/ml_calibration.csv) · [split audit](../results/predictive_split_audit.json).

## 8. Figures and link to implemented outputs

The code produces **eight** research figures, each available as PNG. Figures are model diagnostics or descriptive displays, **not causal response surfaces**.

| Figure | Computational source | Primary content |
|---|---|---|
| [1. Visits distribution](../figures/figure_1_visits_distribution.png) | `plots.py` | Count distribution, 16+ values pooled |
| [2. Observed grouped means](../figures/figure_2_costshare_disease_heterogeneity.png) | `plots.py` | Price categories × global disease thirds |
| [3. Poisson conditional ratios](../figures/figure_3_interaction_margins.png) | `plots.py` | HC3 delta-method interval at burden percentiles |
| [4. Predictive benchmark](../figures/figure_4_ml_comparison.png) | `plots.py` | One-split test Poisson deviance |
| [5. Permutation reliance](../figures/figure_5_permutation_importance.png) | `plots.py` | Test-set feature disruption |
| [6. Prediction calibration](../figures/figure_6_prediction_calibration.png) | `plots.py` | Predicted vs observed quintile means |
| [7. Hurdle margins](../figures/figure_7_hurdle_decomposition.png) | `hurdle.py` | Occurrence, positive intensity and unconditional fitted mean |
| [8. Price functional form](../figures/figure_8_price_functional_form.png) | `functional_form.py` | Categorical vs logged price NB2 on non-IDP sample |

## 9. Research reproducibility protocol

### 9.1 Local steps

From the repository root (Python ≥3.10):

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
python -m health_econ.cli
```

For exact recreation of the repository's archived-data environment, install `statsmodels==0.14.6` and run `python scripts/materialize_rand.py` prior to the CLI. The [`pyproject.toml`](../pyproject.toml) permits a broader compatible statsmodels range, whereas [the GitHub Actions workflow](../.github/workflows/reproduce.yml) explicitly pins **0.14.6** when reproducing the historical sample. Do not treat a successful import under another dependency version as a guarantee of byte-for-byte identical numerical outputs.

### 9.2 Automated quality checks

The GitHub Actions workflow (`Python 3.10` and `3.12`):

1. Installs package and development dependencies and pins `statsmodels==0.14.6`.
2. Rematerializes the historical dataset, checks the SHA-256 manifest, and runs `ruff`.
3. Runs `pytest`.
4. Executes `python -m health_econ.cli` to regenerate statistical results and all eight figures.
5. Publishes workflow artifacts for review and, under its configured main-branch conditions, commits refreshed results/figures if changed.

**Reproduction claim:** The current repository contains runnable code and machine-generated outputs, not a claim that all possible platforms will produce byte-identical boosted-tree estimates. Differences across numerical library and platform versions should be checked against the committed metrics and the workflow artifacts. The current documentation revision does not rerun or modify any estimation code.

### 9.3 Output-to-claim traceability

| Claim or diagnostic | Authoritative committed artifact |
|---|---|
| Authentic sample, count distribution | [`data_diagnostics.json`](../results/data_diagnostics.json) |
| Model fit, dispersion and zero share | [`econometric_model_diagnostics.json`](../results/econometric_model_diagnostics.json) |
| Parameters, nominal errors and p-values | [`econometric_coefficients.csv`](../results/econometric_coefficients.csv) |
| HC3 count-ratio intervals | [`heterogeneous_count_ratios.csv`](../results/heterogeneous_count_ratios.csv) |
| Count sensitivity across samples | [`interaction_sensitivity_checks.csv`](../results/interaction_sensitivity_checks.csv) |
| Two-part log-likelihood/AIC | [`hurdle_model_diagnostics.json`](../results/hurdle_model_diagnostics.json) |
| Hurdle-standardized descriptive means | [`hurdle_standardized_margins.csv`](../results/hurdle_standardized_margins.csv) |
| Price functional form comparisons | [`price_functional_form_comparisons.csv`](../results/price_functional_form_comparisons.csv) |
| Price-supported subgroup margins | [`price_support_sensitivity_margins.csv`](../results/price_support_sensitivity_margins.csv) |
| Machine Learning comparison | [`ml_holdout_metrics.csv`](../results/ml_holdout_metrics.csv) |
| Prediction split limitations | [`predictive_split_audit.json`](../results/predictive_split_audit.json) |
| Permutation importance and calibration | [`ml_permutation_importance.csv`](../results/ml_permutation_importance.csv) and [`ml_calibration.csv`](../results/ml_calibration.csv) |

## 10. Threats to validity and responsible healthcare-policy interpretation

**Internal validity:** This reduced extract lacks original random assignment, family/participant identifiers, baseline morbidity timing, and repeated-measure information. The observed cost-sharing variable may be mechanically related to other plan characteristics. Neither regression adjustment nor model-standardized margins identifies an insurance intervention effect.

**Statistical inference:** HC3 adjusts certain heteroskedasticity-related row-level variance calculations but cannot restore appropriate clustering or correct unknown dependence. NB2/hurdle maximum-likelihood standard errors assume the specified row likelihood structure. No family-clustered uncertainty, randomization inference, multiplicity control, or complete trial-based identification is available.

**Predictive validity:** Identical predictor-pattern grouping removes one narrow type of train/test overlap but does not demonstrate independent-patient prediction, transportability, subgroup fairness, or prospective validity. Performance comparisons come from one realized split and fixed model settings.

**External validity:** The source data are historical, incomplete, and distinct from modern Medicare claims, current insurance markets, and contemporary care delivery. A high quality of code reproduction does not imply that the fitted price-response patterns generalize to present patients.

**Ethics and privacy:** The repository uses a public historical extract, no identified current patient records, and no private claims datasets. Independent results should not be presented as RAND Corporation's conclusions or clinical guidance.

**Evidence-calibrated policy conclusion:** Before using cost-sharing associations to design chronic-disease benefits, future work should distinguish access to any care from intensity conditional on care; audit plan-level and covariate support; consider severity and the risk of forgone effective care; establish a credible treatment variation source; and use appropriate clustering and multi-site or multi-period validation. **This archive alone cannot determine the optimal coinsurance rate or the effect of reducing cost sharing for patients with chronic illness.**

## 11. Next-stage original study: design requirements, not completed work

A follow-up causal research project would require documented access to the **complete** RAND HIE experimental files (or a separately identified modern natural experiment), including participant and household IDs, assignment unit, randomized benefit allocation, trial dates, baseline burden, attrition and health outcomes.

For randomized insurance assignment $Z_g$ at the family/assignment-unit level, a defensible *future* estimand could be the baseline-burden-specific intention-to-treat effect:

```math
\tau(d)
=\mathrm{E}[Y_{ig}(1)-Y_{ig}(0)\mid D^{\mathrm{baseline}}_{ig}=d],
```

where $Y(1)$ and $Y(0)$ are potential outcomes under the assigned plan categories. For multi-arm randomization, prespecify the relevant pair of assignments. Estimation would use the original allocation structure, inference clustered at the randomization unit, attrition checks, and prespecified heterogeneous-effect hypotheses. A price-elasticity estimand would require additional contract-price and behavioral assumptions and cannot be substituted for ITT.

**None of these causal analyses has been implemented or produced empirical results in this repository.** The scientific rationale and surrounding literature are described in the [research agenda](RESEARCH_AGENDA.md) and [targeted evidence map](LITERATURE_REVIEW_2026.md).

## References and methodological foundations

- Aron-Dine, A., Einav, L., & Finkelstein, A. (2013). The RAND health insurance experiment, three decades later. *Journal of Economic Perspectives, 27*(1), 197–222. https://doi.org/10.1257/jep.27.1.197
- Cameron, A. C., & Trivedi, P. K. (2013). *Regression analysis of count data* (2nd ed.). Cambridge University Press. https://doi.org/10.1017/CBO9781139013567
- Deb, P., & Norton, E. C. (2018). Modeling health care expenditures and use. *Annual Review of Public Health, 39*, 489–505. https://doi.org/10.1146/annurev-publhealth-040617-013517
- Manning, W. G., Newhouse, J. P., Duan, N., Keeler, E. B., Leibowitz, A., & Marquis, M. S. (1987). Health insurance and the demand for medical care: Evidence from a randomized experiment. *American Economic Review, 77*(3), 251–277. https://pubmed.ncbi.nlm.nih.gov/10284091/
- Statsmodels Developers. (n.d.). *RAND Health Insurance Experiment Data* [Historical sample and documentation]. https://www.statsmodels.org/stable/datasets/generated/randhie.html

**Latest supporting references and related research design comparisons:** [Literature review, updated 2026](LITERATURE_REVIEW_2026.md). This is a targeted evidence map, **not** a completed systematic novelty review.
