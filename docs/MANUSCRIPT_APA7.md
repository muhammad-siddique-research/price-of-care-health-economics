# The Price of Care: Chronic Disease Burden and Cost-Sharing in Physician Utilization

### A Real-Data Econometric and Explainable Machine Learning Investigation of the Historical RAND Health Insurance Experiment

**Research type:** Reproducible computational reanalysis of historical, real observations.  
**Dataset:** 20,190 observations and 10 variables, RAND Health Insurance Experiment research extract, publicly distributed via `statsmodels` (v0.14.6).  
**Study scope:** Exploratory and associational; **not** a re-estimation of the original randomized experimental treatment effect.  
**Reporting format:** APA 7th edition in-text citations and reference list.

## Abstract

**Background.** Patient cost-sharing has important implications for healthcare utilization, but its apparent association with demand may vary by health burden and by statistical specification. **Objective.** To assess whether the relationship between recorded insurance cost-sharing and outpatient physician visits varies with chronic disease burden, and whether nonlinear Machine Learning improves predictive performance over conventional Econometrics. **Methods.** We analyzed 20,190 real historical RAND Health Insurance Experiment observations distributed in a reduced research extract, fitting Poisson Generalized Linear Models with HC3 robust standard errors, Negative Binomial (NB2) models, a binary visit-occurrence model, and estimator/sample sensitivity checks. Regularized Poisson regression and Gradient-Boosted Trees were evaluated against a baseline on a covariate-pattern-group holdout. **Results.** Physician visits averaged 2.86 per observation; 31.2% reported zero visits. The Poisson cost-sharing × disease-burden coefficient was 0.0141 (exploratory *p* = .003), compared with 0.0079 (exploratory *p* = .094) in NB2. The NB2 model represented the observed zeros much more closely. Gradient-Boosted Trees reduced held-out mean Poisson deviance from 5.040 (uninformative baseline) to 4.437 (a 12.0% improvement). **Conclusion.** Evidence of cost-sharing moderation is model-sensitive; nonlinear Machine Learning improves prediction within this historical extract, but does not establish causal price effects or current policy relevance. This project contributes a reproducible cross-model robustness and explanation framework, not a verified novel causal discovery.

**Keywords:** Health Economics; Cost-Sharing; Physician Utilization; Chronic Disease Burden; Poisson Regression; Negative Binomial; Applied Econometrics; Machine Learning; Model Robustness.

## 1. Introduction and literature context

Healthcare insurance reduces direct payment at the point of service, potentially influencing demand through changes in the marginal out-of-pocket price. Demand also reflects health need, liquidity constraints, provider access and institutional rules. The original RAND Health Insurance Experiment used random insurance assignments and documented reductions in utilization under cost sharing (Manning et al., 1987; Newhouse & the Insurance Experiment Group, 1993). Later work demonstrated that a move to high deductibles can reduce service quantities without relying primarily on price shopping (Brot-Goldberg et al., 2017).

However, average utilization associations can obscure variation in health needs. Chronic disease burden plausibly changes clinical necessity, patient responsiveness and the distribution of visits. This raises an applied methodological question: **Does the apparent moderation of cost-sharing by chronic disease burden survive models that adequately accommodate overdispersed healthcare-use counts?**

**Proposed contribution:** This code-driven study triangulates three facets that are too often conflated: coefficient heterogeneity under alternative Econometric distributions, utilization occurrence versus total count outcomes, and predictive information in a flexible Poisson-loss model. The value added is a transparent, executable demonstration of **how a conclusion changes across models and sample restrictions**. No comprehensive systematic review has yet established that this constitutes an unprecedented gap in published literature.

### Exploratory hypotheses

**H1:** Higher recorded cost-sharing is negatively associated with visit counts, conditional on included covariates.  
**H2:** Chronic disease burden moderates the apparent cost-sharing association, but the strength of moderation is model-dependent.  
**H3:** Nonlinear Machine Learning achieves lower held-out Poisson deviance than both the sample-mean baseline and regularized Poisson regression.

The hypotheses are **exploratory post hoc specifications**, not prospectively registered randomized-trial tests.

## 2. Data and data quality

The data are an authentic, archived subset of the historical RAND experiment, distributed as `statsmodels.datasets.randhie` and derived from a sample used by Cameron and Trivedi (2005). The file contains **20,190 real observations** with no missing values across the 10 distributed variables. A SHA-256 manifest in `data/PROVENANCE.json` verifies exact local data bytes. The original experiment was randomized; however, this subset omits **participant identifiers, treatment-assignment records, visit dates, study year, age, income and geography**. Randomized treatment effects **cannot** be reconstructed from this extract.

There are **11,065 identical complete rows** across its 10 variables. These are retained: a repeated value combination is not proof of a duplicate individual or a data error, and the dataset lacks identifiers required for deduplication or within-subject covariance estimation. The chronic disease variable `disea` is sometimes fractional, despite its abbreviated source label referring to disease counts. It is therefore analyzed as a continuous **recorded chronic-disease burden measure**, not a fabricated integer diagnosis count. The `physlm` measure also contains fractional values and is not forcibly binarized.

| Original field | Definition used |
|---|---|
| `mdvis` | Nonnegative number of outpatient physician visits (0–77) |
| `lncoins` | Logarithm of 1 plus recorded coinsurance percentage |
| `disea` | Continuous chronic-disease burden measure; fractional values preserved |
| `idp` | Individual deductible plan indicator |
| `lpi` | Log annual participation incentive |
| `fmde` | Transformed maximum deductible / coinsurance quantity |
| `physlm` | Recorded physical-limitation measure |
| `hlthg`, `hlthf`, `hlthp` | Good, fair and poor health indicators; excellent is reference |

**Descriptive results:** Mean physician visits = **2.860**, variance = **20.289**, and zero-visit fraction = **31.24%**. Actual recorded coinsurance categories reconstructed from `lncoins` are 0%, 25%, 50%, 95% and 100%.

![Historical physician visit count distribution](../figures/figure_1_visits_distribution.png)

*Figure 1.* Distribution of outpatient physician visit counts in the real archived RAND sample. Categories of 16 or more visits are pooled for visualization only, not for estimation.

## 3. Mathematical and computational methodology

### 3.1 Conditional mean model

Let \(Y_i\) represent physician visits, \(C_i=\ln(1+\text{coinsurance}_i)\), and \(D_i\) be the continuous chronic-disease burden measure. Cost-sharing is centered and disease burden standardized. The exploratory log-link specification is:

\[
E(Y_i\mid C_i,D_i,X_i)=\exp\left(\beta_0+\beta_1 C_i^c+\beta_2D_i^z+\beta_3C_i^cD_i^z+X_i^{\top}\gamma\right).
\]

The disease-dependent conditional count ratio associated with a **one-unit change in logged (coinsurance + 1)** is:

\[
CR(d)=\exp\left(\beta_1+\beta_3d\right).
\]

This is **not** a marginal effect of a ten-percentage-point insurance change, nor a causally identified price elasticity. The model contains `idp`, `lpi`, `fmde`, `physlm` and the three health category indicators as additional predictors. Because some insurance variables are mechanically related and potential post-assignment behavior is not identified, interpreting an individual coefficient structurally would be unwarranted.

### 3.2 Distributional adequacy

A Poisson GLM specifies a log conditional mean and its full-likelihood variance assumption is \(\operatorname{Var}(Y_i\mid X_i)=\mu_i\). We estimate HC3 robust covariance for that specification. A Negative Binomial NB2 model instead allows:

\[
\operatorname{Var}(Y_i\mid X_i)=\mu_i+\alpha\mu_i^2,\qquad\alpha>0.
\]

Pearson dispersion, AIC and expected-zero diagnostics assess model adequacy, not causal validity. In Poisson, the zero probability is \(\exp(-\mu_i)\). In NB2, the predicted zero probability is \((1+\alpha\mu_i)^{-1/\alpha}\). A logistic model estimates the probability of **any** physician visit as an auxiliary decomposition, not as a complete hurdle-count likelihood.

### 3.3 Heterogeneity and sample-sensitivity checks

A Poisson-interaction slope is evaluated at disease-burden percentiles using the HC3 delta method. The interaction is re-estimated after (a) excluding records above the 99th percentile of visits and (b) restricting to non-individual-deductible plans. The latter modifies the nuisance formula to remove a constant `idp` regressor. Both operations are exploratory sample-composition tests rather than randomization-based contrasts. Nominal p-values must be interpreted cautiously, especially given multiple specifications and unavailable participant-level clustering.

### 3.4 Predictive ML and model-reliance analysis

The algorithmically distinct benchmark models are a **training-set mean**, regularized **Poisson GLM**, and **Gradient-Boosted Trees with Poisson loss**. Training and test sets are separated using a **grouped covariate fingerprint split**: observations with identical nine-variable input profiles cannot appear in both partitions. The resulting holdout contains 14,881 training and 5,309 test rows. This restriction prevents exact covariate-pattern overlap, **but does not guarantee person-level separation**. It also may create an atypical covariate shift, so performance must not be presented as transportability to the modern population.

Evaluation criteria are held-out mean Poisson deviance (primary), mean absolute error and root mean squared error. Permutation importance estimates change in predictive loss when a single covariate is shuffled on test rows; it is **not causal attribution**. Correlated insurance variables may redistribute importance. The pipeline uses fixed seeds, deterministic algorithm settings, no fitted imputation and a published source checksum.

## 4. Empirical results

### 4.1 Interaction estimates vary materially by estimator

| Model | Cost-sharing × disease-burden beta | Approximate SE | Exploratory p-value |
|---|---:|---:|---:|
| Poisson (HC3) | **0.0141** | 0.0048 | .003 |
| NB2 maximum likelihood | **0.0079** | 0.0047 | .094 |
| Logistic model, any visit | **0.0053** | 0.0092 | .564 |

The positive Poisson moderation is **not unequivocally supported by the models better accommodating overdispersion or the extensive margin**. Even the Poisson nominal p-value does not account for repeated-subject dependence because the IDs are missing.

![Observed visits by cost-sharing and burden tertile](../figures/figure_2_costshare_disease_heterogeneity.png)

*Figure 2.* Actual descriptive mean visits by recorded cost-sharing categories and disease-burden tertiles. Lines join observed group means; they are not counterfactual individual price responses. The 100% category is structurally different because it occurs under the individual deductible plan in this extract.

![Poisson conditional count ratio across disease burden](../figures/figure_3_interaction_margins.png)

*Figure 3.* Model-based conditional count ratios, with HC3 delta-method 95% intervals across chronic disease percentiles. These intervals rely on the assumption of independent observations, which is not verifiable from this reduced dataset.

### 4.2 Negative Binomial captures the zero mass more adequately

| Diagnostic | Poisson interaction | NB2 interaction |
|---|---:|---:|
| AIC | 124,799 | 86,789 |
| Pearson dispersion, model variance | 6.29 | 1.42 |
| Observed zero fraction | 31.24% | 31.24% |
| Model-implied zero fraction | 8.42% | 31.97% |

NB2's improvement is substantial for the given sample and likelihood assumptions. It does not establish a unique true clinical demand process or imply that a zero-inflated mixture is required. The excess zeros and overdispersion should motivate distributional diagnostics before advancing interpretive claims.

### 4.3 Sensitivity and model-selection uncertainty

| Sample restriction | Estimator | N | Interaction beta | Exploratory p |
|---|---|---:|---:|---:|
| Complete extract | Poisson HC3 | 20,190 | 0.0141 | .003 |
| Complete extract | NB2 | 20,190 | 0.0079 | .094 |
| Exclude top ~1% utilization | Poisson HC3 | 20,007 | 0.0131 | .001 |
| Exclude top ~1% utilization | NB2 | 20,007 | 0.0084 | .064 |
| Non-IDP plans | Poisson HC3 | 14,941 | 0.0214 | less than .001 |
| Non-IDP plans | NB2 | 14,941 | 0.0164 | .002 |

*Table 1.* All coefficients are exploratory. The interaction is more pronounced in the non-IDP subset, suggesting plan composition or functional-form sensitivity; this does not isolate a treatment mechanism.

### 4.4 Machine Learning provides modest predictive value

| Model | Held-out mean Poisson deviance ↓ | MAE ↓ | RMSE ↓ |
|---|---:|---:|---:|
| Training mean | 5.040 | 2.858 | 4.837 |
| Regularized Poisson | 4.589 | 2.692 | 4.686 |
| Gradient-Boosted Poisson Trees | **4.437** | **2.622** | **4.641** |

The boosted model reduces Poisson deviance by approximately **12.0% compared with the uninformative baseline** and **3.3% compared with regularized Poisson** in one fixed historical holdout. Gains should not be interpreted as causal discovery or established generalization to a modern insurer population.

![Econometric and ML out-of-pattern predictive model comparison](../figures/figure_4_ml_comparison.png)

*Figure 4.* Identical held-out rows were used for all predictive models. Smaller Poisson deviance indicates better average count-distribution prediction.

![Held-out permutation importance](../figures/figure_5_permutation_importance.png)

*Figure 5.* Model-reliance scores, not causal feature impacts. Chronic-disease burden ranked highest; deductible construction and participation incentives also contributed to prediction. Correlated features may share importance.

![Held-out prediction calibration](../figures/figure_6_prediction_calibration.png)

*Figure 6.* Mean fitted and actual counts by quintile of held-out model predictions. This describes historical predictive calibration, not patient risk prediction in a contemporary health system.

## 5. Discussion and potential literature contribution

Three findings are useful for Health Economics methods. First, **health need may shape observed utilization gradients**, but the study's abbreviated fields cannot separate need from insurance assignment, socioeconomic status or unobserved factors. Second, **a seemingly precise coefficient interaction may weaken when a realistic count variance is allowed**. Here, the conventional Poisson interaction is less compelling under NB2 and an any-visit model. Third, **nonlinear ML can exploit additional predictive structure** while still lacking the design information needed to attribute demand changes to insurance prices.

The **literature-facing value addition** is a transparent, reproducible *robustness triangulation template*: readers can independently examine the same data and understand when model-dependent heterogeneity claims should be withheld. This is a useful methodological contribution for researchers and trainees, but a claim of original empirical or theoretical novelty would require further systematic review and independent validation.

A next-stage causal study should recover the RAND trial's original study IDs, randomization records, baseline chronic disease status and follow-up timing (where access permits). It could then estimate preregistered **intention-to-treat heterogeneity** in physician demand, with clustering at participant/family randomization units and checks of health-need effects. That design is not implemented here.

## 6. Limitations and ethical considerations

The analytic subset lacks participant IDs, treatment assignment, year, age, income and other determinants. Randomized assignment in the original experiment **does not make this extract's regression coefficients causal**. HC3 uncertainty assumes independent row contributions; unknown repeated individuals may make nominal uncertainty too narrow. The chronic burden and physical-limitation fields contain fractional entries, and data cannot be verified as contemporary diagnoses. Insurance variables are mechanically related. Excluding high utilizers changes the study population, while the non-IDP subset omits a structurally different plan type. The predictive split prevents identical covariate patterns from crossing but does not establish patient-level independence. Modern health systems, medical technology and benefit designs differ dramatically from the historical study period. Multiple post hoc hypotheses mean that p-values are exploratory. The dataset is publicly distributed, and no new protected patient records have been obtained.

## 7. Conclusion

This project analyzes **authentic historical Health Economics observations**, documents source integrity, and applies advanced Econometrics and Machine Learning without conflating fitted associations with causal effects. Physician visits are highly overdispersed and zero-heavy. The chronic-disease moderation estimate is *model-sensitive* rather than robustly conclusive, while Poisson-loss boosting modestly improves held-out predictions. **The principal contribution is rigorous analytical judgment, reproducibility, and interpretable method comparison**, not an unsupported new causal claim.

## 8. Reproducibility and research transparency

All original archived observations in this extract, a SHA-256 source manifest, Python code, unit tests, model diagnostics, sensitivity estimates, ML benchmarking tables, and six figures (PNG and SVG) are included with the repository. Run `python -m health_econ.cli` to regenerate the results. The data come from the historical RAND research program as distributed through statsmodels. All new results and potential errors in this independent reanalysis remain the responsibility of the project authors, not RAND.

## References

Brot-Goldberg, Z. C., Chandra, A., Handel, B. R., & Kolstad, J. T. (2017). What does a deductible do? The impact of cost-sharing on health care prices, quantities, and spending dynamics. *The Quarterly Journal of Economics, 132*(3), 1261–1318. https://doi.org/10.1093/qje/qjx013

Cameron, A. C., & Trivedi, P. K. (2005). *Microeconometrics: Methods and applications*. Cambridge University Press. https://cameron.econ.ucdavis.edu/mmabook/mma.html

Cameron, A. C., & Trivedi, P. K. (2013). *Regression analysis of count data* (2nd ed.). Cambridge University Press. https://doi.org/10.1017/CBO9781139013567

Manning, W. G., Newhouse, J. P., Duan, N., Keeler, E. B., Leibowitz, A., & Marquis, M. S. (1987). Health insurance and the demand for medical care: Evidence from a randomized experiment. *The American Economic Review, 77*(3), 251–277. https://pubmed.ncbi.nlm.nih.gov/10284091/

Newhouse, J. P., & the Insurance Experiment Group. (1993). *Free for all? Lessons from the RAND Health Insurance Experiment*. Harvard University Press. https://books.google.com/books?id=SVUJ4W9Lk5IC

Statsmodels Developers. (n.d.). *RAND Health Insurance Experiment Data* [Data set and documentation]. Statsmodels. https://www.statsmodels.org/stable/datasets/generated/randhie.html