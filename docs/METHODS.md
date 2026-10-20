# Detailed Methods, Identification Audit, and Computational Reproducibility

## Scope and research estimand

This is an **observational reanalysis of real archived RAND Health Insurance Experiment (HIE) sample records**. Despite the historical experiment's randomized assignment, this 10-variable extract excludes participant IDs and treatment allocation. Consequently, it does **not** estimate a randomized intention-to-treat effect or identified price elasticity.

The estimand is a **conditional association**: how physician-visit counts vary with recorded transformed coinsurance and whether that association differs along a chronic-disease burden measure.

### Primary variables

- Outcome: `mdvis`, nonnegative outpatient physician-visit count.
- Cost-sharing: `lncoins = log(1 + coinsurance percent)`, corresponding to approximate 0%, 25%, 50%, 95%, and 100% plan levels.
- Disease burden: `disea`; despite the source description as a disease count, fractional values occur. Treat as a continuous recorded burden measure, standardized to unit population standard deviation.
- Controls: `idp`, `lpi`, `fmde`, `physlm`, `hlthg`, `hlthf`, `hlthp`. Insurance-related terms may be mechanically related; fitted associations are not experimentally isolated effects.

## Data integrity and provenance

1. Retrieve all **20,190 real archived observations** from the historical public-domain statsmodels package, not synthetic simulations.
2. Reject altered bytes using the SHA-256 fingerprint in `data/PROVENANCE.json`.
3. Validate field completeness, finite values, nonnegative integer physician visits, binary plan indicator and mutually exclusive health category indicators.
4. **Retain identical observations:** missing patient/occasion IDs prevent identifying legitimate repeated records. Deleting duplicate rows would silently alter the research extract.
5. Record mean and variance of visits, fraction with zero visits, disease measure peculiarities, and cost-sharing categories.

## Poisson and NB2 Econometric specification

Define centered coinsurance (C_i^c=C_i-\bar C) and standardized disease burden (D_i^z=(D_i-\bar D)/s_D). The count conditional mean model is

$$
\log E[Y_i\mid C_i^c,D_i^z,\mathbf X_i]
=\beta_0+\beta_1C_i^c+\beta_2 D_i^z+\beta_3 C_i^cD_i^z+\mathbf X_i^\top\gamma.
$$

The conditional count ratio for a unit change in log coinsurance at burden level (d) is

$$
CR(d)=\exp(\widehat\beta_1+d\widehat\beta_3).
$$

For (g(d)=\beta_1+d\beta_3), the delta-method variance is

$$
\widehat{\operatorname{Var}}(\widehat g(d))
=\widehat V_{11}+d^2\widehat V_{33}+2d\widehat V_{13}.
$$

Exponentiating (\widehat g(d) \pm 1.96\sqrt{\widehat{\operatorname{Var}}(\widehat g(d))}) gives the reported 95% HC3 Wald interval. **HC3 does not correct unknown repeated-person clustering.**

### Poisson regression

The Poisson conditional mean can be useful with HC3 sandwich variance even with overdispersion if its conditional mean is correctly specified and observations are independent at the relevant unit. Neither assumption can be fully verified from this dataset.

### Negative binomial (NB2) sensitivity

The model assumes (\operatorname{Var}(Y_i\mid X_i)=\mu_i+\alpha\mu_i^2), where (\alpha>0). Fit the interaction model by ML and report convergence, AIC, Pearson dispersion and observed/predicted zero counts. The NB2 expected zero probability is ((1+\alpha\mu)^{-1/\alpha}); the Poisson probability is (e^{-\mu}).

Compare same-sample fitted models on their respective assumptions; lower AIC indicates relative in-sample fit, **not** causal identification.

### Binary occurrence model

Fit logit for (\Pr(Y_i>0\mid X_i)). This explores the extensive margin but is **not** an estimated full two-part/hurdle likelihood. Do not compare binomial coefficients directly to count-model slopes.

## Expanded count Econometrics: hurdle likelihood and plan overlap

### True two-part count likelihood

A standalone occurrence logit is not a hurdle count model. The expanded code fits (a) binomial logit for whether any visits occur, followed by (b) a **zero-truncated NB2** positive-count model by maximum likelihood. Both parts use the same baseline covariates and their log-likelihoods are added, with parameters counted separately for AIC. In symbols:

$
\pi_i = \Pr(Y_i>0 \mid X_i)=\operatorname{logit}^{-1}(X_i^\top\delta),
$

$
Y_i \mid (Y_i>0,X_i) \sim \operatorname{TruncatedNB2}(\mu_i,\alpha),
\quad \mu_i=\exp(X_i^\top\theta),
$

$
E(Y_i\mid X_i)=\pi_i\,\mu_i/
\left[1-(1+\alpha\mu_i)^{-1/\alpha}\right].
$

This is a distributional decomposition of **any care versus number of visits among users**, not an econometrically identified two-stage insurance decision. When the logit fits its intercept on the complete sample, its in-sample predicted zero fraction can match the observed fraction very closely **by construction**; this is not external calibration (Deb & Norton, 2018).

### Common-support and functional-form diagnosis

The 100% coinsurance category exists only under the structurally different individual deductible plan. A support-respecting regression comparison therefore uses the **14,941 non-IDP observations** at 0%, 25%, 50%, and 95% coinsurance, excluding \`idp\` as a constant regressor. Two models are compared **on exactly the same sample**: a continuous logged cost-sharing term and a categorical coinsurance term (both interacted with recorded burden) using Poisson HC3 and NB2.

NB2 AIC is approximately 65,509 under continuous log-price and 65,502 under categories; BIC is approximately 65,593 versus 65,616. Thus one information criterion favors the flexible model and the other the simpler model. Neither selection criterion establishes a causal elasticity.

For each observed disease-burden third, **model-standardized margins** evaluate the fitted prediction at supported coinsurance levels while averaging over the original empirical covariates. These remain **descriptive associations**, and setting a price regressor while holding mechanically related benefit-design covariates constant may generate counterfactual plan combinations that could never be assigned. These graphs are not policy simulations.

### Threats to inferential validity

The reduced archive omits original family and participant identifiers, assignment blocks, dates, and several clinical/demographic variables. Neither model's default independent-row information matrix, HC3 robust standard errors, nor AIC/BIC can correct missing participant-level clustering, unobserved panel dependence, post-hoc model choice or selective covariate availability. Coefficients and nominal *p* values are exploratory.

Current implementation: [\`hurdle.py\`](../src/health_econ/hurdle.py) and [\`functional_form.py\`](../src/health_econ/functional_form.py). Outputs: \`hurdle_model_diagnostics.json\`, \`hurdle_standardized_margins.csv\`, \`price_functional_form_comparisons.csv\`, \`price_support_sensitivity_margins.csv\`, and **Figures 7–8**.

## Robustness checks

- Full archived sample versus dropping observations above the 99th percentile of visits; report changed N.
- Full sample versus non-individual-deductible plans (re-estimate without constant `idp` predictor).
- HC3 Poisson versus NB2, zero-fraction fit and overdispersion.
- All p-values exploratory; no adjustments for unobserved clustering or multiplicity.

## Predictive Machine Learning

Create a group-preserving split via hashed **identical covariate patterns**. This prevents identical predictor rows entering both sets, but it **does not create a true patient-level holdout**: patients might appear with altered covariates across rows. No patient ID is available.

Benchmarks:

- Dummy predictor using the training mean.
- Standardized regularized Poisson regression.
- Histogram gradient-boosted regression with Poisson loss, maximum leaves, shrinkage and a fixed random seed.

Held-out primary loss, mean Poisson deviance:

$$
D_P(y,\widehat\mu)
=\frac{2}{n}\sum_{i=1}^n
\left[y_i\log\left(\frac{y_i}{\widehat\mu_i}\right)
-y_i+\widehat\mu_i\right],
$$

where (0\log(0/\mu)=0). Supplement with MAE, RMSE and calibration by predicted-count quintile. Perform permutation importance on the held-out sample with four random shuffles per predictor. Importance measures **predictive model reliance**, not SHAP causal effects or policy impacts.

## Graphs, reporting and reproducibility

Recreate six results in PNG and SVG: empirical physician-visit distribution; observed visit averages by cost-sharing/disease tertiles; HC3 conditional count-ratio intervals; ML deviance comparison; held-out permutation feature reliance; and predicted/observed calibration.

Every result is generated by `python -m health_econ.cli` using the SHA-256-validated archived research sample. GitHub Actions must reproduce tests and outputs, not import figures from outside unverified sources.

## Threats to validity

This subset excludes person IDs, experimental randomization/strata, observation dates, age, income and additional trial controls. It cannot identify a causal cost-sharing effect, a contemporaneous Medicare policy impact, or out-of-person predictive generalization. The original RAND experiment had richer data; the limitation concerns **the reduced extract**, not the original study design.

For complete APA 7th citations and findings see [the research manuscript](MANUSCRIPT_APA7.md).


For verified 2024–2026 studies contextualizing cost-sharing and chronic illness and for APA references, consult [the literature evidence map](LITERATURE_REVIEW_2026.md) and the [revised manuscript](MANUSCRIPT_APA7.md).
