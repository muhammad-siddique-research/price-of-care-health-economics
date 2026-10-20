# Research Contribution and Future Studies

## Research question and value added

**Question:** How sensitive is the association between insurance cost-sharing and recorded outpatient physician utilization to chronic disease burden, insurance-plan composition and count-distribution assumptions?

The landmark RAND Health Insurance Experiment has been studied extensively. This project **does not claim to be the first** study of price responsiveness, heterogeneity, or count-model regression in the dataset. Its contribution is a **transparent, reproducible robustness and predictive-methods comparison**: Poisson HC3, NB2, subsample sensitivity, binary occurrence, and modern nonlinear count prediction on the same real historical records.

The initial Poisson interaction suggests stronger moderation than the NB2 model, demonstrating why statistically significant coefficients alone cannot establish a stable empirical narrative.

## Research hypotheses

- **H1:** Recorded cost sharing is negatively associated with visits, conditional on the limited variables available.
- **H2:** Disease burden moderates the association, but apparent interaction strength may vary across Poisson, NB2 and binary occurrence models.
- **H3:** Nonlinear Machine Learning with a count-aware Poisson loss can outperform a generalized-linear predictive baseline on a held-out covariate-pattern split.

Hypotheses were developed for a computational portfolio, **not preregistered**, and cannot establish randomized causal effects or structural price elasticities.

## Next-stage causal research (not implemented)

Obtain approved access to the **complete RAND experimental design files**, including participant/family identifiers, randomization unit, plan assignment, original assignment strata, participation dates, baseline morbidity and attrition. Predefine primary outcomes and estimate intention-to-treat differences by randomized benefit assignment with clustering at the appropriate assignment unit. Test moderation using **baseline** disease burden, not a possibly post-treatment disease measure.

Research extending to modern Medicare or other claims cohorts must secure suitable privacy and data-use approvals and choose a credible experiment, policy discontinuity or panel identification strategy. Difference-in-differences additionally requires careful pretrend/parallel-trend and treatment-timing checks. None of these are estimable from the current reduced extract.

## Literature-review protocol

Search peer-reviewed Health Economics and insurance/Health Services Research literature for RAND HIE, coinsurance, chronic disease, utilization heterogeneity, count models, zero inflation and nonlinear prediction. Extract dataset coverage, unit of observation, causal estimand, clustering, model specification, real-world meaning and validation. Novelty must follow evidence of a distinct estimand, mechanism or empirical contribution—not merely the use of new model names.

## Professional research positioning

The repository demonstrates authentic data handling, Econometric modeling, model-critical interpretation, Machine Learning, documentation, and reproducibility. It does not replace professional experience with patient-identifiable or restricted Medicare claims, and is not a peer-reviewed journal article.

[Full manuscript](MANUSCRIPT_APA7.md) · [Methods and limitations](METHODS.md) · [Data provenance](../data/README.md)
