# Contemporary Evidence Map and Novelty Audit — Updated October 2026

**Scope:** Insurance cost-sharing, outpatient utilization, chronic disease, model adequacy, and applied predictive methods.

**Evidence status:** Targeted review of verified primary publications, **not** a systematic review. This research portfolio does **not** claim priority for hurdle methods, count models, insurance heterogeneity, or RAND HIE analysis.

## What is already established?

The RAND Health Insurance Experiment (HIE) randomized families to different benefit arrangements and is foundational for understanding cost sharing and demand (Manning et al., 1987). Aron-Dine et al. (2013) later revisited the experiment, emphasizing that nonlinear insurance contracts cannot be reduced uncritically to one price elasticity. **These findings rely on richer experimental documentation than the reduced 10-variable sample supplied by statsmodels.**

Modern literature remains substantively relevant but empirically distinct. Gidwani et al. (2024) examined reduced telehealth cost sharing among chronically ill enrollees using difference-in-differences. Chen et al. (2026) examined chronic-disease outpatient coverage among lower-income insured patients in China using modern administrative records. Tran et al. (2026) studied unexpected preventive-care cost-sharing among patients with chronic conditions in the United States. None supplies missing RAND participant identifiers or validates a causal reinterpretation of this project's regressions.

Deb and Norton (2018) provide a key statistical foundation for Poisson, negative binomial, and two-part/hurdle healthcare-use models, highlighting that the mass at zero and skewness warrant attention to extensive versus intensive utilization.

## Comparative research-evidence matrix

| Source | Research setting | Design contribution | Why it cannot establish this repository's causal claims |
|---|---|---|---|
| Manning et al. (1987) | Historical RAND randomized experiment | Original insurance-plan contrasts | The reduced data lack randomized assignment IDs, units and dates |
| Aron-Dine et al. (2013) | RAND reanalysis | Design and elasticity critique | The public 10-variable subset is less complete |
| Deb & Norton (2018) | Health Econometrics methodological review | Distribution-aware counts and hurdle models | Modeling alone is not identification |
| Gidwani et al. (2024) | Contemporary U.S. chronically ill enrollees | Telehealth cost-sharing DiD | Different treatment, population and period |
| Chen et al. (2026) | Chinese lower-income chronic-disease patients | Matching plus DiD | Different insurance program and data |
| Tran et al. (2026) | U.S. privately insured preventive care patients | Unexpected payment exposure, matched cohort | Different outcome: inappropriate OOP costs |

## Defensible contribution of this GitHub project

The **value added here is an open, same-data research-engineering benchmark** combining:
- Poisson conditional mean and HC3 robust covariance;
- estimated-dispersion Negative Binomial NB2;
- a **true logistic + zero-truncated NB2 hurdle likelihood**, separating any use from positive-use intensity;
- a plan-support audit demonstrating that 100% coinsurance is structurally tied to the individual-deductible plan;
- categorical price functional-form sensitivity within comparable non-IDP plan support;
- noncausal, observed-support-standardized model margins;
- count-aware predictive Machine Learning with a covariate-pattern holdout and explicit leakage limits.

This integration can help students, hiring committees and research collaborators inspect how interpretations change with plausible modeling choices. It is **not a new empirical cost-sharing effect** and is not sufficient for a publishable causal paper without the complete RAND trial design or an independently justified modern identification strategy.

## Limits and next research stage

The dataset is genuinely historical, but missing participant IDs prevent clustering and verification of independent observations. Unknown experimental allocation and baseline disease timing prevent causal intention-to-treat estimation. No method of sandwich covariance, hurdle fitting, explainability, or model selection restores these variables. A future original article would need an approved full RAND dataset, actual family-level assignment unit, baseline need, follow-up timing, pre-specified estimand and an independently audited systematic literature search.

## References (APA 7th edition)

Aron-Dine, A., Einav, L., & Finkelstein, A. (2013). The RAND health insurance experiment, three decades later. *Journal of Economic Perspectives, 27*(1), 197–222. https://doi.org/10.1257/jep.27.1.197

Chen, X., Feng, J., Wang, Z., & Yu, D. (2026). Insurance coverage for chronic diseases and healthcare of low-income people: Evidence from Chinese administrative data. *Health Economics, 35*(2), 294–311. https://doi.org/10.1002/hec.70050

Deb, P., & Norton, E. C. (2018). Modeling health care expenditures and use. *Annual Review of Public Health, 39*, 489–505. https://doi.org/10.1146/annurev-publhealth-040617-013517

Gidwani, R., Yank, V., Burgette, L., Kofner, A., Asch, S. M., & Wagner, Z. (2024). The impact of telehealth cost-sharing on healthcare utilization: Evidence from high-deductible health plans. *Health Services Research, 59*(6), e14343. https://doi.org/10.1111/1475-6773.14343

Manning, W. G., Newhouse, J. P., Duan, N., Keeler, E. B., Leibowitz, A., & Marquis, M. S. (1987). Health insurance and the demand for medical care: Evidence from a randomized experiment. *The American Economic Review, 77*(3), 251–277. https://pubmed.ncbi.nlm.nih.gov/10284091/

Tran, A., Laporte, A., Nauenberg, E., & Hoagland, A. (2026). Role of chronic conditions in out-of-pocket costs for preventive care in the US. *JAMA Network Open, 9*(1), e2553157. https://doi.org/10.1001/jamanetworkopen.2025.53157
