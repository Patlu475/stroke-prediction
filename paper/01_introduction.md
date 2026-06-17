# 1. Introduction

Stroke is the second leading cause of death globally, responsible for approximately 6.5 million deaths annually [WHO 2022]. Each year, an estimated 13 million people experience a stroke, and 5 million are left with permanent disability [WSO 2022]. Early identification of at-risk individuals is essential for preventive intervention, and machine learning (ML) has been widely proposed as a tool for stroke risk prediction from electronic health records and clinical data [Chadha 2024, Asadi 2024].

The volume of ML-based stroke prediction research has grown rapidly, with over 50 studies published since 2017 across journals in computer science, clinical informatics, and neurology [Chadha 2024]. These studies consistently report high predictive accuracy — frequently exceeding 95% — using models such as Random Forest, XGBoost, and logistic regression. Taken at face value, these results suggest that the problem of stroke prediction is largely solved.

However, a closer examination of this literature reveals a troubling pattern. We systematically reviewed 13 recent papers (11 modeling studies and 2 systematic reviews) and found that the field has converged on a narrow methodology that obscures the true difficulty of the task:

- **Data.** Nine of eleven modeling papers use the same Kaggle Stroke Prediction Dataset, a publicly available file of 5,110 rows with 10 self-reported features and no laboratory measurements. The dataset's provenance is undocumented, and its features exclude clinically validated stroke risk factors such as blood pressure, cholesterol, atrial fibrillation, and medication history.

- **Metrics.** The dominant evaluation metric is accuracy, which is misleading on imbalanced data. At the 4–5% stroke prevalence typical of these datasets, a model that predicts "no stroke" for every patient achieves over 95% accuracy. Three papers in our review independently demonstrated this: models with 95–96% accuracy detected zero stroke cases [Wu and Fang 2020, Frontiers 2025, Scientific Reports 2025].

- **Validation.** Not one of the eleven modeling papers validated on an independent dataset. Every reported metric comes from the same data source used for training, typically via a random holdout split or cross-validation on the Kaggle dataset.

- **Clinical context.** No paper compared its ML model against established clinical risk scores such as the Framingham Stroke Risk Profile [Wolf 1991], evaluated performance across demographic subgroups, or reported probability calibration.

The result is a body of literature that benchmarks similar models on the same small dataset, reports optimistic metrics, and provides no evidence of real-world clinical utility. This is not a criticism of any individual study, but a structural observation about a field that has optimized for publication metrics rather than clinical relevance.

In this study, we apply the same models that dominate the literature — logistic regression, Random Forest, and XGBoost — to a fundamentally different dataset: six cycles of the National Health and Nutrition Examination Survey (NHANES, 2007–2018), comprising 34,719 adults with 1,398 stroke cases and 22 clinically measured features including laboratory values, examiner-measured blood pressure, comorbidity history, and sociodemographic variables. We evaluate these models using the methodological standards the field has neglected:

1. **Temporal validation.** We train on 2007–2014 data and test on 2015–2018 data, simulating prospective deployment.
2. **Clinically meaningful metrics.** We report PR-AUC as the primary metric, supplemented by calibration curves, Brier scores, and sensitivity at fixed specificity — not accuracy alone.
3. **Clinical baseline comparison.** We benchmark all models against a simplified Framingham-like risk score computed from the same data.
4. **Fairness analysis.** We disaggregate performance by sex and race/ethnicity.
5. **Statistical rigor.** We report bootstrap 95% confidence intervals and DeLong tests for pairwise model comparison.
6. **Transparency.** We follow the TRIPOD reporting guidelines [Collins 2015] and release all code and data.

Our results are sobering. The best model achieves a PR-AUC of 0.162 (95% CI: 0.142–0.188) — an honest measure of performance on a problem with 4% prevalence. This barely exceeds a simple clinical risk score (PR-AUC = 0.144). No model is statistically distinguishable from any other on ROC-AUC (DeLong p > 0.74 for all pairs). Recall across racial subgroups ranges from 49% (Hispanic) to 76% (Non-Hispanic White). Calibration is poor. These findings do not mean ML is useless for stroke prediction — they mean the field's current approach is inadequate, and honest evaluation is the first step toward genuine progress.
