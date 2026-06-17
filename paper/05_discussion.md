# 5. Discussion

## 5.1 Summary of Key Findings

We evaluated three commonly used ML models for stroke prediction on NHANES clinical data with temporal validation, proper evaluation metrics, clinical baseline comparison, and fairness analysis. Six findings define this study:

1. The "accuracy trap" — models without class balancing achieve 95.7% accuracy while detecting 0–9 of 482 stroke cases — is reproducible across datasets and models.
2. Logistic regression outperformed Random Forest and XGBoost on PR-AUC in all 12 configurations, but no pairwise difference was statistically significant on ROC-AUC (DeLong p > 0.74).
3. Adding 14 clinical features (measured BP, cholesterol, comorbidities, demographics) to the Kaggle-equivalent feature set improved PR-AUC by 12.9% (0.143 to 0.162).
4. The best ML model (LR, full clinical, PR-AUC = 0.162) only marginally outperformed a simplified Framingham risk score (PR-AUC = 0.144).
5. Recall ranged from 48.6% (Other Hispanic) to 76.1% (Non-Hispanic White) — a 27-percentage-point fairness gap that no prior study measured.
6. Calibration was poor (Brier = 0.151), with the model systematically overestimating stroke probability.

## 5.2 The Accuracy Trap Is Real and Reproducible

Our results confirm what three prior papers found independently [Wu and Fang 2020, Frontiers 2025, Scientific Reports 2025]: on data with 4–5% stroke prevalence, models trained without class balancing learn to predict the majority class and achieve superficially impressive accuracy. What is new in our study is the demonstration that this occurs on a different, larger dataset (NHANES, N = 34,719 vs. Kaggle N = 5,110 or CLHLS N = 1,131) and persists across all three models. The accuracy trap is not a property of a specific dataset or model — it is an inevitable consequence of evaluating imbalanced classification with accuracy.

This finding has implications beyond stroke prediction. Any binary classification task with low prevalence — sepsis prediction, rare disease screening, fraud detection — is susceptible to the same trap. The remedy is straightforward: report precision-recall metrics (PR-AUC, F1) instead of or alongside accuracy, and always report the confusion matrix.

## 5.3 Why Logistic Regression Outperforms Complex Models

A surprising finding is that logistic regression — the simplest model in our comparison — achieved the highest PR-AUC in every configuration. This result becomes less surprising in context. On a dataset where the positive class represents 4% of samples and the signal-to-noise ratio is low, regularized linear models are less prone to overfitting spurious patterns. Random Forest and XGBoost, with their capacity for complex interactions, may overfit to the training distribution in ways that do not transfer across the temporal split.

This echoes Spittal et al.'s [2025] meta-analytic finding across 53 studies that "ML algorithms were no better than traditional risk assessment scales." Our DeLong tests formalize this: the ROC-AUC differences between models are not statistically significant (all p > 0.74). The field's emphasis on ever-more-complex architectures — stacking ensembles [Sensors 2022], meta-learning frameworks [Sensors 2025], WGAN-GP data augmentation [Gao 2024] — produces diminishing returns when the underlying signal is weak and the dataset is small.

## 5.4 Clinical Features Help, but the Ceiling Is Low

The full clinical feature set (22 features) improved PR-AUC by 12.9% relative to the Kaggle-equivalent set (8 features). Measured blood pressure, cholesterol, hemoglobin, comorbidity history, and sociodemographic variables each contributed additional signal. This confirms what the literature has assumed but never tested: clinical measurements matter for stroke prediction, and the Kaggle dataset's 10 self-reported features are insufficient.

However, the absolute performance ceiling remains low. A PR-AUC of 0.162 means that at any recall level, the majority of flagged patients will be false positives. This reflects a fundamental challenge: stroke is a rare event (4% prevalence) with many shared risk factors among stroke and non-stroke populations (hypertension, diabetes, and aging are common in both groups). Feature enrichment helps, but the overlap between classes limits separability regardless of the model.

## 5.5 ML Barely Beats Framingham

The comparison with the simplified Framingham score is perhaps the most consequential finding. The Framingham score — a simple point-based system using age, blood pressure, diabetes, and cardiovascular history — achieved PR-AUC = 0.144, compared to the best ML model's 0.162. The ML model's advantage is real but modest (12.5% relative improvement), and it comes at the cost of worse calibration (Brier 0.151 vs. 0.092).

No prior stroke prediction paper compared against any clinical risk score. This means the field has no evidence that ML adds value over what clinicians already use. Our results suggest the added value is marginal — at least for this task, this population, and this feature set. Future studies must include clinical baselines to justify the complexity, interpretability loss, and deployment burden of ML systems.

## 5.6 Fairness Gaps Demand Attention

The 27-percentage-point recall gap between Non-Hispanic White (76.1%) and Other Hispanic (48.6%) participants is a critical finding. A model deployed in clinical practice with this level of disparity would systematically underdetect stroke in Hispanic patients — a population already facing barriers to stroke care [Cruz-Flores 2011].

The inverse specificity pattern — higher specificity for Hispanic subgroups (0.836–0.867) than for White and Black subgroups (0.728–0.735) — suggests the model has learned to be more conservative for Hispanic patients. This may reflect differences in feature distributions, sample sizes, or stroke risk factor profiles across groups. Regardless of the mechanism, the disparity must be measured and reported. No prior stroke prediction study did so.

Reeves et al. [2022] demonstrated that race-aware resampling could reduce fairness gaps by more than 50% in a structured-data suicide prediction task. Whether similar techniques would improve equity in stroke prediction is an open question that our results motivate.

## 5.7 Calibration Is Poor

The calibration curve (Figure 3) reveals systematic overestimation of stroke probability. When the model assigns a 50% stroke probability, the true frequency is approximately 7%. The Brier score of 0.151 quantifies this miscalibration. By contrast, the Framingham score achieves a Brier score of 0.092, indicating substantially better probability estimates despite lower discrimination.

This finding has direct implications for deployment. A model used for clinical decision support must produce reliable probability estimates — not just correct rankings. If a clinician acts on a model's "high risk" designation, the probability behind that designation must be trustworthy. Post-hoc calibration methods (e.g., Platt scaling, isotonic regression) could address this, but no paper in our literature review explored calibration at all.

Cost-sensitive weighting, which improved recall, appears to be the source of miscalibration. By upweighting the minority class during training, the model shifts its decision boundary toward predicting more positives, inflating predicted probabilities. This recall-calibration trade-off is well-documented in the ML literature but absent from the stroke prediction literature.

## 5.8 Limitations

This study has several limitations that should guide interpretation:

1. **Self-reported outcome.** Stroke history was ascertained by participant self-report, not clinical adjudication. This introduces recall bias and misclassification. The model predicts stroke prevalence, not future stroke incidence — a weaker clinical objective than prospective risk prediction.

2. **Cross-sectional design.** NHANES collects data at a single time point per participant. We cannot model how risk factors evolve over time. The temporal split across survey cycles provides stronger validation than a random split, but it is not equivalent to a longitudinal cohort study with incident stroke outcomes.

3. **No imaging or ECG data.** NHANES does not include brain imaging or electrocardiography. Atrial fibrillation, a major stroke risk factor, is not captured in our feature set.

4. **HbA1c as glucose proxy.** We used HbA1c as a proxy for fasting blood glucose. While HbA1c reflects average glycemic control over 2–3 months (arguably a more stable measure), it is not the same variable as the Kaggle dataset's average glucose level. Direct comparison of feature importance across datasets should account for this.

5. **Survivor bias.** NHANES surveys living, non-institutionalized adults. Individuals who died from fatal strokes are excluded, biasing the sample toward less severe stroke cases.

6. **Simplified Framingham score.** Our clinical baseline is an approximation of the Framingham Stroke Risk Profile, not the validated original. It omits atrial fibrillation and does not use sex-specific coefficients. A comparison against the full Framingham model or CHA₂DS₂-VASc would be more definitive.

7. **Median imputation.** We used a simple imputation strategy. Multiple imputation by chained equations (MICE) could better preserve distributional properties, particularly for features with higher missingness such as HDL cholesterol (17.9%).

## 5.9 Recommendations for the Field

Based on our literature review and experimental findings, we offer seven recommendations for future stroke prediction studies:

1. **Use clinical data, not Kaggle.** The Kaggle Stroke Prediction Dataset has served as a useful pedagogical tool, but it lacks the feature richness, sample size, and provenance documentation needed for clinical research. NHANES, MIMIC-IV, UK Biobank, and hospital EHR systems provide richer alternatives.

2. **Report PR-AUC, not accuracy.** On imbalanced data, accuracy is meaningless. PR-AUC, F1, and calibration (Brier score) should be the primary metrics. Always include the confusion matrix.

3. **Validate temporally or externally.** Random train/test splits on the same dataset do not test generalization. Temporal splits, cross-institutional validation, or geographic hold-outs are necessary to demonstrate robustness.

4. **Compare against clinical baselines.** If an ML model does not outperform a Framingham score or CHA₂DS₂-VASc, there is no justification for its deployment complexity. Every ML study should include at least one clinical comparator.

5. **Evaluate fairness.** Performance must be disaggregated by sex, race/ethnicity, and age group at minimum. Disparities should be measured and reported even if they cannot be resolved within the study.

6. **Report calibration.** Predicted probabilities must be reliable for clinical use. Calibration curves and Brier scores should be standard. Post-hoc calibration techniques should be explored when miscalibration is detected.

7. **Follow TRIPOD.** The TRIPOD reporting guidelines [Collins 2015] exist specifically for prediction model studies. Adherence ensures transparent, reproducible, and comparable reporting. Of the 53 studies evaluated in Spittal et al.'s [2025] meta-analysis, only 3 were at low risk of bias. The bar must be raised.
