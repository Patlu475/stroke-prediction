# 5. Discussion

## 5.1 Principal Findings

We evaluated three commonly used ML models for stroke prediction on NHANES clinical data with temporal validation, clinically meaningful metrics, a clinical baseline comparison, and fairness analysis. The results reveal a substantial gap between the performance reported in the literature and what these models achieve under rigorous evaluation. The best configuration — logistic regression with cost-sensitive weighting on 22 clinical features — achieved a PR-AUC of 0.162 (95% CI: 0.142–0.188), marginally above a simplified Framingham risk score (PR-AUC = 0.144) and far below what the field's reported 95–99% accuracy rates would suggest. No pairwise model comparison reached statistical significance on ROC-AUC (DeLong p > 0.74), and a 27-percentage-point recall gap between Non-Hispanic White and Hispanic subgroups indicates that even the best model does not serve all populations equitably.

## 5.2 The Accuracy Trap

Our finding that models without class balancing achieve 95.7% accuracy while detecting 0–9 of 482 stroke cases confirms what three prior studies observed independently [Wu and Fang 2020, Frontiers 2025, Scientific Reports 2025]. What distinguishes our result is the demonstration that this pattern persists on a different, substantially larger dataset (NHANES, N = 34,719 vs. Kaggle N = 5,110 or CLHLS N = 1,131) and across all three model families. The accuracy trap is not a property of any specific dataset or algorithm — it is an inevitable consequence of applying threshold-based classification to severely imbalanced data and evaluating with accuracy. This finding extends beyond stroke prediction to any binary classification task with low prevalence, including sepsis prediction, rare disease screening, and fraud detection. The remedy is straightforward: report precision-recall metrics alongside or instead of accuracy, and always include the confusion matrix.

## 5.3 Model Complexity Does Not Help

Logistic regression — the simplest model in our comparison — achieved the highest PR-AUC in every one of the 12 feature-set and strategy combinations. This result is less surprising than it appears. On a dataset where the positive class represents 4% of samples and the signal-to-noise ratio is low, regularized linear models are less prone to overfitting spurious patterns than tree ensembles. Random Forest and XGBoost, with their capacity for complex interactions, may capture training-set idiosyncrasies that do not transfer across the temporal split.

This finding echoes Spittal et al.'s [2025] meta-analytic conclusion across 53 studies that "ML algorithms were no better than traditional risk assessment scales." Our DeLong tests formalize this observation: the ROC-AUC differences between all model pairs are statistically non-significant. The field's emphasis on increasingly complex architectures — stacking ensembles [Sensors 2022], meta-learning frameworks [Sensors 2025], WGAN-GP augmentation [Gao 2024] — may produce diminishing returns when the underlying signal is weak and the dataset is modest in size.

## 5.4 Clinical Features Help, but the Ceiling Is Low

The full clinical feature set (22 features) improved PR-AUC by 12.9% relative to the Kaggle-equivalent set (8 features). Measured blood pressure, cholesterol, hemoglobin, comorbidity history, and sociodemographic variables each contributed additional signal. This confirms what the literature has assumed but never tested: clinical measurements matter for stroke prediction, and the Kaggle dataset's self-reported features are insufficient.

However, the absolute performance ceiling remains low. A PR-AUC of 0.162 means that at any recall level, the majority of flagged patients will be false positives. This reflects a fundamental challenge: stroke is a rare event with many shared risk factors between stroke and non-stroke populations. Hypertension, diabetes, and aging are common in both groups, limiting class separability regardless of the model or feature set. Substantially improving discrimination may require data modalities not captured in NHANES, such as brain imaging, electrocardiography, or longitudinal trajectories of risk factor change.

## 5.5 ML Barely Beats Framingham

The comparison between ML models and the simplified Framingham score is perhaps the most consequential finding. The Framingham score — a point-based system using age, blood pressure, diabetes, and cardiovascular history — achieved PR-AUC = 0.144, compared to the best ML model's 0.162. The ML model's advantage is real but modest, and it comes at the cost of substantially worse calibration (Brier 0.151 vs. 0.092).

No prior stroke prediction paper compared against any clinical risk score. This means the field has accumulated no evidence that ML adds value over what clinicians already use. Our results suggest the added value is marginal for this task, population, and feature set. Future studies must include clinical baselines to justify the complexity, interpretability loss, and deployment burden of ML systems. If a simple risk score performs comparably, the burden of proof lies with the ML model to demonstrate concrete advantages — whether in discrimination, calibration, actionable insights, or clinical workflow integration.

## 5.6 Fairness Gaps

The 27-percentage-point recall gap between Non-Hispanic White participants (76.1%) and Other Hispanic participants (48.6%) represents a critical finding. A model deployed in clinical practice with this disparity would systematically underdetect stroke in Hispanic patients — a population already facing barriers to stroke care [Cruz-Flores 2011]. The inverse specificity pattern, with higher specificity for Hispanic subgroups (0.836–0.867) than for White and Black subgroups (0.728–0.735), suggests the model has learned to be more conservative for Hispanic patients, reducing false alarms at the cost of missing true cases.

These disparities may reflect differences in feature distributions, sample sizes, or stroke risk factor profiles across groups, but regardless of mechanism, they must be measured and reported. Reeves et al. [2022] demonstrated that race-aware resampling could reduce fairness gaps by more than 50% in a structured-data suicide prediction task. Whether similar techniques would improve equity in stroke prediction is an open question motivated by our findings. No prior stroke prediction study reported any subgroup analysis.

## 5.7 Calibration

The calibration curve reveals systematic overestimation of stroke probability. When the model predicts a 50% probability, the observed frequency is approximately 7%; at a predicted 90%, the observed frequency is approximately 25%. The Brier score of 0.151 quantifies this miscalibration, compared to 0.092 for the Framingham score.

This finding has direct implications for deployment. A model used for clinical decision support must produce reliable probability estimates, not just correct rankings. Cost-sensitive weighting, which improved recall, appears to be the source of miscalibration: by upweighting the minority class during training, the model shifts its decision boundary toward predicting more positives, inflating predicted probabilities. This recall-calibration trade-off is well-documented in the general ML literature but entirely absent from the stroke prediction literature. Post-hoc calibration methods such as Platt scaling or isotonic regression could address this and should be explored in future work.

## 5.8 Limitations

Several limitations should guide interpretation of our findings. The stroke outcome is self-reported rather than clinically adjudicated, introducing recall bias and misclassification; the model predicts stroke prevalence rather than future incidence, which is a weaker clinical objective than prospective risk prediction. NHANES collects data at a single time point per participant, preventing modeling of risk factor trajectories over time. The temporal split across survey cycles provides stronger validation than a random split but is not equivalent to a longitudinal cohort study with incident outcomes. Our feature set excludes brain imaging, electrocardiography, and atrial fibrillation — a major modifiable stroke risk factor not captured in NHANES. We used HbA1c as a proxy for fasting blood glucose, which, while arguably a more stable measure of glycemic status, is not the same variable as the Kaggle dataset's average glucose level. NHANES surveys living, non-institutionalized adults, excluding individuals who died from fatal strokes and biasing the sample toward less severe cases. Our Framingham baseline is a simplified approximation that omits atrial fibrillation and sex-specific coefficients. Finally, we used median imputation rather than more sophisticated methods such as multiple imputation by chained equations.

## 5.9 Recommendations for the Field

Based on our literature review and experimental findings, we offer seven recommendations for future stroke prediction studies.

First, studies should use clinical data rather than the Kaggle dataset. NHANES, MIMIC-IV, UK Biobank, and hospital EHR systems provide richer feature sets, larger samples, and documented provenance.

Second, PR-AUC, F1, and calibration metrics should replace or supplement accuracy as the primary evaluation criteria. On imbalanced data, accuracy is misleading; the confusion matrix should always be reported.

Third, temporal or external validation is essential. Random train-test splits on the same dataset do not test generalization; temporal splits, cross-institutional validation, or geographic hold-outs are necessary to demonstrate robustness.

Fourth, every ML study should compare against at least one clinical baseline such as the Framingham Stroke Risk Profile or CHA₂DS₂-VASc. Without such comparison, there is no evidence that ML adds value over existing tools.

Fifth, performance must be disaggregated by sex, race/ethnicity, and age group at minimum. Disparities should be measured and reported even if they cannot be resolved within the study.

Sixth, calibration curves and Brier scores should be standard. Predicted probabilities must be reliable for clinical use, and post-hoc calibration techniques should be explored when miscalibration is detected.

Seventh, adherence to the TRIPOD reporting guidelines [Collins 2015] should be expected by reviewers and editors. Of the 53 studies evaluated in Spittal et al.'s [2025] meta-analysis, only 3 were at low risk of bias. Transparent, standardized reporting is a prerequisite for the field to mature.
