# 4. Results

## 4.1 Study Population

The final sample included 34,719 adults aged 20–80 years across six NHANES cycles (Table 1). The training set (2007–2014) comprised 23,446 participants with 916 stroke cases (3.9% prevalence). The temporal test set (2015–2018) comprised 11,273 participants with 482 stroke cases (4.3% prevalence). The slight increase in stroke prevalence in the test set is consistent with the aging US population over the study period.

Demographic characteristics were similar across splits. Mean age was comparable (training: approximately 50 years; test: approximately 50 years). Males comprised approximately 48% of both sets. Hypertension prevalence was approximately 32%, diabetes approximately 13%, and coronary heart disease approximately 4%. The racial/ethnic composition reflected NHANES oversampling design, with Non-Hispanic White, Non-Hispanic Black, and Mexican American participants each well-represented. Full descriptive statistics are provided in Table 1.

## 4.2 The Accuracy Trap

Table 4 and Figure 5 present the results of training all models without any class imbalance correction (strategy = none). Every model achieved accuracy between 95.6% and 95.7% — a range that would be reported as successful performance in the majority of reviewed papers. However, this accuracy was achieved almost entirely by predicting the majority class.

On the Kaggle-equivalent feature set, logistic regression detected zero stroke cases out of 482 in the test set (TP = 0, FN = 482). Random forest detected 3 (recall = 0.6%) and XGBoost detected 2 (recall = 0.4%). On the full clinical feature set, the pattern was identical: random forest detected zero cases, logistic regression detected 3, and XGBoost detected 9 — the best result, at 1.9% recall.

Figure 5 visualizes this contrast. Panel (a) shows the confusion matrix for random forest without balancing: 10,790 true negatives, 1 false positive, 482 false negatives, and 0 true positives. The model has learned to output "no stroke" for every input. Panel (b) shows the confusion matrix for the best-performing configuration (logistic regression with cost-sensitive weighting), which detects 325 of 482 stroke cases at the cost of 2,387 false positives.

These results independently replicate the findings of three papers in our literature review. Wu and Fang [2020] reported sensitivity "approximately 0.00" on imbalanced CLHLS data. The Frontiers in Neurology study [2025] reported zero true positives at 95.11% accuracy on the Kaggle dataset. The Scientific Reports study [2025] reported precision, recall, and F1 of zero for the stroke class at 96% accuracy.

## 4.3 Main Experiment Results

Table 5 presents the full results of all 18 experiments (3 models × 2 feature sets × 3 imbalance strategies), evaluated on the 2015–2018 temporal test set. Cross-validation scores (mean ± standard deviation of PR-AUC across 5 folds) are provided in the Supplementary Materials.

The best-performing configuration was logistic regression with cost-sensitive weighting on the full clinical feature set, achieving PR-AUC = 0.162 (95% CI: 0.142–0.188), ROC-AUC = 0.823 (95% CI: 0.807–0.839), F1 = 0.204, recall = 0.674, and Brier score = 0.151. At a specificity of 90%, this model achieved 44.4% sensitivity.

Logistic regression outperformed random forest and XGBoost on PR-AUC across all 12 feature-set/strategy combinations. However, DeLong tests comparing pairwise ROC-AUC differences within the same feature set and strategy found no statistically significant differences between any model pair (all p > 0.74; Supplementary Table S2). This indicates that, on this dataset and at this sample size, model choice does not significantly affect discriminative performance — a finding consistent with Spittal et al.'s [2025] meta-analytic conclusion that "ML algorithms were no better than traditional risk assessment scales."

## 4.4 Kaggle-Equivalent vs. Full Clinical Features

Table 6 compares the best model per feature set under the same imbalance strategy (cost-sensitive weighting).

The Kaggle-equivalent set (8 features) achieved PR-AUC = 0.143 (95% CI: 0.126–0.170). The full clinical set (22 features) achieved PR-AUC = 0.162 (95% CI: 0.142–0.188), a 12.9% relative improvement. ROC-AUC improved from 0.802 to 0.823. The additional features — measured blood pressure, cholesterol, hemoglobin, platelet count, comorbidity history, race/ethnicity, income, and education — provided a modest but consistent benefit across all models and strategies.

The improvement was most pronounced for random forest (PR-AUC: 0.106 → 0.124, +17.0%) and smallest for logistic regression (PR-AUC: 0.143 → 0.162, +12.9%). This suggests that the additional clinical features contribute predictive signal that tree-based models can exploit through nonlinear interactions, but the overall ceiling remains low.

## 4.5 Clinical Baseline Comparison

Table 8 and Figure 2 compare ML models against the simplified Framingham-like clinical risk score (Section 3.9).

The Framingham score achieved ROC-AUC = 0.797 and PR-AUC = 0.144 on the test set, with a Brier score of 0.092. The best ML model (logistic regression, full clinical, cost-sensitive) achieved ROC-AUC = 0.823 and PR-AUC = 0.162, with a Brier score of 0.151. ML outperformed the clinical baseline on discrimination (ROC-AUC: +0.026; PR-AUC: +0.018) but performed worse on calibration (Brier: 0.151 vs. 0.092).

Notably, the Kaggle-equivalent ML model (PR-AUC = 0.143) performed marginally below the Framingham score (PR-AUC = 0.144). This suggests that ML models trained on the limited feature set typical of the Kaggle literature do not outperform a simple point-based clinical tool that has been available since 1991 [Wolf 1991].

Figure 2 shows the ROC and precision-recall curves for all models alongside the Framingham baseline. On the ROC plot (panel a), the curves appear well-separated; on the precision-recall plot (panel b), all models cluster near the prevalence line, illustrating the "ROC illusion" described by the Frontiers in Neurology study [2026] — ROC-AUC can mask poor performance in imbalanced settings.

## 4.6 Calibration

Figure 3 presents the calibration curve for the best model. The model is severely miscalibrated: it systematically overestimates stroke probability. When the model predicts a 50% probability of stroke, the observed fraction of positives is approximately 7%. When it predicts 90%, the observed fraction is approximately 25%. The Brier score of 0.151 reflects this poor calibration.

By contrast, the Framingham score achieved a Brier score of 0.092, indicating substantially better-calibrated probability estimates despite lower discrimination. This finding has direct clinical implications: a model deployed for patient communication or treatment decisions must produce reliable probabilities, not just correct rankings. The cost-sensitive weighting that improved recall appears to have degraded calibration — a trade-off that no paper in our literature review examined.

## 4.7 Fairness Analysis

Table 9 and Figure 6 present performance disaggregated by sex and race/ethnicity.

**By sex.** Performance was similar between males (recall = 0.677, PR-AUC = 0.154) and females (recall = 0.671, PR-AUC = 0.180). The slightly higher PR-AUC for females may reflect a higher proportion of older females in the test set. No clinically meaningful disparity was observed between sexes.

**By race/ethnicity.** Substantial disparities emerged. Non-Hispanic White participants had the highest recall at 0.761, followed by Other/Multiracial at 0.717 and Non-Hispanic Black at 0.636. Mexican American and Other Hispanic participants had markedly lower recall at 0.500 and 0.486, respectively. This means the model detects approximately 76% of strokes in White patients but only 49% in Hispanic patients — a 27-percentage-point gap.

Specificity showed an inverse pattern: Hispanic subgroups had higher specificity (0.836–0.867) than White and Black subgroups (0.728–0.735), indicating the model is more conservative (fewer false alarms) for Hispanic patients but at the cost of missing more true cases.

PR-AUC ranged from 0.102 (Other Hispanic) to 0.236 (Other/Multiracial), a 2.3-fold difference across racial groups. These disparities are consistent with the broader algorithmic fairness literature [Obermeyer 2019] and represent a critical finding given that no prior stroke prediction study evaluated demographic equity.

## 4.8 Feature Importance

Figure 4 presents the standardized logistic regression coefficients for the full clinical feature set.

Age was the dominant predictor (coefficient = 1.028), followed by hypertension (0.523) and smoking status (0.285). These align with established clinical knowledge and with the Kaggle literature, which consistently identifies age as the top feature [Chadha 2024, Asadi 2024].

Several findings diverged from the Kaggle literature. Income-to-poverty ratio had a negative coefficient (−0.217), indicating that higher income is associated with lower stroke probability — a social determinant of health that no Kaggle dataset captures. BMI showed a negative coefficient (−0.198), suggesting a protective association after adjusting for other covariates. This is consistent with the "obesity paradox" reported in stroke epidemiology [Oesch 2017], where higher BMI is associated with better post-stroke survival, and with confounding by age (older patients tend to have lower BMI). Notably, HbA1c (the glucose proxy) had the smallest coefficient magnitude (−0.030), suggesting it is captured indirectly through the diabetes diagnosis variable.

The features that the Kaggle papers assume are important — ever married, residence type, work type — had coefficients near zero or were excluded entirely, confirming the findings of both systematic reviews.

## 4.9 Comparison with Published Results

Table 10 places our results alongside seven published Kaggle-based studies. The contrast is stark:

Published papers report accuracy of 82–99.2% and ROC-AUC of 0.82–1.00. Our best model reports accuracy of 77.4% and ROC-AUC of 0.823 — modest by comparison. However, our model detects 325 of 482 stroke cases (recall = 67.4%), while three of the published models detect zero or near-zero cases despite reporting higher accuracy.

We are the only study to report PR-AUC (0.162), Brier score (0.151), bootstrap confidence intervals, DeLong statistical tests, temporal validation, clinical baseline comparison, and fairness analysis. The difference between our results and the published literature is not that our models are worse — it is that we report what the models actually do, rather than what the accuracy metric suggests they do.
