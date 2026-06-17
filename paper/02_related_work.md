# 2. Related Work

## 2.1 Machine Learning for Stroke Prediction

Machine learning approaches to stroke prediction have grown rapidly since 2017, with Random Forest emerging as the most frequently used and best-performing algorithm across studies [Chadha 2024, Asadi 2024]. We reviewed 13 papers — 11 modeling studies and 2 systematic reviews — to understand the current landscape.

The majority of modeling studies use the Kaggle Stroke Prediction Dataset, a publicly available tabular dataset of 5,110 records with 10 features (age, sex, hypertension, heart disease, marital status, work type, residence type, average glucose level, BMI, and smoking status). Among the 11 modeling papers we reviewed, 9 used this dataset or a near-identical variant [Sinkron 2025, IJACSA 2021, Gao 2024, Diagnostics 2024, Frontiers 2025, Frontiers 2026, Sensors 2022, Sensors 2025, Scientific Reports 2025]. Only Wu and Fang [2020] used a clinical cohort (the Chinese Longitudinal Healthy Longevity Study, N = 1,131), and only one study [Scientific Reports 2024] used hospital medical records (N = 663 from a Tehran hospital — though with severe methodological issues, discussed below).

Reported accuracy ranges from 82% [IJACSA 2021] to 99.2% [Sensors 2025], and ROC-AUC from 0.82 to 1.00. These numbers suggest the problem is nearly solved. However, closer examination reveals systemic issues.

## 2.2 The Accuracy Trap

Three papers independently demonstrated that high accuracy on imbalanced stroke data is illusory. Wu and Fang [2020] found that all three of their models (regularized logistic regression, SVM, random forest) achieved 90–95% accuracy on the CLHLS dataset but detected no stroke cases (sensitivity approximately 0.00, AUC approximately 0.50). The Frontiers in Neurology study [2025] reported that logistic regression achieved 95.11% accuracy with an ROC-AUC of 0.837 on the Kaggle dataset, yet detected zero true positives out of 50 stroke cases in the test set. The Scientific Reports study [2025] confirmed the pattern: 96% accuracy with precision, recall, and F1 of zero for the stroke class.

Despite these findings, subsequent papers continued to report accuracy as the headline metric without acknowledging its limitations in imbalanced settings [Sinkron 2025, Diagnostics 2024, Sensors 2025].

## 2.3 Methodological Concerns

**No external validation.** Not one of the 11 modeling papers validated on an independent dataset. Every model was trained and tested on the same data source, using either random holdout splits or cross-validation. The two systematic reviews confirmed this as a field-wide pattern: Chadha [2024] found that external validation was absent across 31 reviewed studies, and Asadi et al. [2024] reported the same across 20 studies.

**SMOTE leakage.** Several papers applied SMOTE or other oversampling techniques with ambiguous methodology. Papers by Dritsas and Trigka [2022] and Sensors [2025] did not clearly state whether resampling was applied before or after cross-validation splits. If synthetic minority samples generated from the full dataset appear in both training and validation folds, performance estimates are inflated — a form of data leakage. The near-perfect results reported by these papers (AUC > 0.98) are consistent with this leakage pattern.

**Target leakage.** One study [Scientific Reports 2024] included aphasia, hemiparesis, and cerebrovascular accident (CVA) as input features for stroke prediction. These are clinical manifestations or diagnoses of stroke itself, meaning the model was detecting stroke from its own symptoms rather than predicting it from antecedent risk factors. The reported RF accuracy of 99.9% is a direct consequence of this leakage.

**Feature poverty.** The Kaggle dataset provides 10 features, none of which are laboratory measurements. Clinically important stroke risk factors — total cholesterol, HDL, blood pressure (measured, not self-reported), HbA1c, atrial fibrillation, medication history, and family history — are absent. Both systematic reviews noted this limitation but no modeling paper addressed it.

## 2.4 Systematic Reviews

Two systematic reviews provide field-level assessments. Chadha [2024] reviewed 31 studies (2017–2023) using PRISMA guidelines, finding that Random Forest was the most popular and best-performing model and that the Kaggle dataset dominated. The review identified the absence of external validation and clinical deployment as recurring limitations.

Asadi et al. [2024] reviewed 20 studies (2019–2023), reporting that Random Forest was the best algorithm in 25% of articles and that 10 of 20 studies used Kaggle data. Age was identified as the most consistently significant feature. The review excluded deep learning and imaging-based studies, and did not apply a formal quality assessment tool (e.g., QUADAS-2).

Neither review assessed calibration, fairness, or comparison against established clinical risk scores — gaps our study addresses.

## 2.5 Gaps in the Literature

Our review identified 11 recurring gaps across the 13 papers. The most critical are:

1. **No external or temporal validation** (0/11 modeling papers). Every model is evaluated on the same data it was trained on.
2. **Kaggle monoculture** (9/11 papers). The same limited dataset is reused, producing a benchmarking echo chamber.
3. **Accuracy as the primary metric** (majority of papers). On a 4–5% prevalence dataset, a model that always predicts "no stroke" achieves >95% accuracy.
4. **No comparison with clinical tools** (0/11 papers). No paper benchmarks against the Framingham Stroke Risk Profile, CHA₂DS₂-VASc, or any validated clinical risk score.
5. **No fairness analysis** (0/11 papers). No paper evaluates whether models perform equitably across demographic subgroups.
6. **No calibration reporting** (10/11 papers). Only one paper [Frontiers 2026] reported a Brier score; none reported calibration curves.

The present study was designed to address these gaps systematically using a single, publicly available clinical dataset and transparent methodology.
