# 3. Materials and Methods

## 3.1 Data Source

We used data from the National Health and Nutrition Examination Survey (NHANES), a program of studies conducted by the National Center for Health Statistics (NCHS) within the Centers for Disease Control and Prevention (CDC). NHANES employs a complex, multistage probability sampling design to produce a nationally representative sample of the noninstitutionalized civilian US population [CDC NHANES]. Each survey cycle spans two years. Data collection includes standardized household interviews, physical examinations conducted in mobile examination centers (MECs), and laboratory analyses of blood and urine specimens.

We combined six consecutive survey cycles spanning 12 years: 2007–2008, 2009–2010, 2011–2012, 2013–2014, 2015–2016, and 2017–2018. For each cycle, we merged 11 component data files on the unique respondent identifier (SEQN): demographics (DEMO), medical conditions questionnaire (MCQ), blood pressure examination (BPX), body measures (BMX), total cholesterol (TCHOL), HDL cholesterol (HDL), glycohemoglobin (GHB), complete blood count (CBC), blood pressure and cholesterol questionnaire (BPQ), diabetes questionnaire (DIQ), and smoking questionnaire (SMQ). All data files are publicly available from the CDC NHANES website without restrictions.

## 3.2 Study Population

We included all NHANES participants aged 20 years and older with a valid response to the stroke history question (MCQ160F). Participants who refused to answer or responded "don't know" were excluded. No further exclusion criteria were applied. The final analytic sample comprised 34,719 adults, of whom 1,398 (4.0%) reported a history of stroke. The sample was split into a training set (survey cycles 2007–2008 through 2013–2014; N = 23,446; 916 stroke cases, 3.9%) and a temporal hold-out test set (survey cycles 2015–2016 through 2017–2018; N = 11,273; 482 stroke cases, 4.3%).

## 3.3 Outcome Definition

The outcome variable was self-reported stroke history, derived from NHANES question MCQ160F: "Has a doctor or other health professional ever told you that you had a stroke?" Respondents who answered "yes" were coded as stroke-positive (STROKE = 1); those who answered "no" were coded as stroke-negative (STROKE = 0). This operationalization captures stroke prevalence rather than incidence — the model identifies individuals who have experienced a stroke, not those who will experience one in the future. We acknowledge this as a limitation (Section 5.8).

## 3.4 Predictor Variables

We defined two feature sets to enable a controlled comparison between the limited features available in the Kaggle Stroke Prediction Dataset (used by 9 of 11 modeling papers in our literature review) and the richer clinical measurements available in NHANES.

**Kaggle-equivalent feature set (8 features).** Age, sex (binary), hypertension (self-reported physician diagnosis), coronary heart disease (self-reported), ever married (binary), body mass index (BMI; measured by examination staff), glycated hemoglobin (HbA1c; laboratory-measured, used as a proxy for the Kaggle dataset's average glucose level), and smoking status (never/former/current, derived from SMQ020 and SMQ040). This set omits two Kaggle features — work type and residence type — which were consistently identified as the lowest-importance predictors across all reviewed papers [Chadha 2024, Asadi 2024].

**Full clinical feature set (22 features).** The Kaggle-equivalent features plus 14 additional clinically measured variables: systolic and diastolic blood pressure (average of up to three examiner readings), total cholesterol, HDL cholesterol, white blood cell count, hemoglobin, platelet count, diabetes diagnosis, congestive heart failure, history of heart attack, waist circumference, race/ethnicity (five NHANES categories), income-to-poverty ratio, and education level. LDL cholesterol and triglycerides were excluded due to approximately 43% missingness (these require fasting blood draws administered to a subsample only).

Table 2 provides a detailed comparison of the two feature sets against the Kaggle dataset.

## 3.5 Missing Data

Missing data rates were computed for each predictor in the training set and are reported in Table 3. Among Kaggle-equivalent features, missingness ranged from 0.1% (hypertension, smoking status) to 8.3% (HbA1c). Among the additional clinical features, missingness ranged from 0.1% (diabetes) to 17.9% (HDL cholesterol). All missing values were imputed using the median of the training set. The same fitted imputer was applied to the test set to prevent information leakage. We chose median imputation for its robustness to outliers in clinical measurements; we acknowledge that more sophisticated imputation methods (e.g., multiple imputation by chained equations) could be explored in future work.

## 3.6 Experimental Design

**Temporal validation.** We split the data by survey cycle rather than by random assignment. The training set comprised the four earlier cycles (2007–2014) and the test set comprised the two later cycles (2015–2018). This temporal split simulates a prospective deployment scenario in which a model trained on historical data is applied to a future population. No stroke prediction paper in our literature review employed temporal validation.

**Cross-validation.** Within the training set, we performed 5-fold stratified cross-validation to obtain variance estimates of model performance (PR-AUC mean ± standard deviation). Stratification ensured that each fold maintained the class distribution of the full training set.

**Imbalance handling.** Stroke affected 4.0% of participants, creating a severe class imbalance. We compared three strategies:

The first strategy, *no balancing*, trained models on the original imbalanced distribution and served as a baseline to demonstrate the "accuracy trap" — models that achieve high accuracy by predicting the majority class exclusively. The second strategy, *cost-sensitive learning*, penalized misclassification of the minority class more heavily. For logistic regression and random forest, we set class weights inversely proportional to class frequency; for XGBoost, we set the positive-class weight equal to the ratio of negative to positive samples in the training set (approximately 24.6). The third strategy, *SMOTE* (Synthetic Minority Over-sampling Technique) [Chawla 2002], generated synthetic stroke-positive samples in the training set. During cross-validation, SMOTE was applied independently within each training fold to prevent data leakage — synthetic samples generated from one fold's training data never appeared in that fold's validation set. For the final temporal evaluation, SMOTE was applied to the full training set; the test set was never modified.

## 3.7 Models

We selected three models that appear most frequently in the stroke prediction literature [Chadha 2024, Asadi 2024]:

Logistic Regression (LR) was configured as a linear model with L2 regularization and a maximum of 1,000 iterations to ensure convergence. Random Forest (RF) was configured as an ensemble of 300 decision trees with a maximum depth of 15. XGBoost was configured with 300 gradient-boosted trees, a maximum depth of 6, a learning rate of 0.1, and log loss as the evaluation metric.

All models used a fixed random seed of 42 for reproducibility. Hyperparameters were not tuned via grid search; we used reasonable defaults consistent with those reported in the reviewed literature. This design choice was intentional: the paper's thesis is that data quality and evaluation methodology matter more than model optimization.

## 3.8 Evaluation Metrics

We designated the area under the precision-recall curve (PR-AUC, also called average precision) as the primary evaluation metric. PR-AUC is more informative than the area under the receiver operating characteristic curve (ROC-AUC) on imbalanced datasets because it is sensitive to false positives in the minority class [Saito and Rehmsmeier 2015]. We report ROC-AUC for comparability with prior work.

Additional metrics included: F1 score, precision (positive predictive value), recall (sensitivity), specificity, accuracy, Brier score (a measure of calibration quality, where 0 is perfect), and sensitivity at 90% specificity (a clinically relevant operating point). Raw confusion matrix counts (TP, FP, TN, FN) are reported for all experiments to enable independent verification.

## 3.9 Clinical Baseline

To assess whether machine learning models provide value beyond existing clinical tools, we implemented a simplified Framingham-like stroke risk score using variables available in NHANES. The score assigns points based on age (0–4 points by decade), systolic blood pressure (0–3 points by severity), diabetes (2 points), hypertension (1 point), coronary heart disease (2 points), congestive heart failure (2 points), and history of heart attack (1 point). The total score was normalized to the [0, 1] range and treated as a predicted probability for metric computation. While this is an approximation of the full Framingham Stroke Risk Profile [Wolf 1991], it uses the same core risk factors and provides a meaningful clinical reference point. No prior stroke prediction paper in our literature review compared ML models against any clinical risk score.

## 3.10 Fairness Analysis

We disaggregated the best-performing model's test-set performance by sex (male, female) and race/ethnicity (Mexican American, Other Hispanic, Non-Hispanic White, Non-Hispanic Black, Other/Multiracial — the five categories defined by NHANES variable RIDRETH1). For each subgroup, we computed the full metric suite (Section 3.8). Subgroups with fewer than 10 members or fewer than 2 stroke cases were excluded from analysis to ensure metric stability. No prior stroke prediction paper in our literature review reported fairness or subgroup analysis.

## 3.11 Statistical Analysis

**Bootstrap confidence intervals.** For the best-performing model on each feature set, we computed 95% confidence intervals for all metrics using 1,000 bootstrap iterations on the temporal test set. In each iteration, we sampled the test set with replacement and recomputed all metrics. The 2.5th and 97.5th percentiles of the bootstrap distribution defined the confidence interval bounds.

**DeLong test.** We performed pairwise DeLong tests [DeLong 1988] to compare ROC-AUC values between models within the same feature set and imbalance strategy. The DeLong test provides a nonparametric z-statistic and associated p-value for the null hypothesis that two ROC curves have equal AUC. We report two-sided p-values.

## 3.12 Software and Reproducibility

All analyses were conducted in Python using scikit-learn (v1.9.0), XGBoost (v3.2.0), imbalanced-learn (v0.14.2), and SciPy. Figures were generated with Matplotlib and Seaborn. The complete source code, merged dataset, and all results are publicly available at https://github.com/Patlu475/stroke-prediction. The study followed the TRIPOD reporting guidelines [Collins 2015]; a completed checklist is provided in the Supplementary Materials.

## 3.13 Ethics Statement

NHANES protocols were approved by the NCHS Research Ethics Review Board. All participants provided written informed consent. The data used in this study are de-identified and publicly available; no additional institutional review board approval was required.
