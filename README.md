# Brain Stroke Prediction: NHANES Reality Check

A methodological critique of ML-based stroke prediction. We run the same models the field uses (RF, XGBoost, LR) on real NHANES clinical data instead of the Kaggle dataset that dominates the literature, with proper evaluation methodology.

## Why this exists

We reviewed 13 published papers on stroke prediction with ML. Key findings:
- 9/11 modeling papers use the same Kaggle dataset (5,110 rows, 10 features)
- 3 papers independently show that "95% accuracy" = 0 stroke cases detected
- 0/11 papers validate externally, compare against clinical baselines, evaluate fairness, or report calibration
- 0/11 papers compare against Framingham or CHA2DS2-VASc risk scores

This project addresses those gaps using NHANES (National Health and Nutrition Examination Survey) data from the CDC.

## Dataset: NHANES 2007-2018

|                    | Kaggle (used by 9/11 papers) | NHANES (this project)         |
|--------------------|------------------------------|-------------------------------|
| Sample size        | 5,110                        | **34,719**                    |
| Stroke cases       | 249                          | **1,398**                     |
| Features           | 10 (self-reported)           | **22+ (clinically measured)** |
| Lab measurements   | 0                            | **7** (cholesterol, HDL, HbA1c, WBC, hemoglobin, platelets) |
| Blood pressure     | No                           | **Yes** (3 readings averaged, examiner-measured) |
| Race/ethnicity     | No                           | **Yes** (enables fairness analysis) |
| Income/SES         | No                           | **Yes** (income-to-poverty ratio) |
| Temporal split     | No                           | **Yes** (train 2007-2014, test 2015-2018) |
| Data source        | Unknown provenance           | **CDC NHANES** (nationally representative, peer-reviewed methodology) |

### Data files

- `nhanes_merged.csv` (6.4 MB) — Pre-merged dataset, ready to use. 34,719 rows, 42 columns.
- `*.xpt` files — Raw NHANES SAS transport files downloaded from [CDC NHANES](https://wwwn.cdc.gov/Nchs/Nhanes/). Only needed if you want to rebuild the CSV from scratch.

### How the CSV was built

The merge joins 13 NHANES component files per cycle across 6 survey cycles (2007-2008 through 2017-2018) on the participant ID (`SEQN`):

| Component      | File prefix | Key variables                                  |
|----------------|-------------|------------------------------------------------|
| Demographics   | DEMO        | Age, sex, race/ethnicity, education, income    |
| Medical Cond.  | MCQ         | Stroke (MCQ160F), CHD, CHF, heart attack       |
| BP Exam        | BPX         | Systolic/diastolic BP (3 readings each)        |
| Body Measures  | BMX         | BMI, waist circumference                       |
| Total Chol.    | TCHOL       | Total cholesterol (lab)                         |
| HDL Chol.      | HDL         | HDL cholesterol (lab)                           |
| HbA1c          | GHB         | Glycohemoglobin (lab)                           |
| CBC            | CBC         | WBC, hemoglobin, platelets (lab)                |
| BP Quest.      | BPQ         | Told high BP (self-report)                      |
| Diabetes       | DIQ         | Diabetes diagnosis (self-report)                |
| Smoking        | SMQ         | Smoking history and current status              |

Filters applied: adults 20+ with valid stroke response (MCQ160F = 1 or 2).

### Derived columns

| Column             | Source                          | Values                        |
|--------------------|---------------------------------|-------------------------------|
| STROKE             | MCQ160F                         | 1 = yes, 0 = no (target)     |
| SEX                | RIAGENDR                        | 1 = male, 0 = female         |
| HYPERTENSION       | BPQ020                          | 1 = told high BP, 0 = no     |
| DIABETES           | DIQ010                          | 1 = yes, 0 = no/borderline   |
| HEART_DISEASE_CHD  | MCQ160C                         | 1 = yes, 0 = no              |
| HEART_FAILURE      | MCQ160B                         | 1 = yes, 0 = no              |
| HEART_ATTACK       | MCQ160E                         | 1 = yes, 0 = no              |
| EVER_MARRIED       | DMDMARTL                        | 1 = married/widowed/divorced/separated, 0 = never married |
| AVG_BPXSY          | mean(BPXSY1, BPXSY2, BPXSY3)   | Average systolic BP (mmHg)    |
| AVG_BPXDI          | mean(BPXDI1, BPXDI2, BPXDI3)   | Average diastolic BP (mmHg)   |
| SMOKING_STATUS     | SMQ020 + SMQ040                 | 0 = never, 1 = former, 2 = current |
| SPLIT              | CYCLE                           | "train" (2007-2014), "test" (2015-2018) |

### Known limitations of the dataset

- **Stroke outcome is self-reported** ("Has a doctor ever told you that you had a stroke?"), not clinically adjudicated. This means it captures stroke prevalence, not incidence — the model predicts "who has had a stroke" not "who will have a stroke."
- **LDL cholesterol and triglycerides** were excluded due to ~43% missingness (they require fasting blood draws, which only a subsample provides).
- **Cross-sectional design.** NHANES examines each participant once. There is no longitudinal follow-up within the dataset itself, though the temporal split across survey cycles simulates prospective validation.
- **HbA1c as proxy for blood glucose.** The Kaggle dataset uses `avg_glucose_level` (a single fasting glucose reading). NHANES provides HbA1c (glycated hemoglobin), which measures average blood sugar over 2-3 months. Arguably a better measure, but not the same variable.

## Experimental Design

### Models (same as the literature)
- Logistic Regression (LR)
- Random Forest (RF, n_estimators=300, max_depth=15)
- XGBoost (n_estimators=300, max_depth=6, lr=0.1)

All hyperparameters are defined in `config.py`.

### Feature sets
1. **Kaggle-equivalent (8 features):** age, sex, hypertension, heart disease, ever married, BMI, HbA1c, smoking status
2. **Full clinical (22 features):** Kaggle-equiv + measured SBP/DBP, total cholesterol, HDL, WBC, hemoglobin, platelets, diabetes, CHF, heart attack, waist circumference, race/ethnicity, income, education

Feature lists are defined in `config.py` and can be modified without touching any other code.

### Imbalance handling (3 strategies compared)
1. **None** — demonstrates the "accuracy trap" (high accuracy, zero recall)
2. **Cost-sensitive** — class_weight='balanced' for LR/RF, scale_pos_weight for XGBoost
3. **SMOTE** — applied to training data only; applied inside each CV fold during cross-validation to prevent leakage

### Validation
- **5-fold stratified cross-validation** within the training set (2007-2014) for model comparison with variance estimates. SMOTE is applied inside each fold (`src/models.py:cross_validate`).
- **Temporal hold-out test** on 2015-2018 data for final evaluation. This test set is never seen during training or tuning.

### Metrics reported
| Metric              | Why                                                        |
|---------------------|------------------------------------------------------------|
| PR-AUC (primary)    | Honest on imbalanced data; ROC-AUC overestimates performance |
| ROC-AUC             | For comparability with prior work                          |
| F1, Precision, Recall | Standard classification metrics                          |
| Specificity         | How well the model avoids false alarms                     |
| Brier Score         | Calibration quality (lower = better)                       |
| Sensitivity@90% Specificity | Clinically relevant operating point               |
| TP, FP, TN, FN     | Raw confusion matrix counts                                |

### Additional analyses
- **Framingham-style baseline** (`src/evaluation.py:framingham_score`): Simplified clinical risk score using age, SBP, diabetes, hypertension, and CVD history. Benchmarks whether ML adds value over existing clinical tools.
- **Fairness analysis** (`src/fairness.py`): Performance disaggregated by sex (male/female) and race/ethnicity (5 NHANES categories). No prior stroke ML paper does this. Subgroups are configured in `config.py:FAIRNESS_GROUPS`.
- **Feature importance** (`src/plots.py:fig4_feature_importance`): LR coefficients (with direction of effect) or tree-based importances for the best model.

## Project Structure

```
NHANES_data/
  run.py                     # Entry point — thin orchestrator, no business logic
  config.py                  # All constants: features, hyperparams, paths, subgroups
  requirements.txt           # Python dependencies
  README.md                  # This file
  nhanes_merged.csv          # Pre-merged dataset (6.4 MB, 34,719 rows, 42 cols)
  src/
    __init__.py
    data.py                  # load_data(), prepare_features(), get_missing_rates()
    models.py                # get_models(), train_and_evaluate(), cross_validate()
    evaluation.py            # compute_metrics(), framingham_score()
    fairness.py              # run_fairness_analysis()
    tables.py                # table1_demographics() ... table10_vs_published()
    plots.py                 # fig1_pr_auc_bars() ... fig6_fairness_bars()
  results/                   # Output directory (created by run.py)
    main_results.csv         # All 18 experiments with full metrics
    cv_results.csv           # 5-fold CV PR-AUC (mean +/- std) per experiment
    fairness_results.csv     # Per-group performance by sex and race
    feature_importance.csv   # Feature importance scores from best model
    table1_demographics.csv  # Descriptive statistics (train vs test)
    table2_dataset_comparison.csv  # NHANES vs Kaggle comparison
    table3_missing_data.csv  # Missing data rates per feature per set
    table8_framingham_comparison.csv  # Framingham vs ML models
    table10_vs_published.csv # Our results vs published Kaggle papers
    fig1_pr_auc_comparison.png  # PR-AUC by strategy and feature set
    fig2_roc_pr_curves.png      # ROC + PR curves with Framingham baseline
    fig3_calibration_curve.png  # Calibration diagram for best model
    fig4_feature_importance.png # LR coefficients / feature importances
    fig5_confusion_matrices.png # Accuracy trap vs best model side-by-side
    fig6_fairness_bars.png      # Fairness metrics by sex and race
  *.xpt                      # Raw NHANES files (not needed if using CSV)
```

### Module responsibilities

| Module             | What it does                                     | Key functions |
|--------------------|--------------------------------------------------|---------------|
| `config.py`        | Central configuration — edit this to change experiments | `FEATURE_SETS`, `MODEL_PARAMS`, `STRATEGIES` |
| `src/data.py`      | Loads CSV, imputes missing values, scales features | `load_data()`, `prepare_features()` |
| `src/models.py`    | Creates model instances, trains, runs CV          | `get_models()`, `train_and_evaluate()`, `cross_validate()` |
| `src/evaluation.py`| Computes all metrics, Framingham baseline score   | `compute_metrics()`, `framingham_score()` |
| `src/fairness.py`  | Disaggregates performance by demographic groups   | `run_fairness_analysis()` |
| `src/tables.py`    | Generates paper-ready CSV tables                  | `table1_demographics()` ... `table10_vs_published()` |
| `src/plots.py`     | Generates all figures as PNGs                     | `fig1_pr_auc_bars()` ... `fig6_fairness_bars()` |
| `run.py`           | Orchestrates the full pipeline                    | `main()` |

## How to Run

### Option 1: Google Colab (recommended)

1. Upload the full project folder (or at minimum: `nhanes_merged.csv`, `run.py`, `config.py`, and the `src/` directory)
2. Run:
```python
!pip install scikit-learn xgboost imbalanced-learn matplotlib seaborn
!python run.py
```
3. Results appear in `results/`

### Option 2: Local machine

```bash
# Create virtual environment (Python 3.10-3.13 recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python run.py
```

### Option 3: Kaggle Notebook

1. Create a new Kaggle notebook
2. Upload `nhanes_merged.csv` as a dataset
3. Upload `run.py`, `config.py`, and `src/` directory
4. Set the environment variable or edit `config.py`:
```python
# In config.py, change:
CSV_PATH = '/kaggle/input/your-dataset-name/nhanes_merged.csv'
```
5. Run `!python run.py`

### Customization

To modify the experiment without touching any logic:

- **Add/remove features:** Edit `KAGGLE_EQUIV_FEATURES` or `FULL_CLINICAL_FEATURES` in `config.py`
- **Change models or hyperparameters:** Edit `MODEL_PARAMS` in `config.py`
- **Change imbalance strategies:** Edit `STRATEGIES` in `config.py`
- **Add fairness subgroups:** Edit `FAIRNESS_GROUPS` in `config.py`
- **Change CV folds:** Edit `CV_FOLDS` in `config.py`
- **Change file paths:** Set `NHANES_CSV` and `NHANES_RESULTS` environment variables, or edit `config.py`

## Output

The script produces 9 CSV tables + 6 PNG figures + console summary.

**Tables:**

| File | Paper table | Description |
|------|-------------|-------------|
| `main_results.csv` | Table 5 | All 18 experiments (3 models x 2 feature sets x 3 strategies) |
| `cv_results.csv` | Table 5 (supplement) | 5-fold CV scores with mean +/- std |
| `table1_demographics.csv` | Table 1 | Descriptive statistics: train vs test |
| `table2_dataset_comparison.csv` | Table 2 | NHANES vs Kaggle side-by-side |
| `table3_missing_data.csv` | Table 3 | Missing data rates per feature |
| `table8_framingham_comparison.csv` | Table 8 | Framingham baseline vs ML models |
| `table10_vs_published.csv` | Table 10 | Our results vs 7 published Kaggle papers |
| `fairness_results.csv` | Table 9 | Performance by sex and race subgroups |
| `feature_importance.csv` | Table 6 | Feature importances / LR coefficients |

**Figures:**

| File | Paper figure | Description |
|------|--------------|-------------|
| `fig1_pr_auc_comparison.png` | Fig 1 | PR-AUC bars by strategy and feature set |
| `fig2_roc_pr_curves.png` | Fig 2 | ROC + PR curves with Framingham baseline |
| `fig3_calibration_curve.png` | Fig 3 | Calibration diagram for best model |
| `fig4_feature_importance.png` | Fig 4 | LR coefficients with direction of effect |
| `fig5_confusion_matrices.png` | Fig 5 | Accuracy trap vs best model side-by-side |
| `fig6_fairness_bars.png` | Fig 6 | Fairness metrics by sex and race |

## Gaps Addressed

| Gap from Literature Review | How This Project Addresses It |
|----------------------------|-------------------------------|
| G1 — No external validation | Temporal split (train 2007-2014, test 2015-2018) |
| G2 — Kaggle monoculture | Real NHANES clinical data (CDC, nationally representative) |
| G3 — Accuracy trap | Demonstrated explicitly, then superseded by proper metrics |
| G4 — SMOTE leakage | SMOTE applied inside CV folds; temporal test set never touched |
| G5 — Feature poverty | 22 clinical features vs 10 Kaggle features |
| G8 — No clinical baseline | Framingham-style risk score comparison |
| G9 — No fairness analysis | Per-group evaluation by sex and race/ethnicity |
| G10 — No calibration | Brier score + calibration curves reported |

## Citation

If you use this work, please cite the NHANES data source:

> Centers for Disease Control and Prevention (CDC). National Center for Health Statistics (NCHS). National Health and Nutrition Examination Survey Data. Hyattsville, MD: U.S. Department of Health and Human Services, Centers for Disease Control and Prevention, 2007-2018. https://www.cdc.gov/nchs/nhanes/
