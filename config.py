"""
Central configuration — feature definitions, model hyperparameters, paths.
Edit this file to change the experiment without touching any logic.
"""

import os

RANDOM_STATE = 42
CSV_PATH = os.environ.get('NHANES_CSV', 'nhanes_merged.csv')
RESULTS_DIR = os.environ.get('NHANES_RESULTS', 'results')

# ── Feature sets ──

KAGGLE_EQUIV_FEATURES = [
    'RIDAGEYR',           # age
    'SEX',                # gender (1=Male, 0=Female)
    'HYPERTENSION',       # hypertension (1=Yes, 0=No)
    'HEART_DISEASE_CHD',  # heart_disease (1=Yes, 0=No)
    'EVER_MARRIED',       # ever_married (1=Yes, 0=No)
    'BMXBMI',             # bmi (measured)
    'LBXGH',              # HbA1c — proxy for avg_glucose_level
    'SMOKING_STATUS',     # 0=never, 1=former, 2=current
]

FULL_CLINICAL_FEATURES = KAGGLE_EQUIV_FEATURES + [
    'AVG_BPXSY',          # systolic BP (examiner-measured, avg of 3 readings)
    'AVG_BPXDI',          # diastolic BP (examiner-measured, avg of 3 readings)
    'LBXTC',              # total cholesterol (lab)
    'LBDHDD',             # HDL cholesterol (lab)
    'LBXWBCSI',           # white blood cell count (lab)
    'LBXHGB',             # hemoglobin (lab)
    'LBXPLTSI',           # platelet count (lab)
    'DIABETES',           # diabetes diagnosis (1=Yes, 0=No)
    'HEART_FAILURE',      # congestive heart failure (1=Yes, 0=No)
    'HEART_ATTACK',       # heart attack history (1=Yes, 0=No)
    'BMXWAIST',           # waist circumference (measured)
    'RIDRETH1',           # race/ethnicity (1-5 categorical)
    'INDFMPIR',           # income-to-poverty ratio (continuous)
    'DMDEDUC2',           # education level (1-5 categorical)
]

FEATURE_SETS = {
    'kaggle_equiv': KAGGLE_EQUIV_FEATURES,
    'full_clinical': FULL_CLINICAL_FEATURES,
}

STRATEGIES = ['none', 'cost_sensitive', 'smote']

# ── Model hyperparameters ──

MODEL_PARAMS = {
    'LR':      {'max_iter': 1000},
    'RF':      {'n_estimators': 300, 'max_depth': 15, 'n_jobs': -1},
    'XGBoost': {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.1,
                'eval_metric': 'logloss', 'verbosity': 0},
}

CV_FOLDS = 5

# ── Fairness subgroups ──

FAIRNESS_GROUPS = [
    ('Sex', 'SEX', {1: 'Male', 0: 'Female'}),
    ('Race/Ethnicity', 'RIDRETH1', {
        1: 'Mexican American', 2: 'Other Hispanic',
        3: 'NH White', 4: 'NH Black', 5: 'Other/Multi'}),
]

# ── Published paper results for comparison table ──

PUBLISHED_RESULTS = [
    {'Paper': 'Sinkron 2025',      'Model': 'XGBoost',      'Dataset': 'Kaggle 5,110',
     'Reported Accuracy': 0.95,    'Reported AUC': None,     'Imbalance': 'SMOTE'},
    {'Paper': 'IJACSA 2021',       'Model': 'Naive Bayes',  'Dataset': 'Kaggle 498',
     'Reported Accuracy': 0.82,    'Reported AUC': 0.82,     'Imbalance': 'Undersampling'},
    {'Paper': 'Gao 2024',          'Model': 'DNN',          'Dataset': 'Kaggle 10K (WGAN-GP)',
     'Reported Accuracy': None,    'Reported AUC': None,     'Imbalance': 'WGAN-GP',
     'Reported F1': 0.97},
    {'Paper': 'Diagnostics 2024',  'Model': 'XGBoost',      'Dataset': 'Kaggle 4,981',
     'Reported Accuracy': 0.92,    'Reported AUC': 0.97,     'Imbalance': 'Down+Upsample'},
    {'Paper': 'Frontiers 2025',    'Model': 'LR',           'Dataset': 'Kaggle 5,110',
     'Reported Accuracy': 0.95,    'Reported AUC': 0.84,     'Imbalance': 'Random Oversample',
     'Note': '0 true positives'},
    {'Paper': 'Sensors 2022',      'Model': 'Stacking',     'Dataset': 'Kaggle 3,254',
     'Reported Accuracy': 0.98,    'Reported AUC': 0.989,    'Imbalance': 'SMOTE (leakage risk)'},
    {'Paper': 'Sensors 2025',      'Model': 'RF+LGBM Meta', 'Dataset': '3 Kaggle sets',
     'Reported Accuracy': 0.992,   'Reported AUC': 1.000,    'Imbalance': 'SMOTE-SMOTEENN (leakage risk)'},
]
