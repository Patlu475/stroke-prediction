"""TRIPOD checklist generator for prediction model reporting."""

import pandas as pd
from config import RESULTS_DIR

TRIPOD_ITEMS = [
    ('1',  'Title',              'Identify the study as developing a multivariable prediction model',
     'Title', 'Title identifies ML prediction models and clinical data source'),
    ('2',  'Abstract',           'Structured summary',
     'Abstract', 'Structured abstract: Background, Methods, Results, Conclusions with key numbers'),
    ('3a', 'Background',         'Explain the medical context and rationale',
     'Introduction', 'Stroke burden, ML promise, literature gaps'),
    ('3b', 'Objectives',         'Specify study objectives including target population and outcome',
     'Introduction', 'Predict stroke history in US adults using NHANES clinical data'),
    ('4a', 'Source of data',     'Describe the study design and setting',
     'Methods 3.1', 'NHANES 2007-2018, CDC nationally representative survey'),
    ('4b', 'Dates',              'Specify dates of data collection',
     'Methods 3.1', 'Six biennial cycles: 2007-2008 through 2017-2018'),
    ('5a', 'Participants',       'Eligibility criteria',
     'Methods 3.2', 'Adults 20+, valid stroke response (MCQ160F = 1 or 2)'),
    ('5b', 'Treatment',          'Details of any treatments received',
     'N/A', 'Cross-sectional study — no treatment assignment'),
    ('6a', 'Outcome',            'Clearly define the outcome',
     'Methods 3.3', 'Self-reported physician-diagnosed stroke (MCQ160F)'),
    ('6b', 'Outcome blinding',   'State whether assessors were blinded',
     'Methods 3.3', 'Self-reported; no assessor involvement'),
    ('7a', 'Predictors',         'Define all predictors, including timing of measurement',
     'Methods 3.4', 'Two feature sets defined: 8 Kaggle-equivalent + 22 full clinical. All measured at survey visit.'),
    ('7b', 'Predictor assessment','Report how predictors were assessed',
     'Methods 3.4', 'Labs: blood draw analyzed by CDC lab. BP: 3 examiner readings averaged. Comorbidities: self-reported.'),
    ('8',  'Sample size',        'Explain how sample size was determined',
     'Methods 3.1', 'All eligible NHANES participants across 6 cycles. N=34,719 (1,398 stroke cases).'),
    ('9',  'Missing data',       'Describe how missing data were handled',
     'Methods 3.5', 'Rates reported in Table 3. Median imputation for continuous, applied before scaling.'),
    ('10a','Statistical methods', 'Describe how predictors were handled in the analysis',
     'Methods 3.6-3.7', 'Standardized (z-score). No feature selection applied — all features in each set used.'),
    ('10b','Model building',     'Specify type of model and method for selecting predictors',
     'Methods 3.7', 'LR, RF, XGBoost. Fixed hyperparameters from config. No predictor selection step.'),
    ('10d','Validation',         'Specify measures used to assess model performance',
     'Methods 3.8', 'PR-AUC (primary), ROC-AUC, F1, Brier score, sensitivity at 90% specificity'),
    ('11', 'Risk groups',        'Provide details on how risk groups were created',
     'N/A', 'No risk groups defined — binary classification only'),
    ('12', 'Development vs validation', 'Identify model development vs validation',
     'Methods 3.6', 'Temporal split: development on 2007-2014, validation on 2015-2018'),
    ('13a','Flow of participants','Describe flow of participants',
     'Results 4.1', 'NHANES total → age filter → valid stroke response → train/test split. Flow diagram provided.'),
    ('14a','Model development',  'Specify number of participants and events',
     'Results 4.1', 'Train: 23,446 (916 stroke). Test: 11,273 (482 stroke).'),
    ('15a','Model specification', 'Present the full prediction model',
     'Methods 3.7 + Config', 'All hyperparameters in config.py. LR coefficients in Table 6 / feature_importance.csv.'),
    ('16', 'Model performance',  'Report performance measures with CIs',
     'Results 4.3', 'All metrics with 95% bootstrap CIs in Table 5. DeLong tests in supplementary.'),
    ('17', 'Model updating',     'Report results of model updating if done',
     'N/A', 'No model updating performed'),
    ('18', 'Discussion-limits',  'Discuss limitations',
     'Discussion 5.8', 'Self-reported outcome, cross-sectional, no imaging, HbA1c proxy, survivor bias'),
    ('19a','Interpretation',     'Give overall interpretation of results',
     'Discussion 5.1-5.7', 'Accuracy trap confirmed. LR > complex models. ML barely exceeds Framingham. Fairness gaps exist.'),
    ('19b','Implications',       'Discuss potential clinical use and implications',
     'Discussion 5.9', 'Seven recommendations for the field'),
    ('20', 'Supplementary',      'Provide supplementary information as appropriate',
     'Supplementary', 'Full 18-experiment table, CV details, TRIPOD checklist, DeLong matrix'),
    ('21', 'Funding',            'Funding information',
     'End matter', 'No external funding'),
    ('22', 'Data availability',  'Data and code availability',
     'End matter', 'NHANES: public CDC data. Code: github.com/Patlu475/stroke-prediction'),
]


def generate_tripod_checklist():
    """Generate and save the TRIPOD adherence checklist."""
    rows = []
    for item_num, item_name, description, section, how_addressed in TRIPOD_ITEMS:
        rows.append({
            'TRIPOD Item': item_num,
            'Item Name': item_name,
            'Requirement': description,
            'Manuscript Section': section,
            'How Addressed': how_addressed,
        })
    df = pd.DataFrame(rows)
    df.to_csv(f'{RESULTS_DIR}/supplementary/tripod_checklist.csv', index=False)
    return df
