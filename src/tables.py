"""Paper table generators — each function returns a DataFrame and saves a CSV."""

import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from config import FEATURE_SETS, PUBLISHED_RESULTS, RESULTS_DIR
from src.evaluation import framingham_score


def table1_demographics(train_df, test_df):
    """Table 1: Descriptive statistics, train vs test."""
    rows = []
    for label, sub in [('Train (2007-2014)', train_df), ('Test (2015-2018)', test_df)]:
        n = len(sub)
        stroke_n = sub['STROKE'].sum()
        rows.append({
            'Split': label,
            'N': n,
            'Stroke n (%)': f"{stroke_n} ({stroke_n/n*100:.1f}%)",
            'Age mean (SD)': f"{sub['RIDAGEYR'].mean():.1f} ({sub['RIDAGEYR'].std():.1f})",
            'Male n (%)': f"{(sub['SEX']==1).sum()} ({(sub['SEX']==1).mean()*100:.1f}%)",
            'Hypertension n (%)': f"{(sub['HYPERTENSION']==1).sum()} ({sub['HYPERTENSION'].mean()*100:.1f}%)",
            'Diabetes n (%)': f"{(sub['DIABETES']==1).sum()} ({sub['DIABETES'].mean()*100:.1f}%)",
            'Heart Disease n (%)': f"{(sub['HEART_DISEASE_CHD']==1).sum()} ({sub['HEART_DISEASE_CHD'].mean()*100:.1f}%)",
            'Heart Failure n (%)': f"{(sub['HEART_FAILURE']==1).sum()} ({sub['HEART_FAILURE'].mean()*100:.1f}%)",
            'Heart Attack n (%)': f"{(sub['HEART_ATTACK']==1).sum()} ({sub['HEART_ATTACK'].mean()*100:.1f}%)",
            'BMI mean (SD)': f"{sub['BMXBMI'].mean():.1f} ({sub['BMXBMI'].std():.1f})",
            'SBP mean (SD)': f"{sub['AVG_BPXSY'].mean():.1f} ({sub['AVG_BPXSY'].std():.1f})",
            'DBP mean (SD)': f"{sub['AVG_BPXDI'].mean():.1f} ({sub['AVG_BPXDI'].std():.1f})",
            'HbA1c mean (SD)': f"{sub['LBXGH'].mean():.2f} ({sub['LBXGH'].std():.2f})",
            'Total Chol. mean (SD)': f"{sub['LBXTC'].mean():.1f} ({sub['LBXTC'].std():.1f})",
            'HDL mean (SD)': f"{sub['LBDHDD'].mean():.1f} ({sub['LBDHDD'].std():.1f})",
            'Smoking: Never n (%)': f"{(sub['SMOKING_STATUS']==0).sum()} ({(sub['SMOKING_STATUS']==0).sum()/n*100:.1f}%)",
            'Smoking: Former n (%)': f"{(sub['SMOKING_STATUS']==1).sum()} ({(sub['SMOKING_STATUS']==1).sum()/n*100:.1f}%)",
            'Smoking: Current n (%)': f"{(sub['SMOKING_STATUS']==2).sum()} ({(sub['SMOKING_STATUS']==2).sum()/n*100:.1f}%)",
            'NH White n (%)': f"{(sub['RIDRETH1']==3).sum()} ({(sub['RIDRETH1']==3).sum()/n*100:.1f}%)",
            'NH Black n (%)': f"{(sub['RIDRETH1']==4).sum()} ({(sub['RIDRETH1']==4).sum()/n*100:.1f}%)",
            'Mexican American n (%)': f"{(sub['RIDRETH1']==1).sum()} ({(sub['RIDRETH1']==1).sum()/n*100:.1f}%)",
            'Other Hispanic n (%)': f"{(sub['RIDRETH1']==2).sum()} ({(sub['RIDRETH1']==2).sum()/n*100:.1f}%)",
            'Other/Multi n (%)': f"{(sub['RIDRETH1']==5).sum()} ({(sub['RIDRETH1']==5).sum()/n*100:.1f}%)",
        })
    t = pd.DataFrame(rows).set_index('Split').T
    t.to_csv(f'{RESULTS_DIR}/table1_demographics.csv')
    return t


def table2_dataset_comparison():
    """Table 2: NHANES vs Kaggle dataset comparison."""
    rows = [
        {'Dimension': 'Source',              'Kaggle Stroke Dataset': 'Unknown provenance (fedesoriano)',          'NHANES (This Study)': 'CDC, nationally representative'},
        {'Dimension': 'Sample size',         'Kaggle Stroke Dataset': '5,110',                                    'NHANES (This Study)': '34,719'},
        {'Dimension': 'Stroke cases',        'Kaggle Stroke Dataset': '249 (4.9%)',                               'NHANES (This Study)': '1,398 (4.0%)'},
        {'Dimension': 'Features',            'Kaggle Stroke Dataset': '10 (self-reported)',                        'NHANES (This Study)': '22 (clinically measured)'},
        {'Dimension': 'Lab measurements',    'Kaggle Stroke Dataset': '0',                                        'NHANES (This Study)': '7 (chol., HDL, HbA1c, WBC, Hgb, plt, total chol.)'},
        {'Dimension': 'Blood pressure',      'Kaggle Stroke Dataset': 'No',                                       'NHANES (This Study)': 'Yes (examiner-measured, 3 readings averaged)'},
        {'Dimension': 'Race/ethnicity',      'Kaggle Stroke Dataset': 'No',                                       'NHANES (This Study)': 'Yes (5 categories)'},
        {'Dimension': 'Income/SES',          'Kaggle Stroke Dataset': 'No',                                       'NHANES (This Study)': 'Yes (income-to-poverty ratio)'},
        {'Dimension': 'Temporal validation', 'Kaggle Stroke Dataset': 'No (single snapshot)',                      'NHANES (This Study)': 'Yes (6 cycles, 2007-2018)'},
        {'Dimension': 'Stroke outcome',      'Kaggle Stroke Dataset': 'Unknown labeling method',                   'NHANES (This Study)': 'Self-reported physician diagnosis'},
    ]
    t = pd.DataFrame(rows)
    t.to_csv(f'{RESULTS_DIR}/table2_dataset_comparison.csv', index=False)
    return t


def table3_missing_data(train_df):
    """Table 3: Missing data rates for both feature sets."""
    rows = []
    for fs_name, fs_cols in FEATURE_SETS.items():
        for col in fs_cols:
            pct = train_df[col].isnull().mean() * 100
            rows.append({'feature_set': fs_name, 'feature': col, 'missing_pct': round(pct, 1)})
    t = pd.DataFrame(rows)
    t.to_csv(f'{RESULTS_DIR}/table3_missing_data.csv', index=False)
    return t


def table8_framingham(test_df, res_df):
    """Table 8: Framingham baseline vs best ML per feature set."""
    f_scores = framingham_score(test_df)
    y_te = test_df['STROKE'].values

    rows = [{
        'Model': 'Framingham (clinical baseline)',
        'Features': 'Age+SBP+DM+HTN+CVD',
        'Strategy': 'N/A',
        'ROC-AUC': roc_auc_score(y_te, f_scores),
        'PR-AUC': average_precision_score(y_te, f_scores),
        'Brier': brier_score_loss(y_te, f_scores),
    }]

    for fs in ['kaggle_equiv', 'full_clinical']:
        best = res_df[res_df['features'] == fs].sort_values('pr_auc', ascending=False).iloc[0]
        rows.append({
            'Model': best['model'], 'Features': fs, 'Strategy': best['strategy'],
            'ROC-AUC': best['roc_auc'], 'PR-AUC': best['pr_auc'], 'Brier': best['brier'],
        })

    t = pd.DataFrame(rows)
    t.to_csv(f'{RESULTS_DIR}/table8_framingham_comparison.csv', index=False)
    return t


def table10_vs_published(res_df):
    """Table 10: Our results vs published Kaggle paper results."""
    our_best = res_df.loc[res_df['pr_auc'].idxmax()]
    our_row = {
        'Paper': 'This study', 'Model': our_best['model'],
        'Dataset': 'NHANES 34,719 (real clinical)',
        'Reported Accuracy': our_best['accuracy'],
        'Reported AUC': our_best['roc_auc'],
        'Imbalance': our_best['strategy'],
        'PR-AUC': our_best['pr_auc'],
        'Recall': our_best['recall'],
        'Brier': our_best['brier'],
        'Note': 'Temporal validation, clinical data',
    }
    rows = PUBLISHED_RESULTS + [our_row]
    t = pd.DataFrame(rows)
    t.to_csv(f'{RESULTS_DIR}/table10_vs_published.csv', index=False)
    return t
