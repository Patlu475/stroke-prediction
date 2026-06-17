"""Metric computation and clinical baseline scoring."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, roc_curve,
)


def compute_metrics(y_true, y_pred, y_prob):
    """Compute the full metric suite reported in the paper."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    idx_90 = np.argmin(np.abs((1 - fpr) - 0.90))

    return {
        'accuracy':       accuracy_score(y_true, y_pred),
        'f1':             f1_score(y_true, y_pred, zero_division=0),
        'precision':      precision_score(y_true, y_pred, zero_division=0),
        'recall':         recall_score(y_true, y_pred, zero_division=0),
        'specificity':    tn / (tn + fp) if (tn + fp) else 0,
        'roc_auc':        roc_auc_score(y_true, y_prob),
        'pr_auc':         average_precision_score(y_true, y_prob),
        'brier':          brier_score_loss(y_true, y_prob),
        'sens_at_90spec': tpr[idx_90],
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
    }


def framingham_score(df):
    """Simplified Framingham-like stroke risk score.

    Uses the same risk factors as the Framingham Stroke Risk Profile:
    age, SBP, diabetes, hypertension, cardiovascular disease history.
    Returns scores normalized to [0, 1].
    """
    s = np.zeros(len(df))

    age = df['RIDAGEYR'].values
    s += np.where(age >= 65, 4, np.where(age >= 55, 3,
         np.where(age >= 45, 2, np.where(age >= 35, 1, 0))))

    sbp = df['AVG_BPXSY'].fillna(120).values
    s += np.where(sbp >= 160, 3, np.where(sbp >= 140, 2,
         np.where(sbp >= 130, 1, 0)))

    s += df['DIABETES'].fillna(0).values * 2
    s += df['HYPERTENSION'].fillna(0).values
    s += df['HEART_DISEASE_CHD'].fillna(0).values * 2
    s += df['HEART_FAILURE'].fillna(0).values * 2
    s += df['HEART_ATTACK'].fillna(0).values

    return (s - s.min()) / (s.max() - s.min() + 1e-8)
