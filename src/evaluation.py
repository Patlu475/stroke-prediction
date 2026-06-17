"""Metric computation, clinical baseline scoring, and statistical tests."""

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, roc_curve,
)

from config import RANDOM_STATE


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


def bootstrap_metrics(y_true, y_pred, y_prob, n_iterations=1000, ci=0.95):
    """Compute bootstrap 95% confidence intervals for key metrics.

    Returns dict of {metric_name: (point_estimate, ci_lower, ci_upper)}.
    """
    rng = np.random.RandomState(RANDOM_STATE)
    n = len(y_true)
    alpha = (1 - ci) / 2

    boot_results = {k: [] for k in ['roc_auc', 'pr_auc', 'f1', 'recall',
                                      'precision', 'brier', 'accuracy']}

    for _ in range(n_iterations):
        idx = rng.choice(n, size=n, replace=True)
        yb_true = y_true[idx]
        yb_pred = y_pred[idx]
        yb_prob = y_prob[idx]

        if len(np.unique(yb_true)) < 2:
            continue

        boot_results['roc_auc'].append(roc_auc_score(yb_true, yb_prob))
        boot_results['pr_auc'].append(average_precision_score(yb_true, yb_prob))
        boot_results['f1'].append(f1_score(yb_true, yb_pred, zero_division=0))
        boot_results['recall'].append(recall_score(yb_true, yb_pred, zero_division=0))
        boot_results['precision'].append(precision_score(yb_true, yb_pred, zero_division=0))
        boot_results['brier'].append(brier_score_loss(yb_true, yb_prob))
        boot_results['accuracy'].append(accuracy_score(yb_true, yb_pred))

    point = compute_metrics(y_true, y_pred, y_prob)
    result = {}
    for metric in boot_results:
        values = np.array(boot_results[metric])
        if len(values) == 0:
            result[metric] = (point[metric], np.nan, np.nan)
        else:
            result[metric] = (
                point[metric],
                float(np.percentile(values, alpha * 100)),
                float(np.percentile(values, (1 - alpha) * 100)),
            )
    return result


# ── DeLong test for comparing two ROC-AUCs ──

def _compute_midrank(x):
    """Compute midranks for DeLong test."""
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    rank = np.zeros(n)
    i = 0
    while i < n:
        j_start = i
        while i < n - 1 and z[i] == z[i + 1]:
            i += 1
        midrank = (j_start + i) / 2.0
        for k in range(j_start, i + 1):
            rank[k] = midrank
        i += 1
    result = np.zeros(n)
    result[j] = rank
    return result


def _fast_delong(predictions_sorted_transposed, label_1_count):
    """Core DeLong AUC computation."""
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]

    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)

    for r in range(k):
        midrank = _compute_midrank(predictions_sorted_transposed[r, :])
        tx[r, :] = midrank[:m]
        ty[r, :] = midrank[m:]

    aucs = (tx.sum(axis=1) / m - (m + 1.0) / 2.0) / n

    v01 = (ty - tx.mean(axis=1, keepdims=True)) / n
    v10 = 1.0 - (tx - ty.mean(axis=1, keepdims=True)) / m

    sx = np.cov(v10) if m > 1 else np.zeros((k, k))
    sy = np.cov(v01) if n > 1 else np.zeros((k, k))

    if isinstance(sx, np.floating):
        sx = np.array([[sx]])
    if isinstance(sy, np.floating):
        sy = np.array([[sy]])

    delongcov = sx / m + sy / n

    return aucs, delongcov


def delong_test(y_true, y_prob_a, y_prob_b):
    """Two-sided DeLong test comparing two ROC-AUCs.

    Returns (auc_a, auc_b, z_statistic, p_value).
    """
    order = (-y_true).argsort()
    label_1_count = int(y_true.sum())

    predictions = np.vstack([y_prob_a, y_prob_b])
    predictions_sorted = predictions[:, order]

    aucs, delongcov = _fast_delong(predictions_sorted, label_1_count)

    diff = aucs[0] - aucs[1]
    var = delongcov[0, 0] + delongcov[1, 1] - 2 * delongcov[0, 1]

    if var <= 0:
        return float(aucs[0]), float(aucs[1]), 0.0, 1.0

    z = diff / np.sqrt(var)
    p = 2 * stats.norm.sf(abs(z))

    return float(aucs[0]), float(aucs[1]), float(z), float(p)


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
