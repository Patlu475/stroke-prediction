"""Model factory, training, and cross-validation."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from imblearn.over_sampling import SMOTE

from config import RANDOM_STATE, MODEL_PARAMS, CV_FOLDS


def get_models(strategy, pos_weight=1.0):
    """Return a dict of fresh model instances for the given imbalance strategy."""
    lr_params = {**MODEL_PARAMS['LR'], 'random_state': RANDOM_STATE}
    rf_params = {**MODEL_PARAMS['RF'], 'random_state': RANDOM_STATE}
    xgb_params = {**MODEL_PARAMS['XGBoost'], 'random_state': RANDOM_STATE}

    if strategy == 'cost_sensitive':
        lr_params['class_weight'] = 'balanced'
        rf_params['class_weight'] = 'balanced'
        xgb_params['scale_pos_weight'] = pos_weight

    return {
        'LR': LogisticRegression(**lr_params),
        'RF': RandomForestClassifier(**rf_params),
        'XGBoost': XGBClassifier(**xgb_params),
    }


def train_and_evaluate(X_train, y_train, X_test, y_test, strategy, pos_weight, compute_metrics_fn):
    """Train all models for a given strategy, evaluate on the test set.

    For SMOTE: applied to the full training set before fitting.
    The test set is never touched — temporal separation prevents leakage.

    Returns (list of metric dicts, dict of fitted models).
    """
    models = get_models(strategy, pos_weight)

    if strategy == 'smote':
        X_fit, y_fit = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train, y_train)
    else:
        X_fit, y_fit = X_train, y_train

    results = []
    for name, model in models.items():
        model.fit(X_fit, y_fit)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        m = compute_metrics_fn(y_test, y_pred, y_prob)
        m['model'] = name
        results.append(m)

    return results, models


def cross_validate(X_train, y_train, strategy, pos_weight):
    """5-fold stratified CV on the training set.

    SMOTE is applied INSIDE each fold to prevent leakage.
    Returns dict of {model_name: {mean_pr_auc, std_pr_auc}}.
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}

    for model_name in get_models(strategy, pos_weight):
        fold_scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model = get_models(strategy, pos_weight)[model_name]

            if strategy == 'smote':
                X_tr, y_tr = SMOTE(random_state=RANDOM_STATE).fit_resample(X_tr, y_tr)

            model.fit(X_tr, y_tr)
            y_prob = model.predict_proba(X_val)[:, 1]
            fold_scores.append(average_precision_score(y_val, y_prob))

        cv_results[model_name] = {
            'mean_pr_auc': np.mean(fold_scores),
            'std_pr_auc': np.std(fold_scores),
        }

    return cv_results
