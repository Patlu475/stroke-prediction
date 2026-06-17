"""Fairness analysis — disaggregate model performance by demographic subgroups."""

import pandas as pd

from config import FAIRNESS_GROUPS
from src.evaluation import compute_metrics


def run_fairness_analysis(model, X_test, y_test, test_df):
    """Evaluate model performance per subgroup defined in config.FAIRNESS_GROUPS.

    Returns a DataFrame with one row per subgroup, containing all metrics
    plus group size and stroke count.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    rows = []

    for dimension, column, mapping in FAIRNESS_GROUPS:
        for value, label in mapping.items():
            mask = test_df[column].values == value
            n = mask.sum()
            n_stroke = int(y_test[mask].sum())

            if n < 10 or n_stroke < 2:
                continue

            m = compute_metrics(y_test[mask], y_pred[mask], y_prob[mask])
            m.update({
                'dimension': dimension,
                'group': label,
                'n': n,
                'n_stroke': n_stroke,
            })
            rows.append(m)

    return pd.DataFrame(rows)
