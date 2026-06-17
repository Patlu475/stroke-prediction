"""All figure generators. Each function saves a PNG and returns nothing."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
)
from sklearn.calibration import calibration_curve
from imblearn.over_sampling import SMOTE

from config import (
    RANDOM_STATE, RESULTS_DIR, FEATURE_SETS,
    FULL_CLINICAL_FEATURES,
)
from src.models import get_models


def fig1_pr_auc_bars(res_df):
    """Fig 1: PR-AUC comparison bar chart — Kaggle-equiv vs full clinical."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for i, fs in enumerate(['kaggle_equiv', 'full_clinical']):
        sub = res_df[res_df['features'] == fs]
        piv = sub.pivot(index='model', columns='strategy', values='pr_auc')
        piv.plot(kind='bar', ax=axes[i], rot=0, colormap='Set2')
        axes[i].set_title(f'PR-AUC by Strategy — {fs}')
        axes[i].set_ylabel('PR-AUC (higher = better)')
        axes[i].set_ylim(0, max(0.25, piv.max().max() * 1.3))
        axes[i].legend(title='Strategy', fontsize=8)
    plt.tight_layout()
    fig.savefig(f'{RESULTS_DIR}/fig1_pr_auc_comparison.png', dpi=150)
    plt.close()


def fig2_roc_pr_curves(train_df, test_df, best_fs, best_strat, pos_weight,
                        fram_roc, fram_pr, f_scores):
    """Fig 2: ROC + Precision-Recall curves with Framingham baseline."""
    fs_cols = FEATURE_SETS[best_fs]
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(imputer.fit_transform(train_df[fs_cols]))
    Xte = scaler.transform(imputer.transform(test_df[fs_cols]))
    ytr, yte = train_df['STROKE'].values, test_df['STROKE'].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    models = get_models(best_strat, pos_weight)

    X_fit, y_fit = Xtr, ytr
    if best_strat == 'smote':
        X_fit, y_fit = SMOTE(random_state=RANDOM_STATE).fit_resample(Xtr, ytr)

    for name, model in models.items():
        model.fit(X_fit, y_fit)
        yp = model.predict_proba(Xte)[:, 1]

        fpr, tpr, _ = roc_curve(yte, yp)
        ax1.plot(fpr, tpr, label=f'{name} (AUC={roc_auc_score(yte, yp):.3f})')

        prec, rec, _ = precision_recall_curve(yte, yp)
        ax2.plot(rec, prec, label=f'{name} (AP={average_precision_score(yte, yp):.3f})')

    fpr_f, tpr_f, _ = roc_curve(yte, f_scores)
    ax1.plot(fpr_f, tpr_f, '--', color='gray', label=f'Framingham (AUC={fram_roc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k:', alpha=0.3)
    ax1.set_xlabel('False Positive Rate'); ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves'); ax1.legend(fontsize=9)

    prec_f, rec_f, _ = precision_recall_curve(yte, f_scores)
    ax2.plot(rec_f, prec_f, '--', color='gray', label=f'Framingham (AP={fram_pr:.3f})')
    ax2.axhline(y=yte.mean(), color='k', linestyle=':', alpha=0.3,
                label=f'Prevalence ({yte.mean():.3f})')
    ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curves'); ax2.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(f'{RESULTS_DIR}/fig2_roc_pr_curves.png', dpi=150)
    plt.close()


def fig3_calibration(model, X_test, y_test, model_name, brier):
    """Fig 3: Calibration curve for the best model."""
    y_prob = model.predict_proba(X_test)[:, 1]
    frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(mean_pred, frac_pos, 's-', label=f'{model_name} (Brier={brier:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.set_xlabel('Mean Predicted Probability'); ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Curve — Best Model'); ax.legend()
    plt.tight_layout()
    fig.savefig(f'{RESULTS_DIR}/fig3_calibration_curve.png', dpi=150)
    plt.close()


def fig4_feature_importance(model, feature_names):
    """Fig 4: Feature importance — LR coefficients or tree-based importances."""
    if hasattr(model, 'coef_'):
        fi = pd.DataFrame({'feature': feature_names, 'coefficient': model.coef_[0]})
        fi['abs_coeff'] = fi['coefficient'].abs()
        fi = fi.sort_values('abs_coeff', ascending=False)
        fi.to_csv(f'{RESULTS_DIR}/feature_importance.csv', index=False)

        fig, ax = plt.subplots(figsize=(10, 7))
        colors = ['#e74c3c' if c > 0 else '#3498db' for c in fi['coefficient'].values]
        ax.barh(range(len(fi)), fi['coefficient'].values, color=colors)
        ax.set_yticks(range(len(fi)))
        ax.set_yticklabels(fi['feature'].values)
        ax.invert_yaxis()
        ax.set_title('LR Coefficients (Full Clinical, Cost-Sensitive)\n'
                      'Red = increases stroke risk, Blue = decreases')
        ax.set_xlabel('Standardized Coefficient')
        ax.axvline(x=0, color='black', linewidth=0.5)
    elif hasattr(model, 'feature_importances_'):
        fi = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_})
        fi = fi.sort_values('importance', ascending=False)
        fi.to_csv(f'{RESULTS_DIR}/feature_importance.csv', index=False)

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.barplot(data=fi.head(15), x='importance', y='feature', ax=ax, color='steelblue')
        ax.set_title(f'Top 15 Feature Importances')
        ax.set_xlabel('Importance')
    else:
        return None

    plt.tight_layout()
    fig.savefig(f'{RESULTS_DIR}/fig4_feature_importance.png', dpi=150)
    plt.close()
    return fi


def fig5_confusion_matrices(train_df, test_df):
    """Fig 5: Side-by-side confusion matrices — accuracy trap vs best model."""
    fs_cols = FULL_CLINICAL_FEATURES
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(imputer.fit_transform(train_df[fs_cols]))
    Xte = scaler.transform(imputer.transform(test_df[fs_cols]))
    ytr, yte = train_df['STROKE'].values, test_df['STROKE'].values

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    rf = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(Xtr, ytr)
    cm1 = confusion_matrix(yte, rf.predict(Xte))
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['No Stroke', 'Stroke'], yticklabels=['No Stroke', 'Stroke'])
    axes[0].set_title('RF, No Balancing (Accuracy Trap)\nAccuracy=95.7%, Recall=0.0%')
    axes[0].set_ylabel('Actual'); axes[0].set_xlabel('Predicted')

    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE)
    lr.fit(Xtr, ytr)
    cm2 = confusion_matrix(yte, lr.predict(Xte))
    sns.heatmap(cm2, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
                xticklabels=['No Stroke', 'Stroke'], yticklabels=['No Stroke', 'Stroke'])
    axes[1].set_title('LR, Cost-Sensitive (Best Model)\nRecall=67.4%, Precision=12.0%')
    axes[1].set_ylabel('Actual'); axes[1].set_xlabel('Predicted')

    plt.tight_layout()
    fig.savefig(f'{RESULTS_DIR}/fig5_confusion_matrices.png', dpi=150)
    plt.close()


def fig6_fairness_bars(fair_df):
    """Fig 6: Grouped bar chart of fairness metrics by subgroup."""
    if fair_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for i, (dim, title) in enumerate([('Sex', 'By Sex'), ('Race/Ethnicity', 'By Race/Ethnicity')]):
        sub = fair_df[fair_df['dimension'] == dim].copy()
        if sub.empty:
            continue
        plot_data = sub.melt(
            id_vars=['group'], value_vars=['recall', 'specificity', 'pr_auc'],
            var_name='Metric', value_name='Value')
        sns.barplot(data=plot_data, x='group', y='Value', hue='Metric',
                    ax=axes[i], palette='Set2')
        axes[i].set_title(f'Fairness Analysis — {title}')
        axes[i].set_ylabel('Score'); axes[i].set_xlabel('')
        axes[i].tick_params(axis='x', rotation=25)
        axes[i].set_ylim(0, 1)
        axes[i].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(f'{RESULTS_DIR}/fig6_fairness_bars.png', dpi=150)
    plt.close()
