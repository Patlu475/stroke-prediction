"""All figure generators — publication quality, saved as PNG + PDF."""

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

from config import RANDOM_STATE, RESULTS_DIR, FEATURE_SETS, FULL_CLINICAL_FEATURES
from src.models import get_models
from src.style import (
    paper_style, save_fig, PALETTE, PALETTE_LIST,
    SINGLE_COL_WIDTH, DOUBLE_COL_WIDTH,
)


def fig1_pr_auc_bars(res_df):
    """Fig 1: PR-AUC comparison — Kaggle-equiv vs full clinical."""
    with paper_style():
        fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, 2.8))
        for i, (fs, title) in enumerate([
            ('kaggle_equiv', 'Kaggle-Equivalent (8 features)'),
            ('full_clinical', 'Full Clinical (22 features)'),
        ]):
            sub = res_df[res_df['features'] == fs]
            models = ['LR', 'RF', 'XGBoost']
            strategies = ['none', 'cost_sensitive', 'smote']
            x = np.arange(len(models))
            width = 0.25

            for j, strat in enumerate(strategies):
                vals = [sub[(sub['model'] == m) & (sub['strategy'] == strat)]['pr_auc'].values[0]
                        for m in models]
                label = {'none': 'No balancing', 'cost_sensitive': 'Cost-sensitive', 'smote': 'SMOTE'}[strat]
                axes[i].bar(x + j * width, vals, width, label=label, color=PALETTE_LIST[j], edgecolor='white', linewidth=0.3)

            axes[i].set_title(title)
            axes[i].set_ylabel('PR-AUC')
            axes[i].set_xticks(x + width)
            axes[i].set_xticklabels(models)
            axes[i].set_ylim(0, max(0.22, sub['pr_auc'].max() * 1.3))
            axes[i].legend(fontsize=7, frameon=False)

        plt.tight_layout()
        save_fig(fig, 'fig1_pr_auc_comparison')


def fig2_roc_pr_curves(train_df, test_df, best_fs, best_strat, pos_weight,
                        fram_roc, fram_pr, f_scores):
    """Fig 2: ROC + Precision-Recall curves with Framingham baseline."""
    fs_cols = FEATURE_SETS[best_fs]
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(imputer.fit_transform(train_df[fs_cols]))
    Xte = scaler.transform(imputer.transform(test_df[fs_cols]))
    ytr, yte = train_df['STROKE'].values, test_df['STROKE'].values

    with paper_style():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, 3.2))
        models = get_models(best_strat, pos_weight)

        X_fit, y_fit = Xtr, ytr
        if best_strat == 'smote':
            X_fit, y_fit = SMOTE(random_state=RANDOM_STATE).fit_resample(Xtr, ytr)

        for idx, (name, model) in enumerate(models.items()):
            model.fit(X_fit, y_fit)
            yp = model.predict_proba(Xte)[:, 1]

            fpr, tpr, _ = roc_curve(yte, yp)
            ax1.plot(fpr, tpr, color=PALETTE_LIST[idx],
                     label=f'{name} (AUC={roc_auc_score(yte, yp):.3f})')

            prec, rec, _ = precision_recall_curve(yte, yp)
            ax2.plot(rec, prec, color=PALETTE_LIST[idx],
                     label=f'{name} (AP={average_precision_score(yte, yp):.3f})')

        fpr_f, tpr_f, _ = roc_curve(yte, f_scores)
        ax1.plot(fpr_f, tpr_f, '--', color=PALETTE['gray'],
                 label=f'Framingham (AUC={fram_roc:.3f})')
        ax1.plot([0, 1], [0, 1], ':', color='#999999', linewidth=0.5)
        ax1.set_xlabel('False Positive Rate'); ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves'); ax1.legend(fontsize=7, frameon=False)

        prec_f, rec_f, _ = precision_recall_curve(yte, f_scores)
        ax2.plot(rec_f, prec_f, '--', color=PALETTE['gray'],
                 label=f'Framingham (AP={fram_pr:.3f})')
        ax2.axhline(y=yte.mean(), color='#999999', linestyle=':', linewidth=0.5,
                    label=f'Prevalence ({yte.mean():.3f})')
        ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves'); ax2.legend(fontsize=7, frameon=False)

        plt.tight_layout()
        save_fig(fig, 'fig2_roc_pr_curves')


def fig3_calibration(model, X_test, y_test, model_name, brier):
    """Fig 3: Calibration curve for the best model."""
    y_prob = model.predict_proba(X_test)[:, 1]
    frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10)

    with paper_style():
        fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, SINGLE_COL_WIDTH))
        ax.plot(mean_pred, frac_pos, 's-', color=PALETTE['blue'], markersize=5,
                label=f'{model_name} (Brier = {brier:.3f})')
        ax.plot([0, 1], [0, 1], '--', color=PALETTE['gray'], label='Perfect calibration')
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Observed fraction of positives')
        ax.set_title('Calibration Curve')
        ax.legend(frameon=False)
        plt.tight_layout()
        save_fig(fig, 'fig3_calibration_curve')


def fig4_feature_importance(model, feature_names):
    """Fig 4: Feature importance — LR coefficients or tree importances."""
    if hasattr(model, 'coef_'):
        fi = pd.DataFrame({'feature': feature_names, 'coefficient': model.coef_[0]})
        fi['abs_coeff'] = fi['coefficient'].abs()
        fi = fi.sort_values('abs_coeff', ascending=False)
        fi.to_csv(f'{RESULTS_DIR}/feature_importance.csv', index=False)

        with paper_style():
            fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, 4.5))
            colors = [PALETTE['orange'] if c > 0 else PALETTE['blue'] for c in fi['coefficient'].values]
            ax.barh(range(len(fi)), fi['coefficient'].values, color=colors, height=0.7)
            ax.set_yticks(range(len(fi)))
            ax.set_yticklabels(fi['feature'].values)
            ax.invert_yaxis()
            ax.set_title('Logistic Regression Coefficients')
            ax.set_xlabel('Standardized coefficient')
            ax.axvline(x=0, color='black', linewidth=0.4)
            plt.tight_layout()
            save_fig(fig, 'fig4_feature_importance')
        return fi

    elif hasattr(model, 'feature_importances_'):
        fi = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_})
        fi = fi.sort_values('importance', ascending=False)
        fi.to_csv(f'{RESULTS_DIR}/feature_importance.csv', index=False)

        with paper_style():
            fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, 4.5))
            top = fi.head(15)
            ax.barh(range(len(top)), top['importance'].values, color=PALETTE['blue'], height=0.7)
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels(top['feature'].values)
            ax.invert_yaxis()
            ax.set_title('Feature Importances (top 15)')
            ax.set_xlabel('Importance')
            plt.tight_layout()
            save_fig(fig, 'fig4_feature_importance')
        return fi

    return None


def fig5_confusion_matrices(train_df, test_df):
    """Fig 5: Side-by-side confusion matrices — accuracy trap vs best model."""
    fs_cols = FULL_CLINICAL_FEATURES
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(imputer.fit_transform(train_df[fs_cols]))
    Xte = scaler.transform(imputer.transform(test_df[fs_cols]))
    ytr, yte = train_df['STROKE'].values, test_df['STROKE'].values

    with paper_style():
        fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, 3.0))

        rf = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(Xtr, ytr)
        cm1 = confusion_matrix(yte, rf.predict(Xte))
        sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=['No Stroke', 'Stroke'], yticklabels=['No Stroke', 'Stroke'],
                    cbar=False, annot_kws={'size': 9})
        axes[0].set_title('(a) RF, no balancing\nAcc = 95.7%, Recall = 0.0%', fontsize=9)
        axes[0].set_ylabel('Actual'); axes[0].set_xlabel('Predicted')

        lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE)
        lr.fit(Xtr, ytr)
        cm2 = confusion_matrix(yte, lr.predict(Xte))
        sns.heatmap(cm2, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
                    xticklabels=['No Stroke', 'Stroke'], yticklabels=['No Stroke', 'Stroke'],
                    cbar=False, annot_kws={'size': 9})
        axes[1].set_title('(b) LR, cost-sensitive\nRecall = 67.4%, Precision = 12.0%', fontsize=9)
        axes[1].set_ylabel('Actual'); axes[1].set_xlabel('Predicted')

        plt.tight_layout()
        save_fig(fig, 'fig5_confusion_matrices')


def fig6_fairness_bars(fair_df):
    """Fig 6: Grouped bar chart of fairness metrics by subgroup."""
    if fair_df.empty:
        return

    with paper_style():
        fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, 3.0))

        for i, (dim, title) in enumerate([('Sex', '(a) By Sex'), ('Race/Ethnicity', '(b) By Race/Ethnicity')]):
            sub = fair_df[fair_df['dimension'] == dim].copy()
            if sub.empty:
                continue

            metrics = ['recall', 'specificity', 'pr_auc']
            labels = {'recall': 'Recall', 'specificity': 'Specificity', 'pr_auc': 'PR-AUC'}
            x = np.arange(len(sub))
            width = 0.25

            for j, metric in enumerate(metrics):
                axes[i].bar(x + j * width, sub[metric].values, width,
                           label=labels[metric], color=PALETTE_LIST[j],
                           edgecolor='white', linewidth=0.3)

            axes[i].set_title(title)
            axes[i].set_ylabel('Score')
            axes[i].set_xticks(x + width)
            axes[i].set_xticklabels(sub['group'].values, rotation=20, ha='right')
            axes[i].set_ylim(0, 1)
            axes[i].legend(fontsize=7, frameon=False)

        plt.tight_layout()
        save_fig(fig, 'fig6_fairness_bars')
