#!/usr/bin/env python3
"""
Brain Stroke Prediction: NHANES Reality Check
==============================================
Entry point. Orchestrates data loading, experiments, analysis, and output.

Usage:
    pip install -r requirements.txt
    python run.py
"""

import os, sys, warnings
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from config import FEATURE_SETS, STRATEGIES, RESULTS_DIR
from src.data import load_data, prepare_features, get_missing_rates
from src.models import train_and_evaluate, cross_validate
from src.evaluation import compute_metrics, framingham_score
from src.fairness import run_fairness_analysis
from src.tables import (
    table1_demographics, table2_dataset_comparison, table3_missing_data,
    table8_framingham, table10_vs_published,
)
from src.plots import (
    fig1_pr_auc_bars, fig2_roc_pr_curves, fig3_calibration,
    fig4_feature_importance, fig5_confusion_matrices, fig6_fairness_bars,
)

warnings.filterwarnings('ignore')
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    # ── 1. Load data ──
    print("Loading data...")
    train_df, test_df = load_data()
    pos_weight = (train_df['STROKE'] == 0).sum() / train_df['STROKE'].sum()

    print(f"Train (2007-2014): {len(train_df):,} rows, "
          f"{train_df['STROKE'].sum()} stroke ({train_df['STROKE'].mean()*100:.1f}%)")
    print(f"Test  (2015-2018): {len(test_df):,} rows, "
          f"{test_df['STROKE'].sum()} stroke ({test_df['STROKE'].mean()*100:.1f}%)")

    # ── 2. Run 18 experiments ──
    all_results = []
    all_cv = []
    best_model = None
    best_pr_auc = 0
    best_info = {}

    for fs_name, fs_cols in FEATURE_SETS.items():
        X_train, X_test, y_train, y_test, _, _ = prepare_features(train_df, test_df, fs_cols)

        missing = get_missing_rates(train_df, fs_cols)
        missing_cols = missing[missing > 0]

        print(f"\n{'='*60}")
        print(f"FEATURE SET: {fs_name} ({len(fs_cols)} features)")
        print(f"{'='*60}")
        if len(missing_cols):
            print("Missing data (train, before imputation):")
            for col, pct in missing_cols.items():
                print(f"  {col}: {pct:.1f}%")

        for strategy in STRATEGIES:
            print(f"\n  Strategy: {strategy}")
            print(f"  {'-'*50}")

            cv = cross_validate(X_train, y_train, strategy, pos_weight)
            for mname, scores in cv.items():
                all_cv.append({
                    'features': fs_name, 'strategy': strategy, 'model': mname,
                    'cv_pr_auc_mean': scores['mean_pr_auc'],
                    'cv_pr_auc_std': scores['std_pr_auc'],
                })

            results, models = train_and_evaluate(
                X_train, y_train, X_test, y_test, strategy, pos_weight, compute_metrics)

            for r in results:
                r['features'] = fs_name
                r['strategy'] = strategy
                all_results.append(r)

                mname = r['model']
                cv_info = cv[mname]
                print(f"    {mname:8s}  "
                      f"CV={cv_info['mean_pr_auc']:.3f}+/-{cv_info['std_pr_auc']:.3f}  "
                      f"Test PR-AUC={r['pr_auc']:.3f}  ROC={r['roc_auc']:.3f}  "
                      f"F1={r['f1']:.3f}  Recall={r['recall']:.3f}  "
                      f"TP={r['tp']} FN={r['fn']}")

                if r['pr_auc'] > best_pr_auc:
                    best_pr_auc = r['pr_auc']
                    best_model = models[mname]
                    best_info = {
                        'X_test': X_test, 'y_test': y_test,
                        'model_name': mname, 'fs_name': fs_name,
                        'strategy': strategy, 'fs_cols': fs_cols,
                        'brier': r['brier'],
                    }

    # ── 3. Save experiment results ──
    res = pd.DataFrame(all_results)
    col_order = ['features', 'strategy', 'model', 'accuracy', 'recall', 'precision',
                 'f1', 'specificity', 'roc_auc', 'pr_auc', 'brier',
                 'sens_at_90spec', 'tp', 'fp', 'tn', 'fn']
    res = res[[c for c in col_order if c in res.columns]]
    res.to_csv(f'{RESULTS_DIR}/main_results.csv', index=False)

    cv_df = pd.DataFrame(all_cv)
    cv_df.to_csv(f'{RESULTS_DIR}/cv_results.csv', index=False)

    print(f"\n{'='*60}")
    print("ALL RESULTS — TEMPORAL TEST SET (sorted by PR-AUC)")
    print(f"{'='*60}")
    print(res.sort_values('pr_auc', ascending=False).to_string(index=False, float_format='%.3f'))

    # ── 4. Framingham baseline ──
    print(f"\n{'='*60}")
    print("FRAMINGHAM-STYLE CLINICAL BASELINE")
    print(f"{'='*60}")
    f_scores = framingham_score(test_df)
    y_te = test_df['STROKE'].values
    f_roc = roc_auc_score(y_te, f_scores)
    f_pr = average_precision_score(y_te, f_scores)
    best_row = res.loc[res['pr_auc'].idxmax()]
    print(f"  Framingham:  ROC-AUC={f_roc:.3f}, PR-AUC={f_pr:.3f}")
    print(f"  Best ML:     ROC-AUC={best_row['roc_auc']:.3f}, PR-AUC={best_row['pr_auc']:.3f} "
          f"({best_row['model']}, {best_row['features']}, {best_row['strategy']})")

    # ── 5. Fairness analysis ──
    print(f"\n{'='*60}")
    print(f"FAIRNESS ANALYSIS — {best_info['model_name']} "
          f"({best_info['fs_name']}, {best_info['strategy']})")
    print(f"{'='*60}")
    fair_df = run_fairness_analysis(best_model, best_info['X_test'], best_info['y_test'], test_df)
    if not fair_df.empty:
        show = ['dimension', 'group', 'n', 'n_stroke', 'recall',
                'specificity', 'pr_auc', 'roc_auc', 'brier']
        show = [c for c in show if c in fair_df.columns]
        print(fair_df[show].to_string(index=False, float_format='%.3f'))
        fair_df.to_csv(f'{RESULTS_DIR}/fairness_results.csv', index=False)

    # ── 6. Feature importance ──
    print(f"\n{'='*60}")
    print(f"FEATURE IMPORTANCE — {best_info['model_name']}")
    print(f"{'='*60}")
    fi = fig4_feature_importance(best_model, best_info['fs_cols'])
    if fi is not None:
        display_col = 'coefficient' if 'coefficient' in fi.columns else 'importance'
        print(fi[['feature', display_col]].to_string(index=False, float_format='%.4f'))

    # ── 7. Paper tables ──
    print(f"\n{'='*60}")
    print("GENERATING PAPER TABLES")
    print(f"{'='*60}")

    table1_demographics(train_df, test_df)
    print("  Table 1: table1_demographics.csv")

    table2_dataset_comparison()
    print("  Table 2: table2_dataset_comparison.csv")

    table3_missing_data(train_df)
    print("  Table 3: table3_missing_data.csv")

    t8 = table8_framingham(test_df, res)
    print("  Table 8: table8_framingham_comparison.csv")
    print(t8.to_string(index=False, float_format='%.3f'))

    table10_vs_published(res)
    print("  Table 10: table10_vs_published.csv")

    # ── 8. Figures ──
    print(f"\n{'='*60}")
    print("GENERATING FIGURES")
    print(f"{'='*60}")

    fig1_pr_auc_bars(res)
    print("  Fig 1: fig1_pr_auc_comparison.png")

    fig2_roc_pr_curves(train_df, test_df, best_info['fs_name'], best_info['strategy'],
                       pos_weight, f_roc, f_pr, f_scores)
    print("  Fig 2: fig2_roc_pr_curves.png")

    fig3_calibration(best_model, best_info['X_test'], best_info['y_test'],
                     best_info['model_name'], best_info['brier'])
    print("  Fig 3: fig3_calibration_curve.png")

    if fi is not None:
        print("  Fig 4: fig4_feature_importance.png")

    fig5_confusion_matrices(train_df, test_df)
    print("  Fig 5: fig5_confusion_matrices.png")

    fig6_fairness_bars(fair_df)
    print("  Fig 6: fig6_fairness_bars.png")

    # ── 9. Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    print("\n1. ACCURACY TRAP (no balancing):")
    for _, r in res[res['strategy'] == 'none'].iterrows():
        print(f"   {r['features']:15s} {r['model']:8s}: "
              f"Accuracy={r['accuracy']:.3f}  Recall={r['recall']:.3f}  "
              f"TP={r['tp']}  FN={r['fn']}")

    print(f"\n2. BEST MODEL: {best_row['model']} + {best_row['features']} + {best_row['strategy']}")
    print(f"   PR-AUC={best_row['pr_auc']:.3f}  ROC-AUC={best_row['roc_auc']:.3f}  "
          f"F1={best_row['f1']:.3f}  Brier={best_row['brier']:.3f}")

    print("\n3. KAGGLE-EQUIV vs FULL CLINICAL:")
    for fs in ['kaggle_equiv', 'full_clinical']:
        b = res[res['features'] == fs].sort_values('pr_auc', ascending=False).iloc[0]
        print(f"   {fs:15s}: PR-AUC={b['pr_auc']:.3f}  ROC={b['roc_auc']:.3f}  "
              f"({b['model']}, {b['strategy']})")

    print(f"\n4. CLINICAL BASELINE:")
    print(f"   Framingham PR-AUC={f_pr:.3f}  vs  Best ML PR-AUC={best_row['pr_auc']:.3f}")

    print(f"\nAll outputs saved to {RESULTS_DIR}/")
    print("Done!")


if __name__ == '__main__':
    main()
