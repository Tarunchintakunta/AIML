#!/usr/bin/env python3
"""
main.py - Full Pipeline for SIEM Alert Triage with Cost-Sensitive ML
=====================================================================
Project 9: ML for SIEM Alert Triage
Student: Yogita

Pipeline steps:
    1. Generate synthetic SIEM alert data (CIC-IDS2017 / NSL-KDD / UNSW-NB15)
    2. Train individual models (RF, XGBoost, Isolation Forest)
    3. Build stacking ensemble with LR meta-learner
    4. Evaluate all models (F-beta, cost, alert reduction)
    5. Compute SHAP explanations
    6. Generate all visualisations

Usage:
    python main.py --synthetic
    python main.py --synthetic --n_samples 30000
"""

import argparse
import os
import sys
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Ensure src/ is on the path regardless of invocation directory
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from data_loader import (
    generate_synthetic_siem_data,
    prepare_data,
    FEATURE_NAMES,
    CLASS_NAMES,
)
from model import (
    build_random_forest,
    build_xgboost,
    build_isolation_forest,
    build_stacking_ensemble,
    augment_with_anomaly_scores,
    train_model,
    evaluate_model,
    print_results,
    compute_baseline_cost,
)
from visualize import (
    plot_model_comparison,
    plot_confusion_matrices,
    plot_alert_reduction,
    plot_cost_analysis,
    plot_shap_importance,
    plot_summary_figure,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ML for SIEM Alert Triage - Project 9 (Yogita)",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data (required for reproducible demo)",
    )
    parser.add_argument(
        "--n_samples", type=int, default=20000,
        help="Number of synthetic samples to generate (default: 20000)",
    )
    parser.add_argument(
        "--random_state", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=os.path.join(PROJECT_DIR, "outputs"),
        help="Directory for output artefacts",
    )
    parser.add_argument(
        "--skip_shap", action="store_true",
        help="Skip SHAP analysis (faster execution)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("  Project 9: ML for SIEM Alert Triage")
    print("  Student: Yogita")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Data Generation
    # ------------------------------------------------------------------
    print("\n[Step 1/6] Generating synthetic SIEM alert data ...")
    t0 = time.time()

    if not args.synthetic:
        print("  NOTE: --synthetic flag not set. Using synthetic data anyway")
        print("        (real datasets require separate download).")

    df = generate_synthetic_siem_data(
        n_samples=args.n_samples,
        random_state=args.random_state,
        dataset_name="combined",
    )
    print(f"  Generated {len(df):,} samples in {time.time()-t0:.1f}s")
    print(f"  Class distribution:")
    for label, name in enumerate(CLASS_NAMES):
        count = (df["label"] == label).sum()
        print(f"    {name:20s}: {count:6,} ({count/len(df):6.1%})")
    print(f"  Dataset sources: {df['source_dataset'].value_counts().to_dict()}")

    # Save raw data
    csv_path = os.path.join(args.output_dir, "synthetic_siem_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"  [saved] {csv_path}")

    # ------------------------------------------------------------------
    # Step 2: Data Preparation
    # ------------------------------------------------------------------
    print("\n[Step 2/6] Preparing train/val/test splits ...")
    data = prepare_data(df, test_size=0.2, val_size=0.1,
                        random_state=args.random_state)
    print(f"  Train: {data['X_train'].shape[0]:,}  "
          f"Val: {data['X_val'].shape[0]:,}  "
          f"Test: {data['X_test'].shape[0]:,}")

    # Augment with Isolation Forest anomaly scores
    print("  Fitting Isolation Forest for anomaly scores ...")
    iso_forest = build_isolation_forest(args.random_state)
    X_train_aug, X_val_aug, X_test_aug = augment_with_anomaly_scores(
        iso_forest, data["X_train"], data["X_val"], data["X_test"],
    )
    feature_names_aug = list(data["feature_names"]) + ["anomaly_score"]
    print(f"  Augmented features: {X_train_aug.shape[1]} "
          f"(+1 anomaly score)")

    # ------------------------------------------------------------------
    # Step 3: Train Individual Models
    # ------------------------------------------------------------------
    print("\n[Step 3/6] Training individual models ...")

    # Random Forest
    print("  Training Random Forest (cost-sensitive) ...")
    t0 = time.time()
    rf = build_random_forest(args.random_state)
    rf = train_model(rf, X_train_aug, data["y_train"], "rf")
    print(f"    done in {time.time()-t0:.1f}s")

    # XGBoost
    print("  Training XGBoost (cost-sensitive) ...")
    t0 = time.time()
    xgb = build_xgboost(args.random_state)
    xgb = train_model(xgb, X_train_aug, data["y_train"], "xgb")
    print(f"    done in {time.time()-t0:.1f}s")

    # Stacking Ensemble
    print("  Training Stacking Ensemble (RF + XGBoost -> LR) ...")
    t0 = time.time()
    stacking = build_stacking_ensemble(args.random_state)
    stacking = train_model(stacking, X_train_aug, data["y_train"], "stacking")
    print(f"    done in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # Step 4: Evaluate All Models
    # ------------------------------------------------------------------
    print("\n[Step 4/6] Evaluating models on test set ...")

    results_rf = evaluate_model(rf, X_test_aug, data["y_test"],
                                "Random Forest")
    results_xgb = evaluate_model(xgb, X_test_aug, data["y_test"],
                                 "XGBoost")
    results_stacking = evaluate_model(stacking, X_test_aug, data["y_test"],
                                      "Stacking Ensemble")

    all_results = [results_rf, results_xgb, results_stacking]

    for r in all_results:
        print_results(r)

    baseline_cost = compute_baseline_cost(data["y_test"])
    print(f"  Baseline cost (all escalated): {baseline_cost:,.0f}")

    # Summary table
    print(f"\n{'Model':<22} {'F-beta(2)':>10} {'Cost':>10} {'Alert Red.':>12}")
    print("-" * 56)
    for r in all_results:
        print(f"  {r['model_name']:<20} {r['fbeta_macro']:>10.4f} "
              f"{r['total_cost']:>10.0f} {r['alert_reduction_rate']:>11.1%}")

    # ------------------------------------------------------------------
    # Step 5: SHAP Explanations
    # ------------------------------------------------------------------
    shap_values = None
    if not args.skip_shap:
        print("\n[Step 5/6] Computing SHAP explanations ...")
        try:
            import shap
            t0 = time.time()

            # Use a subsample for speed
            n_shap = min(500, X_test_aug.shape[0])
            X_shap = X_test_aug[:n_shap]

            # SHAP TreeExplainer for XGBoost
            explainer = shap.TreeExplainer(xgb)
            shap_values = explainer.shap_values(X_shap)
            print(f"  SHAP computed for {n_shap} samples in "
                  f"{time.time()-t0:.1f}s")

            # Also get SHAP for stacking's XGBoost component if possible
        except ImportError:
            print("  WARNING: shap not installed. Skipping SHAP analysis.")
        except Exception as e:
            print(f"  WARNING: SHAP failed with: {e}")
            print("  Continuing without SHAP visualisations.")
    else:
        print("\n[Step 5/6] SHAP analysis skipped (--skip_shap flag).")

    # ------------------------------------------------------------------
    # Step 6: Visualisations
    # ------------------------------------------------------------------
    print("\n[Step 6/6] Generating visualisations ...")

    plot_model_comparison(
        all_results,
        save_path=os.path.join(args.output_dir, "model_comparison.png"),
    )
    plot_confusion_matrices(
        all_results,
        save_path=os.path.join(args.output_dir, "confusion_matrices.png"),
    )
    plot_alert_reduction(
        all_results,
        save_path=os.path.join(args.output_dir, "alert_reduction.png"),
    )
    plot_cost_analysis(
        all_results,
        baseline_cost=baseline_cost,
        save_path=os.path.join(args.output_dir, "cost_analysis.png"),
    )

    if shap_values is not None:
        plot_shap_importance(
            shap_values,
            feature_names_aug,
            class_idx=2,
            top_k=15,
            save_path=os.path.join(args.output_dir, "shap_importance.png"),
        )
        plot_summary_figure(
            all_results,
            shap_values,
            feature_names_aug,
            baseline_cost=baseline_cost,
            save_path=os.path.join(args.output_dir, "summary_figure.png"),
        )
    else:
        print("  Skipping SHAP-dependent plots.")

    # ------------------------------------------------------------------
    # Final Summary
    # ------------------------------------------------------------------
    best = max(all_results, key=lambda r: r["fbeta_macro"])
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Best model: {best['model_name']}")
    print(f"    F-beta(2) macro:   {best['fbeta_macro']:.4f}")
    print(f"    Total cost:        {best['total_cost']:,.0f} "
          f"(baseline: {baseline_cost:,.0f})")
    print(f"    Alert reduction:   {best['alert_reduction_rate']:.1%}")
    print(f"    Cost savings:      "
          f"{(1 - best['total_cost']/baseline_cost)*100:.1f}% "
          f"vs. baseline")
    print(f"\n  All outputs saved to: {args.output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
