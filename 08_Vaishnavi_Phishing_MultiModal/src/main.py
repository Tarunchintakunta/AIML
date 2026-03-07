"""
main.py - Full Pipeline for Multi-Modal Phishing Detection with XAI
====================================================================
Project 8: Explainable Multi-Modal Phishing Detection
Student: Vaishnavi Purohit (24260339)

Usage:
    python main.py
    python main.py --lime_instances 100 --shap_samples 300

Pipeline:
    1. Download / load real UCI Phishing Websites dataset (11,055 samples)
    2. Train single-modality baselines (RF per modality)
    3. Train late-fusion models (RF, XGBoost, LightGBM, ProbFusion)
    4. Run XAI pipeline (SHAP + LIME + consistency)
    5. Generate all visualizations
    6. Print summary report
"""

import argparse
import os
import sys
import time
import json
import numpy as np
import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, PROJECT_ROOT)

from data_loader import load_uci_phishing, prepare_splits, MODALITY_RANGES
from model import train_all_models, print_metrics
from explainers import run_xai_pipeline
from visualize import generate_all_plots


def parse_args():
    parser = argparse.ArgumentParser(
        description="Explainable Multi-Modal Phishing Detection Pipeline"
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Directory for dataset cache (default: <project>/data/phishing)"
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2,
        help="Test set fraction (default: 0.2)"
    )
    parser.add_argument(
        "--lime_instances", type=int, default=50,
        help="Number of instances for LIME batch explanations (default: 50)"
    )
    parser.add_argument(
        "--shap_samples", type=int, default=200,
        help="Number of test samples for SHAP computation (default: 200)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for plots and results"
    )
    parser.add_argument(
        "--random_state", type=int, default=42,
        help="Random seed (default: 42)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(args.output_dir, exist_ok=True)

    t_start = time.time()

    print("=" * 70)
    print("  EXPLAINABLE MULTI-MODAL PHISHING DETECTION")
    print("  Student: Vaishnavi Purohit (24260339)")
    print("  Dataset: UCI Phishing Websites (11,055 samples, 30 features)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Data - Download and load real UCI Phishing dataset
    # ------------------------------------------------------------------
    print("\n[Step 1] Loading UCI Phishing Websites dataset...")
    data = load_uci_phishing(data_dir=args.data_dir)

    splits = prepare_splits(
        data, test_size=args.test_size,
        random_state=args.random_state, scale=False
    )

    # Save dataset summary
    data["df"].to_csv(os.path.join(args.output_dir, "dataset.csv"),
                      index=False)
    print(f"  Dataset saved to {args.output_dir}/dataset.csv")

    # ------------------------------------------------------------------
    # Step 2 & 3: Train models
    # ------------------------------------------------------------------
    print("\n[Step 2-3] Training single-modality and fusion models...")
    results = train_all_models(splits, random_state=args.random_state)

    all_metrics = results["all_metrics"]

    # Save metrics summary
    metrics_df = pd.DataFrame([
        {k: v for k, v in m.items() if k != "confusion_matrix"}
        for m in all_metrics
    ])
    metrics_df.to_csv(os.path.join(args.output_dir, "metrics.csv"),
                      index=False)
    print(f"\n  Metrics saved to {args.output_dir}/metrics.csv")

    # ------------------------------------------------------------------
    # Step 4: XAI Pipeline (use best fusion model: XGBoost)
    # ------------------------------------------------------------------
    print("\n[Step 4] Running XAI pipeline on LateFusion-XGB model...")
    best_model = results["fusion_models"]["xgb"]
    feature_names = data["feature_names_all"]

    xai_results = run_xai_pipeline(
        model=best_model,
        X_train=splits["combined"]["X_train"],
        X_test=splits["combined"]["X_test"],
        feature_names=feature_names,
        modality_ranges=MODALITY_RANGES,
        n_lime_instances=args.lime_instances,
        shap_background_size=args.shap_samples,
    )

    # Save XAI outputs
    xai_results["shap_feature_importance"].to_csv(
        os.path.join(args.output_dir, "shap_feature_importance.csv"),
        index=False
    )
    xai_results["lime_feature_importance"].to_csv(
        os.path.join(args.output_dir, "lime_feature_importance.csv"),
        index=False
    )
    xai_results["shap_modality_importance"].to_csv(
        os.path.join(args.output_dir, "shap_modality_importance.csv"),
        index=False
    )
    xai_results["lime_modality_importance"].to_csv(
        os.path.join(args.output_dir, "lime_modality_importance.csv"),
        index=False
    )

    # Save consistency
    consistency_save = {
        k: v for k, v in xai_results["consistency"].items()
        if not isinstance(v, (np.ndarray,))
    }
    # Convert numpy types for JSON
    for k, v in consistency_save.items():
        if isinstance(v, (np.floating, np.integer)):
            consistency_save[k] = float(v)
        elif isinstance(v, np.bool_):
            consistency_save[k] = bool(v)

    with open(os.path.join(args.output_dir, "consistency.json"), "w") as f:
        json.dump(consistency_save, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Step 5: Visualizations
    # ------------------------------------------------------------------
    print("\n[Step 5] Generating visualizations...")
    n_shap = min(args.shap_samples, len(splits["combined"]["X_test"]))
    plot_paths = generate_all_plots(
        all_metrics=all_metrics,
        xai_results=xai_results,
        X_test=splits["combined"]["X_test"][:n_shap],
        feature_names=feature_names,
        output_dir=args.output_dir,
    )

    # ------------------------------------------------------------------
    # Step 6: Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Dataset: UCI Phishing Websites (real data, {len(data['y'])} samples)")
    print(f"\n  Model results:")
    for m in all_metrics:
        auc_str = f"{m['auc_roc']:.4f}" if m['auc_roc'] is not None else "N/A"
        print(f"    {m['model']:20s}  F1={m['f1']:.4f}  AUC={auc_str}")

    print(f"\n  XAI Consistency (SHAP-LIME):")
    c = xai_results["consistency"]
    print(f"    Jaccard Index   : {c['jaccard_index']:.4f}")
    print(f"    Spearman Corr   : {c['spearman_correlation']:.4f}")
    print(f"    Shared top feats: {c['overlap_count']}/{c['top_k']}")

    print(f"\n  Generated files:")
    for name, path in plot_paths.items():
        print(f"    {name}: {path}")
    print(f"    metrics.csv: {os.path.join(args.output_dir, 'metrics.csv')}")
    print(f"    dataset.csv: {os.path.join(args.output_dir, 'dataset.csv')}")
    print(f"    consistency.json: {os.path.join(args.output_dir, 'consistency.json')}")

    print("\n" + "=" * 70)
    print("  Vaishnavi Purohit (24260339) - Project 8 Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
