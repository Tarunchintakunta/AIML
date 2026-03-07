#!/usr/bin/env python3
"""
main.py - Full Pipeline for Transformer-Based Log Anomaly Detection
=====================================================================
Orchestrates the complete workflow:
    1. Download and parse real log data (HDFS, BGL, Thunderbird) from LogHub
    2. Extract features (TF-IDF and simulated transformer embeddings)
    3. Train and evaluate five anomaly detection models
    4. Produce publication-quality figures
    5. Save a JSON results summary

Usage
-----
    python main.py                          # default: all three datasets
    python main.py --datasets hdfs bgl      # specific datasets only
    python main.py --datasets hdfs          # HDFS only

Author : Tejas Vijay Mariyappagoudar (x24213829)
"""

import argparse
import json
import os
import sys
import time
import numpy as np

# Ensure the src directory is on the path when run from project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from data_loader import load_real_datasets, prepare_features
from model import run_all_models
from visualize import generate_all_figures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Transformer-Based Anomaly Detection in Cloud Security Logs"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["hdfs", "bgl", "thunderbird"],
        choices=["hdfs", "bgl", "thunderbird"],
        help="Which LogHub datasets to download and use (default: all three)."
    )
    parser.add_argument(
        "--data_dir", type=str, default=os.path.join(PROJECT_DIR, "data"),
        help="Directory for storing downloaded log files."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)."
    )
    parser.add_argument(
        "--output_dir", type=str, default=os.path.join(SCRIPT_DIR, "figures"),
        help="Directory for output figures."
    )
    parser.add_argument(
        "--results_file", type=str, default=os.path.join(SCRIPT_DIR, "results.json"),
        help="Path for JSON results file."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    overall_start = time.perf_counter()

    print("=" * 70)
    print("  Transformer-Based Anomaly Detection in Cloud Security Logs")
    print("  Student: Tejas Vijay Mariyappagoudar (x24213829)")
    print("=" * 70)

    # ---- Step 1: Download and parse real log data -------------------------
    print(f"\n[Step 1/5] Downloading and parsing real log datasets ...")
    print(f"  Datasets: {args.datasets}")
    print(f"  Data dir: {args.data_dir}")

    df = load_real_datasets(
        data_dir=args.data_dir,
        dataset_sources=args.datasets,
        seed=args.seed,
    )

    # Compute actual anomaly ratio from the real data
    anomaly_ratio = df["label"].mean()
    # Clamp to a reasonable range for contamination parameter
    contamination = max(0.01, min(0.5, anomaly_ratio))

    print(f"\n  Total logs     : {len(df)}")
    print(f"  Normal         : {(df['label']==0).sum()}")
    print(f"  Anomaly        : {(df['label']==1).sum()}")
    print(f"  Anomaly ratio  : {anomaly_ratio:.2%}")
    print(f"  Sources        : {df['source'].value_counts().to_dict()}")

    # ---- Step 2: Feature extraction ---------------------------------------
    print("\n[Step 2/5] Extracting features ...")

    t0 = time.perf_counter()
    X_tfidf, extras_tfidf = prepare_features(df, method="tfidf", max_features=512)
    tfidf_time = time.perf_counter() - t0
    print(f"  TF-IDF shape   : {X_tfidf.shape}  ({tfidf_time:.3f}s)")

    t0 = time.perf_counter()
    X_trans, _ = prepare_features(df, method="transformer", dim=384, seed=args.seed)
    trans_time = time.perf_counter() - t0
    print(f"  Transformer shape: {X_trans.shape}  ({trans_time:.3f}s)")

    y_true = df["label"].values

    # ---- Step 3: Model training & evaluation ------------------------------
    print("\n[Step 3/5] Training and evaluating models ...")
    print(f"  Using contamination = {contamination:.4f} (from real data)")
    results = run_all_models(
        X_tfidf=X_tfidf,
        X_transformer=X_trans,
        y_true=y_true,
        contamination=contamination,
    )

    # ---- Step 4: Visualisation --------------------------------------------
    print("\n[Step 4/5] Generating visualisations ...")
    figure_paths = generate_all_figures(
        results=results,
        X_transformer=X_trans,
        labels=y_true,
        output_dir=args.output_dir,
    )

    # ---- Step 5: Save results JSON ----------------------------------------
    print("\n[Step 5/5] Saving results ...")
    total_time = time.perf_counter() - overall_start

    summary = {
        "project": "Transformer-Based Anomaly Detection in Cloud Security Logs",
        "student": "Tejas Vijay Mariyappagoudar (x24213829)",
        "dataset": {
            "type": "real (LogHub)",
            "sources": args.datasets,
            "total_samples": len(df),
            "anomaly_ratio": round(anomaly_ratio, 4),
            "per_source": df["source"].value_counts().to_dict(),
            "urls": {
                "hdfs_log": "https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS_2k.log",
                "hdfs_structured": "https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS_2k.log_structured.csv",
                "bgl_log": "https://raw.githubusercontent.com/logpai/loghub/master/BGL/BGL_2k.log",
                "bgl_structured": "https://raw.githubusercontent.com/logpai/loghub/master/BGL/BGL_2k.log_structured.csv",
                "thunderbird_log": "https://raw.githubusercontent.com/logpai/loghub/master/Thunderbird/Thunderbird_2k.log",
                "thunderbird_structured": "https://raw.githubusercontent.com/logpai/loghub/master/Thunderbird/Thunderbird_2k.log_structured.csv",
            },
        },
        "features": {
            "tfidf_dim": int(X_tfidf.shape[1]),
            "transformer_dim": int(X_trans.shape[1]),
            "tfidf_extraction_sec": round(tfidf_time, 4),
            "transformer_extraction_sec": round(trans_time, 4),
        },
        "models": [r.as_dict() for r in results],
        "best_model": max(results, key=lambda r: r.f1).as_dict(),
        "figures": {k: os.path.basename(v) for k, v in figure_paths.items()},
        "total_pipeline_time_sec": round(total_time, 4),
    }

    os.makedirs(os.path.dirname(args.results_file) or ".", exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Results saved to: {args.results_file}")

    # ---- Summary ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'Model':<36s} {'F1':>6s} {'Acc':>6s} {'Prec':>6s} {'Rec':>6s} {'Latency':>8s}")
    print("  " + "-" * 68)
    for r in results:
        print(f"  {r.name:<36s} {r.f1:>6.4f} {r.accuracy:>6.4f} "
              f"{r.precision:>6.4f} {r.recall:>6.4f} {r.total_latency_sec:>7.4f}s")

    best = max(results, key=lambda r: r.f1)
    print(f"\n  Best model : {best.name}  (F1 = {best.f1:.4f})")
    print(f"  Total time : {total_time:.2f}s")

    # Print classification report for best model
    print(f"\n  Classification Report ({best.name}):")
    for line in best.report.split("\n"):
        print(f"    {line}")

    print("=" * 70)
    print("  Pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
