"""
Main entry point for XAI NIDS Ensemble project.

Usage:
    python main.py --synthetic
    python main.py --data_path /path/to/unswnb15/
"""

import argparse
import os
import json
import warnings
import numpy as np

warnings.filterwarnings('ignore')

from data_loader import generate_synthetic_data, preprocess_data
from models import get_all_models, train_and_evaluate
from explainers import run_xai_analysis
from visualize import (plot_model_comparison, plot_confusion_matrices,
                       plot_xai_comparison, plot_consistency_fidelity,
                       plot_overhead_comparison)


def main():
    parser = argparse.ArgumentParser(
        description='XAI for NIDS Ensemble Learning'
    )
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--n_samples', type=int, default=8000)
    parser.add_argument('--n_explain', type=int, default=50,
                        help='Number of samples for XAI explanation')
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    print("=" * 60)
    print("XAI for Network Intrusion Detection: Ensemble Learning")
    print("=" * 60)

    # Load data
    print("\nPhase 1: Data Loading & Preprocessing")
    print("-" * 60)
    print("Using synthetic UNSW-NB15-like data for demonstration...")
    features, labels, feature_names = generate_synthetic_data(
        n_samples=args.n_samples
    )
    train_data, val_data, test_data = preprocess_data(
        features, labels, feature_names
    )
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    # Train all ensemble models
    print(f"\n{'=' * 60}")
    print("Phase 2: Ensemble Model Training & Evaluation")
    print("=" * 60)

    all_models = get_all_models()
    model_results = {}
    trained_models = {}

    for name, model in all_models.items():
        model, metrics = train_and_evaluate(
            model, X_train, y_train, X_test, y_test, model_name=name
        )
        model_results[name] = metrics
        trained_models[name] = model

    # XAI Analysis
    print(f"\n{'=' * 60}")
    print("Phase 3: Explainability Analysis (SHAP & LIME)")
    print("=" * 60)

    xai_results = {}
    for name, model in trained_models.items():
        xai = run_xai_analysis(
            model, X_train, X_test, y_test, feature_names,
            model_name=name, n_explain=args.n_explain
        )
        xai_results[name] = xai

    # Generate visualizations
    print(f"\n{'=' * 60}")
    print("Phase 4: Generating Visualizations")
    print("=" * 60)

    plot_model_comparison(model_results)
    plot_confusion_matrices(model_results)
    plot_xai_comparison(xai_results, feature_names)
    plot_consistency_fidelity(xai_results)
    plot_overhead_comparison(xai_results)

    # Summary
    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY")
    print("=" * 60)

    print(f"\n{'Model':<18} {'F1':<8} {'Acc':<8} {'Prec':<8} "
          f"{'Recall':<8} {'Train(s)':<10}")
    print("-" * 60)
    for name, m in model_results.items():
        print(f"{name:<18} {m['f1']:<8.4f} {m['accuracy']:<8.4f} "
              f"{m['precision']:<8.4f} {m['recall']:<8.4f} "
              f"{m['train_time']:<10.2f}")

    print(f"\n{'Model':<18} {'Consist.':<10} {'Fidelity':<10} "
          f"{'SHAP(s)':<10} {'LIME(s)':<10}")
    print("-" * 60)
    for name, x in xai_results.items():
        print(f"{name:<18} {x['consistency_tau']:<10.4f} "
              f"{x['fidelity']:<10.4f} {x['shap_time']:<10.2f} "
              f"{x['lime_time']:<10.2f}")

    # Save results
    save_data = {}
    for name, m in model_results.items():
        save_data[name] = {
            'f1': m['f1'], 'accuracy': m['accuracy'],
            'precision': m['precision'], 'recall': m['recall'],
            'train_time': m['train_time'],
            'consistency_tau': xai_results[name]['consistency_tau'],
            'fidelity': xai_results[name]['fidelity'],
            'shap_time': xai_results[name]['shap_time'],
            'lime_time': xai_results[name]['lime_time'],
        }

    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/results.json")
    print("Figures saved to figures/")


if __name__ == '__main__':
    main()
