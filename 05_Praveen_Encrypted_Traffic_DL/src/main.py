"""
Main entry point for Encrypted Traffic Classification project.

Usage:
    python main.py                          # Auto-download KDD Cup 99 real traffic data
    python main.py --dataset /path/to.csv   # Use local ISCX-VPN-NonVPN CSV file
"""

import argparse
import os
import json
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from data_loader import load_kddcup99_traffic, load_iscx_csv, preprocess_data
from model import (CNNClassifier, LSTMClassifier, CNNLSTMAttention,
                   train_model, evaluate_model, measure_latency,
                   train_random_forest)
from visualize import (plot_model_comparison, plot_confusion_matrix,
                       plot_latency_comparison, plot_per_class_f1)


def main():
    parser = argparse.ArgumentParser(
        description='Deep Learning for Encrypted Traffic Classification'
    )
    parser.add_argument('--dataset', type=str, default=None,
                        help='Path to local ISCX-VPN-NonVPN CSV file. '
                             'If not provided, auto-downloads KDD Cup 99 '
                             'real network traffic data.')
    parser.add_argument('--n_samples', type=int, default=7000,
                        help='Max samples (balanced across classes)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    print("=" * 60)
    print("Deep Learning for Encrypted Traffic Classification")
    print("=" * 60)

    # Data
    print("\nPhase 1: Data Loading & Preprocessing")
    print("-" * 60)
    if args.dataset and os.path.exists(args.dataset):
        features, labels, feature_names, class_names = load_iscx_csv(
            args.dataset
        )
    else:
        if args.dataset:
            print(f"  WARNING: File not found: {args.dataset}")
            print("  Falling back to KDD Cup 99 dataset.")
        features, labels, feature_names, class_names = load_kddcup99_traffic(
            n_samples=args.n_samples
        )

    train_data, val_data, test_data, scaler = preprocess_data(features, labels)
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    input_dim = X_train.shape[1]
    n_classes = len(class_names)

    print(f"  Features: {input_dim} | Classes: {n_classes}")
    print(f"  Classes: {class_names}")

    all_results = {}

    # Random Forest baseline
    print(f"\n{'=' * 60}")
    print("Phase 2: Random Forest Baseline")
    print("=" * 60)
    rf, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test,
                                          class_names)
    print(f"  RF: Acc={rf_metrics['accuracy']:.4f} | "
          f"F1={rf_metrics['f1_macro']:.4f} | "
          f"Train={rf_metrics['train_time']:.2f}s")
    all_results['Random Forest'] = rf_metrics

    # CNN
    print(f"\n{'=' * 60}")
    print("Phase 3: CNN Classifier")
    print("=" * 60)
    cnn = CNNClassifier(input_dim=input_dim, n_classes=n_classes)
    start = time.time()
    cnn = train_model(cnn, X_train, y_train, X_val, y_val, epochs=args.epochs)
    cnn_train_time = time.time() - start
    cnn_metrics = evaluate_model(cnn, X_test, y_test, class_names)
    cnn_metrics['train_time'] = cnn_train_time
    print(f"  CNN: Acc={cnn_metrics['accuracy']:.4f} | "
          f"F1={cnn_metrics['f1_macro']:.4f} | Train={cnn_train_time:.2f}s")
    all_results['CNN'] = cnn_metrics

    # LSTM
    print(f"\n{'=' * 60}")
    print("Phase 4: LSTM Classifier")
    print("=" * 60)
    lstm = LSTMClassifier(input_dim=input_dim, n_classes=n_classes)
    start = time.time()
    lstm = train_model(lstm, X_train, y_train, X_val, y_val, epochs=args.epochs)
    lstm_train_time = time.time() - start
    lstm_metrics = evaluate_model(lstm, X_test, y_test, class_names)
    lstm_metrics['train_time'] = lstm_train_time
    print(f"  LSTM: Acc={lstm_metrics['accuracy']:.4f} | "
          f"F1={lstm_metrics['f1_macro']:.4f} | Train={lstm_train_time:.2f}s")
    all_results['LSTM'] = lstm_metrics

    # CNN-LSTM with Attention (proposed hybrid)
    print(f"\n{'=' * 60}")
    print("Phase 5: CNN-LSTM with Attention (Proposed)")
    print("=" * 60)
    hybrid = CNNLSTMAttention(input_dim=input_dim, n_classes=n_classes)
    start = time.time()
    hybrid = train_model(hybrid, X_train, y_train, X_val, y_val,
                          epochs=args.epochs)
    hybrid_train_time = time.time() - start
    hybrid_metrics = evaluate_model(hybrid, X_test, y_test, class_names)
    hybrid_metrics['train_time'] = hybrid_train_time
    print(f"  CNN-LSTM-Attn: Acc={hybrid_metrics['accuracy']:.4f} | "
          f"F1={hybrid_metrics['f1_macro']:.4f} | Train={hybrid_train_time:.2f}s")
    all_results['CNN-LSTM-Attn'] = hybrid_metrics

    # Latency measurement
    print(f"\n{'=' * 60}")
    print("Phase 6: Inference Latency")
    print("=" * 60)
    latencies = {}
    latencies['CNN'] = measure_latency(cnn, X_test)
    latencies['LSTM'] = measure_latency(lstm, X_test)
    latencies['CNN-LSTM-Attn'] = measure_latency(hybrid, X_test)

    for name, lat in latencies.items():
        print(f"  {name}: {lat*1000:.3f} ms/sample")

    # Visualizations
    print(f"\n{'=' * 60}")
    print("Phase 7: Generating Visualizations")
    print("=" * 60)

    plot_model_comparison(all_results)
    plot_confusion_matrix(hybrid_metrics['confusion_matrix'], class_names,
                          'CNN-LSTM-Attention Confusion Matrix',
                          'figures/cm_hybrid.png')
    plot_confusion_matrix(rf_metrics['confusion_matrix'], class_names,
                          'Random Forest Confusion Matrix',
                          'figures/cm_rf.png')
    plot_latency_comparison(latencies)
    plot_per_class_f1(all_results, class_names)

    # Summary
    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"\n{'Model':<20} {'F1-Macro':<10} {'Acc':<10} {'Prec':<10} "
          f"{'Recall':<10} {'Train(s)':<10}")
    print("-" * 70)
    for name, m in all_results.items():
        print(f"{name:<20} {m['f1_macro']:<10.4f} {m['accuracy']:<10.4f} "
              f"{m['precision']:<10.4f} {m['recall']:<10.4f} "
              f"{m.get('train_time', 0):<10.2f}")

    # Save
    save_data = {}
    for name, m in all_results.items():
        save_data[name] = {
            'f1_macro': m['f1_macro'], 'accuracy': m['accuracy'],
            'precision': m['precision'], 'recall': m['recall'],
            'train_time': m.get('train_time', 0),
        }
    save_data['latencies'] = {k: v*1000 for k, v in latencies.items()}

    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/results.json")
    print("Figures saved to figures/")


if __name__ == '__main__':
    main()
