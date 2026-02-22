"""
Main entry point for GraphSAGE APT Detection project.

Usage:
    # With synthetic data (for testing):
    python main.py --synthetic

    # With CIC-IDS2017 dataset:
    python main.py --dataset cicids2017 --data_path /path/to/cicids2017/

    # With ToN-IoT dataset:
    python main.py --dataset toniot --data_path /path/to/toniot/
"""

import argparse
import os
import sys
import json

from data_loader import prepare_dataset, generate_synthetic_data
from model import GraphSAGEDetector, GraphSAGEThreeLayer
from train import run_training
from baselines import run_all_baselines
from visualize import (plot_training_curves, plot_confusion_matrix,
                       plot_model_comparison, plot_training_time_comparison)


def main():
    parser = argparse.ArgumentParser(
        description='GraphSAGE APT Detection with Neighbour Sampling')
    parser.add_argument('--dataset', type=str, default='cicids2017',
                        choices=['cicids2017', 'toniot'],
                        help='Dataset to use')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to dataset CSV directory')
    parser.add_argument('--label_col', type=str, default='Label',
                        help='Name of label column in dataset')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data for testing')
    parser.add_argument('--max_samples', type=int, default=50000,
                        help='Maximum samples to use from dataset')
    parser.add_argument('--hidden', type=int, default=128,
                        help='Hidden layer size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum training epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Mini-batch size for neighbour sampling')
    parser.add_argument('--minibatch', action='store_true',
                        help='Use mini-batch training with neighbour sampling')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    # Load data
    print("=" * 60)
    print("GraphSAGE APT Detection Pipeline")
    print("=" * 60)

    if args.synthetic:
        print("\nUsing synthetic data for demonstration...")
        data = generate_synthetic_data(n_nodes=5000, n_features=20, attack_ratio=0.2)
    else:
        if args.data_path is None:
            print("\nNo data_path provided. Using synthetic data for demo.")
            print("For real datasets, use: python main.py --data_path /path/to/data/")
            data = generate_synthetic_data(n_nodes=5000, n_features=20, attack_ratio=0.2)
        else:
            print(f"\nLoading {args.dataset} from {args.data_path}...")
            data = prepare_dataset(args.data_path, args.dataset,
                                   args.label_col, args.max_samples)

    print(f"\nGraph Statistics:")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.num_edges}")
    print(f"  Features: {data.num_node_features}")
    print(f"  Train: {data.train_mask.sum().item()}")
    print(f"  Val:   {data.val_mask.sum().item()}")
    print(f"  Test:  {data.test_mask.sum().item()}")
    print(f"  Attack ratio: {data.y.float().mean():.3f}")

    # Train GraphSAGE (2-layer)
    print(f"\n{'='*60}")
    print("Phase 1: Training GraphSAGE (2-layer)")
    print("=" * 60)
    model_2l, results_2l = run_training(
        data, model_class=GraphSAGEDetector,
        hidden_channels=args.hidden, epochs=args.epochs,
        lr=args.lr, use_minibatch=args.minibatch,
        batch_size=args.batch_size
    )

    # Train GraphSAGE (3-layer)
    print(f"\n{'='*60}")
    print("Phase 2: Training GraphSAGE (3-layer)")
    print("=" * 60)
    model_3l, results_3l = run_training(
        data, model_class=GraphSAGEThreeLayer,
        hidden_channels=args.hidden, epochs=args.epochs,
        lr=args.lr, use_minibatch=args.minibatch,
        batch_size=args.batch_size
    )

    # Run baselines
    print(f"\n{'='*60}")
    print("Phase 3: Running Baseline Classifiers")
    print("=" * 60)
    baseline_results = run_all_baselines(data)

    # Generate visualizations
    print(f"\n{'='*60}")
    print("Phase 4: Generating Visualizations")
    print("=" * 60)

    plot_training_curves(results_2l['history'], 'figures/training_curves_2layer.png')
    plot_training_curves(results_3l['history'], 'figures/training_curves_3layer.png')

    plot_confusion_matrix(results_2l['test_metrics']['confusion_matrix'],
                          'figures/confusion_matrix_2layer.png',
                          'GraphSAGE (2-Layer) Confusion Matrix')
    plot_confusion_matrix(results_3l['test_metrics']['confusion_matrix'],
                          'figures/confusion_matrix_3layer.png',
                          'GraphSAGE (3-Layer) Confusion Matrix')

    plot_model_comparison(results_2l['test_metrics'], baseline_results,
                          'figures/model_comparison.png')

    plot_training_time_comparison(results_2l['training_time'], baseline_results,
                                  'figures/training_time.png')

    # Summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"\n{'Model':<25} {'F1-Score':<12} {'Accuracy':<12} {'AUC-ROC':<12} {'Time(s)':<10}")
    print("-" * 71)
    print(f"{'GraphSAGE (2-layer)':<25} "
          f"{results_2l['test_metrics']['f1']:<12.4f} "
          f"{results_2l['test_metrics']['accuracy']:<12.4f} "
          f"{results_2l['test_metrics']['auc_roc']:<12.4f} "
          f"{results_2l['training_time']:<10.2f}")
    print(f"{'GraphSAGE (3-layer)':<25} "
          f"{results_3l['test_metrics']['f1']:<12.4f} "
          f"{results_3l['test_metrics']['accuracy']:<12.4f} "
          f"{results_3l['test_metrics']['auc_roc']:<12.4f} "
          f"{results_3l['training_time']:<10.2f}")

    for name, metrics in baseline_results.items():
        if metrics is not None:
            display = name.replace('_', ' ').title()
            auc = metrics.get('auc_roc', 0)
            print(f"{display:<25} "
                  f"{metrics['f1']:<12.4f} "
                  f"{metrics['accuracy']:<12.4f} "
                  f"{auc:<12.4f} "
                  f"{metrics['training_time']:<10.2f}")

    # Save results
    save_results = {
        'graphsage_2layer': {
            'f1': results_2l['test_metrics']['f1'],
            'accuracy': results_2l['test_metrics']['accuracy'],
            'precision': results_2l['test_metrics']['precision'],
            'recall': results_2l['test_metrics']['recall'],
            'auc_roc': results_2l['test_metrics']['auc_roc'],
            'training_time': results_2l['training_time'],
        },
        'graphsage_3layer': {
            'f1': results_3l['test_metrics']['f1'],
            'accuracy': results_3l['test_metrics']['accuracy'],
            'precision': results_3l['test_metrics']['precision'],
            'recall': results_3l['test_metrics']['recall'],
            'auc_roc': results_3l['test_metrics']['auc_roc'],
            'training_time': results_3l['training_time'],
        },
    }

    for name, metrics in baseline_results.items():
        if metrics is not None:
            save_results[name] = {
                'f1': metrics['f1'],
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'auc_roc': metrics.get('auc_roc', 0),
                'training_time': metrics['training_time'],
            }

    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/results.json")
    print("Figures saved to figures/")


if __name__ == '__main__':
    main()
