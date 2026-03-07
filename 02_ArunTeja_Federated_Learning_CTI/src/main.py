"""
Main entry point for Federated Learning CTI project.

Usage:
    python main.py                           # auto-downloads NSL-KDD
    python main.py --dataset kddcup99        # auto-downloads KDD Cup 99
    python main.py --data_path /path/to/csv/ # use local CSV dataset
"""

import argparse
import os
import json

from data_loader import download_and_prepare_data, load_local_dataset
from train import run_federated_training, run_centralised_baseline
from visualize import (plot_fl_convergence, plot_dp_tradeoff,
                       plot_aggregation_comparison, plot_confusion_matrix)


def main():
    parser = argparse.ArgumentParser(
        description='Federated Learning for CTI Sharing')
    parser.add_argument('--dataset', type=str, default='nslkdd',
                        choices=['nslkdd', 'kddcup99'],
                        help='Real dataset to auto-download (default: nslkdd)')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to local CSV dataset directory '
                             '(overrides --dataset)')
    parser.add_argument('--label_col', type=str, default='label',
                        help='Label column name for local datasets')
    parser.add_argument('--n_clients', type=int, default=5,
                        help='Number of FL clients')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Dirichlet alpha for non-IID partitioning')
    parser.add_argument('--n_rounds', type=int, default=20,
                        help='FL communication rounds')
    parser.add_argument('--local_epochs', type=int, default=5,
                        help='Local training epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--max_samples', type=int, default=50000,
                        help='Maximum samples to use')
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    print("=" * 60)
    print("Federated Learning for Privacy-Preserving CTI Sharing")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load REAL data
    # ------------------------------------------------------------------
    if args.data_path is not None:
        partitions, test_data, input_dim = load_local_dataset(
            data_path=args.data_path,
            label_col=args.label_col,
            n_clients=args.n_clients,
            alpha=args.alpha,
            max_samples=args.max_samples,
        )
    else:
        partitions, test_data, input_dim = download_and_prepare_data(
            dataset=args.dataset,
            n_clients=args.n_clients,
            alpha=args.alpha,
            max_samples=args.max_samples,
        )

    for i, (f, l) in enumerate(partitions):
        print(f"  Client {i+1}: {len(f)} samples, "
              f"attack ratio: {l.mean():.3f}")
    print(f"  Test set: {len(test_data[0])} samples")

    # 1. Centralised baseline
    print(f"\n{'='*60}")
    print("Phase 1: Centralised Baseline")
    print("=" * 60)
    _, central_metrics = run_centralised_baseline(
        partitions, test_data, input_dim, epochs=30
    )

    # 2. FedAvg
    print(f"\n{'='*60}")
    print("Phase 2: FedAvg Aggregation")
    print("=" * 60)
    _, fedavg_metrics = run_federated_training(
        partitions, test_data, input_dim,
        aggregation='fedavg', n_rounds=args.n_rounds,
        local_epochs=args.local_epochs, lr=args.lr
    )

    # 3. Krum
    print(f"\n{'='*60}")
    print("Phase 3: Krum Byzantine-Robust Aggregation")
    print("=" * 60)
    _, krum_metrics = run_federated_training(
        partitions, test_data, input_dim,
        aggregation='krum', n_rounds=args.n_rounds,
        local_epochs=args.local_epochs, lr=args.lr, n_malicious=1
    )

    # 4. Median
    print(f"\n{'='*60}")
    print("Phase 4: Coordinate-wise Median Aggregation")
    print("=" * 60)
    _, median_metrics = run_federated_training(
        partitions, test_data, input_dim,
        aggregation='median', n_rounds=args.n_rounds,
        local_epochs=args.local_epochs, lr=args.lr
    )

    # 5. Differential Privacy trade-off
    print(f"\n{'='*60}")
    print("Phase 5: Differential Privacy Trade-off Analysis")
    print("=" * 60)
    epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    dp_f1_scores = []
    for eps in epsilons:
        _, dp_m = run_federated_training(
            partitions, test_data, input_dim,
            aggregation='fedavg', n_rounds=10,
            local_epochs=3, lr=args.lr, epsilon=eps
        )
        dp_f1_scores.append(dp_m['f1'])

    # Generate visualizations
    print(f"\n{'='*60}")
    print("Phase 6: Generating Visualizations")
    print("=" * 60)

    histories = {
        'fedavg': fedavg_metrics['history'],
        'krum': krum_metrics['history'],
        'median': median_metrics['history'],
    }
    plot_fl_convergence(histories)

    plot_dp_tradeoff(epsilons, dp_f1_scores)

    agg_results = {
        'centralised': central_metrics,
        'fedavg': fedavg_metrics,
        'krum': krum_metrics,
        'median': median_metrics,
    }
    plot_aggregation_comparison(agg_results)

    plot_confusion_matrix(fedavg_metrics['confusion_matrix'],
                          'FedAvg Confusion Matrix',
                          'figures/confusion_matrix_fedavg.png')
    plot_confusion_matrix(krum_metrics['confusion_matrix'],
                          'Krum Confusion Matrix',
                          'figures/confusion_matrix_krum.png')

    # Summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"\n{'Method':<20} {'F1':<10} {'Acc':<10} "
          f"{'Prec':<10} {'Recall':<10} {'Time(s)':<10}")
    print("-" * 70)
    for name, m in agg_results.items():
        t = m.get('training_time', 0)
        print(f"{name.upper():<20} {m['f1']:<10.4f} {m['accuracy']:<10.4f} "
              f"{m['precision']:<10.4f} {m['recall']:<10.4f} {t:<10.2f}")

    # Save results
    save_data = {}
    for name, m in agg_results.items():
        save_data[name] = {
            'f1': m['f1'], 'accuracy': m['accuracy'],
            'precision': m['precision'], 'recall': m['recall'],
            'training_time': m.get('training_time', 0),
        }
    save_data['dp_tradeoff'] = {
        'epsilons': epsilons, 'f1_scores': dp_f1_scores,
    }

    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/results.json")
    print("Figures saved to figures/")


if __name__ == '__main__':
    main()
