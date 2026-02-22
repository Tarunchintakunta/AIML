"""
Visualization for Federated Learning CTI experiments.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_fl_convergence(histories: dict, save_path='figures/fl_convergence.png'):
    """Plot F1-score convergence across FL rounds for different aggregation methods."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = {'fedavg': '#2196F3', 'krum': '#F44336', 'median': '#4CAF50'}

    for name, hist in histories.items():
        color = colors.get(name, '#000000')
        ax1.plot(hist['round'], hist['f1'], '-o', label=name.upper(),
                 color=color, markersize=3, linewidth=2)
        ax2.plot(hist['round'], hist['accuracy'], '-s', label=name.upper(),
                 color=color, markersize=3, linewidth=2)

    ax1.set_xlabel('Communication Round', fontsize=12)
    ax1.set_ylabel('F1-Score', fontsize=12)
    ax1.set_title('F1-Score Convergence', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Communication Round', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Accuracy Convergence', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_dp_tradeoff(epsilons: list, f1_scores: list,
                     save_path='figures/dp_tradeoff.png'):
    """Plot privacy-accuracy trade-off across epsilon values."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epsilons, f1_scores, 'o-', color='#FF9800', linewidth=2, markersize=8)
    ax.set_xlabel('Privacy Budget (ε)', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('Differential Privacy: Privacy-Accuracy Trade-off', fontsize=14)
    ax.grid(True, alpha=0.3)

    for e, f in zip(epsilons, f1_scores):
        ax.annotate(f'{f:.3f}', (e, f), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_aggregation_comparison(results: dict,
                                save_path='figures/aggregation_comparison.png'):
    """Bar chart comparing aggregation methods."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    methods = list(results.keys())
    f1s = [results[m]['f1'] for m in methods]
    accs = [results[m]['accuracy'] for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, f1s, width, label='F1-Score', color='#2196F3')
    bars2 = ax.bar(x + width/2, accs, width, label='Accuracy', color='#4CAF50')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Aggregation Method Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in methods])
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis='y', alpha=0.3)

    for bar in bars1:
        ax.annotate(f'{bar.get_height():.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    for bar in bars2:
        ax.annotate(f'{bar.get_height():.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_confusion_matrix(cm, title='Confusion Matrix',
                          save_path='figures/confusion_matrix.png'):
    """Plot confusion matrix heatmap."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malicious'],
                yticklabels=['Benign', 'Malicious'],
                ax=ax, annot_kws={'size': 14})
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
