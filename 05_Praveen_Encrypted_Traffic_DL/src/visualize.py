"""
Visualization for Encrypted Traffic Classification experiments.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_model_comparison(results, save_path='figures/model_comparison.png'):
    """Bar chart comparing model performance."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    models = list(results.keys())
    f1s = [results[m]['f1_macro'] for m in models]
    accs = [results[m]['accuracy'] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, f1s, width, label='Macro F1', color='#2196F3')
    ax.bar(x + width/2, accs, width, label='Accuracy', color='#4CAF50')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Encrypted Traffic Classification: Model Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis='y', alpha=0.3)

    for bars in ax.containers:
        ax.bar_label(bars, fmt='%.3f', fontsize=9, padding=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_confusion_matrix(cm, class_names, title='Confusion Matrix',
                          save_path='figures/confusion_matrix.png'):
    """Plot multi-class confusion matrix."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, annot_kws={'size': 10})
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_latency_comparison(latencies, save_path='figures/latency_comparison.png'):
    """Plot inference latency comparison."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    models = list(latencies.keys())
    times_ms = [latencies[m] * 1000 for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, times_ms, color='#FF9800')
    ax.set_ylabel('Latency per Sample (ms)', fontsize=12)
    ax.set_title('Model Inference Latency Comparison', fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)

    for bar in bars:
        ax.annotate(f'{bar.get_height():.3f}ms',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_training_curves(history, save_path='figures/training_curves.png'):
    """Plot training loss and validation F1 curves."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for name, hist in history.items():
        if 'train_loss' in hist:
            ax1.plot(hist['train_loss'], label=name, linewidth=2)
        if 'val_f1' in hist:
            ax2.plot(hist['val_f1'], label=name, linewidth=2)

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss Convergence', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation F1-Score', fontsize=12)
    ax2.set_title('Validation F1 Convergence', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_per_class_f1(results, class_names,
                      save_path='figures/per_class_f1.png'):
    """Plot per-class F1-scores for the best model."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    from sklearn.metrics import f1_score as f1_fn

    fig, ax = plt.subplots(figsize=(10, 6))

    # Use confusion matrix to compute per-class metrics
    for model_name, metrics in results.items():
        cm = metrics['confusion_matrix']
        per_class = []
        for i in range(len(class_names)):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0
            per_class.append(f1)

        ax.bar(np.arange(len(class_names)) + list(results.keys()).index(model_name)*0.2,
               per_class, 0.2, label=model_name)

    ax.set_xticks(np.arange(len(class_names)) + 0.3)
    ax.set_xticklabels(class_names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('Per-Class F1-Score Comparison', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
