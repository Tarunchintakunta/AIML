"""
Visualization module for training curves, confusion matrices, and comparison charts.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_curves(history, save_path='figures/training_curves.png'):
    """Plot training loss and validation F1/accuracy over epochs."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss Over Epochs', fontsize=14)
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['val_f1'], 'r-', linewidth=2, label='F1-Score')
    ax2.plot(epochs, history['val_accuracy'], 'g--', linewidth=2, label='Accuracy')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Validation Metrics Over Epochs', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_confusion_matrix(cm, save_path='figures/confusion_matrix.png',
                          title='GraphSAGE Confusion Matrix'):
    """Plot confusion matrix heatmap."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malicious'],
                yticklabels=['Benign', 'Malicious'],
                ax=ax, annot_kws={'size': 14})
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_model_comparison(graphsage_metrics, baseline_results,
                          save_path='figures/model_comparison.png'):
    """Bar chart comparing GraphSAGE vs baselines."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    models = ['GraphSAGE']
    f1_scores = [graphsage_metrics['f1']]
    accuracies = [graphsage_metrics['accuracy']]

    for name, metrics in baseline_results.items():
        if metrics is not None:
            models.append(name.replace('_', ' ').title())
            f1_scores.append(metrics['f1'])
            accuracies.append(metrics['accuracy'])

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, f1_scores, width, label='F1-Score', color='#2196F3')
    bars2 = ax.bar(x + width / 2, accuracies, width, label='Accuracy', color='#4CAF50')

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis='y', alpha=0.3)

    for bar in bars1:
        ax.annotate(f'{bar.get_height():.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.annotate(f'{bar.get_height():.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_training_time_comparison(graphsage_time, baseline_results,
                                  save_path='figures/training_time.png'):
    """Compare training times across models."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    models = ['GraphSAGE']
    times = [graphsage_time]

    for name, metrics in baseline_results.items():
        if metrics is not None:
            models.append(name.replace('_', ' ').title())
            times.append(metrics['training_time'])

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#FF9800', '#2196F3', '#4CAF50', '#9C27B0']
    bars = ax.bar(models, times, color=colors[:len(models)])

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Training Time (seconds)', fontsize=12)
    ax.set_title('Training Time Comparison (CPU)', fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)

    for bar in bars:
        ax.annotate(f'{bar.get_height():.1f}s',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
