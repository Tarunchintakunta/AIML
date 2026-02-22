"""
Visualization for XAI NIDS Ensemble experiments.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_model_comparison(results, save_path='figures/model_comparison.png'):
    """Bar chart comparing model performance metrics."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    models = list(results.keys())
    f1s = [results[m]['f1'] for m in models]
    accs = [results[m]['accuracy'] for m in models]
    precs = [results[m]['precision'] for m in models]
    recs = [results[m]['recall'] for m in models]

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 1.5*width, f1s, width, label='F1-Score', color='#2196F3')
    ax.bar(x - 0.5*width, accs, width, label='Accuracy', color='#4CAF50')
    ax.bar(x + 0.5*width, precs, width, label='Precision', color='#FF9800')
    ax.bar(x + 1.5*width, recs, width, label='Recall', color='#F44336')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Ensemble Model Performance Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis='y', alpha=0.3)

    for bars in ax.containers:
        ax.bar_label(bars, fmt='%.3f', fontsize=8, padding=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_confusion_matrices(results, save_path='figures/confusion_matrices.png'):
    """Plot confusion matrices for all models."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    models = list(results.keys())
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, models):
        cm = results[name]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Benign', 'Malicious'],
                    yticklabels=['Benign', 'Malicious'],
                    ax=ax, annot_kws={'size': 12})
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Actual', fontsize=10)
        ax.set_title(name, fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_xai_comparison(xai_results, feature_names,
                        save_path='figures/xai_comparison.png'):
    """Compare SHAP vs LIME feature importance across models."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    models = list(xai_results.keys())
    n = len(models)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4*n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, name in enumerate(models):
        xai = xai_results[name]

        # Top 10 features by SHAP
        shap_imp = xai['shap_importance']
        top_shap = np.argsort(-shap_imp)[:10]
        axes[i, 0].barh(range(10), shap_imp[top_shap], color='#2196F3')
        axes[i, 0].set_yticks(range(10))
        axes[i, 0].set_yticklabels([feature_names[j] for j in top_shap],
                                     fontsize=8)
        axes[i, 0].set_title(f'{name} - SHAP Top 10', fontsize=11)
        axes[i, 0].invert_yaxis()

        # Top 10 features by LIME
        lime_imp = xai['lime_importance']
        top_lime = np.argsort(-lime_imp)[:10]
        axes[i, 1].barh(range(10), lime_imp[top_lime], color='#FF9800')
        axes[i, 1].set_yticks(range(10))
        axes[i, 1].set_yticklabels([feature_names[j] for j in top_lime],
                                     fontsize=8)
        axes[i, 1].set_title(f'{name} - LIME Top 10', fontsize=11)
        axes[i, 1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_consistency_fidelity(xai_results,
                              save_path='figures/consistency_fidelity.png'):
    """Plot consistency index and fidelity scores across models."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    models = list(xai_results.keys())
    taus = [xai_results[m]['consistency_tau'] for m in models]
    fids = [xai_results[m]['fidelity'] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, taus, width, label='Consistency Index',
                   color='#2196F3')
    bars2 = ax.bar(x + width/2, fids, width, label='Explanation Fidelity',
                   color='#4CAF50')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('XAI Quality Metrics by Ensemble Architecture', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)

    for bar in bars1:
        ax.annotate(f'{bar.get_height():.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center',
                    fontsize=9)
    for bar in bars2:
        ax.annotate(f'{bar.get_height():.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center',
                    fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_overhead_comparison(xai_results,
                             save_path='figures/overhead_comparison.png'):
    """Plot computational overhead ratios for SHAP vs LIME."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    models = list(xai_results.keys())
    shap_times = [xai_results[m]['shap_time'] for m in models]
    lime_times = [xai_results[m]['lime_time'] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, shap_times, width, label='SHAP', color='#2196F3')
    ax.bar(x + width/2, lime_times, width, label='LIME', color='#FF9800')

    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('XAI Computational Overhead: SHAP vs LIME', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
