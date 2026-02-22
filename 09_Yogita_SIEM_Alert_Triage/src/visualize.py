"""
visualize.py - Visualization Module for SIEM Alert Triage
==========================================================
Plotting functions for model comparison, confusion matrices,
alert reduction analysis, cost analysis, and SHAP importance.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for file output
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Dict, List, Any, Optional
import os

# Style configuration
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("Set2", 8)
CLASS_NAMES = ["False Positive", "Indeterminate", "True Positive"]
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Model comparison bar chart (F-beta, precision, recall per class)
# ---------------------------------------------------------------------------
def plot_model_comparison(
    results_list: List[Dict[str, Any]],
    save_path: Optional[str] = None,
) -> None:
    """Bar chart comparing F-beta(2) macro, precision, and recall across models."""
    _ensure_output_dir()
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "model_comparison.png")

    model_names = [r["model_name"] for r in results_list]
    n_models = len(model_names)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # -- F-beta macro --
    fbeta_vals = [r["fbeta_macro"] for r in results_list]
    bars = axes[0].bar(model_names, fbeta_vals, color=PALETTE[:n_models],
                       edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel(r"$F_{\beta=2}$ (macro)")
    axes[0].set_title(r"$F_{\beta=2}$ Score Comparison")
    axes[0].set_ylim(0, 1)
    for bar, val in zip(bars, fbeta_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.02,
                     f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")

    # -- Per-class recall --
    x = np.arange(3)
    width = 0.8 / n_models
    for i, r in enumerate(results_list):
        offset = (i - n_models/2 + 0.5) * width
        axes[1].bar(x + offset, r["recall_per_class"], width,
                    label=r["model_name"], color=PALETTE[i],
                    edgecolor="black", linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(CLASS_NAMES, fontsize=9)
    axes[1].set_ylabel("Recall")
    axes[1].set_title("Per-Class Recall")
    axes[1].set_ylim(0, 1.1)
    axes[1].legend(fontsize=8)

    # -- Per-class precision --
    for i, r in enumerate(results_list):
        offset = (i - n_models/2 + 0.5) * width
        axes[2].bar(x + offset, r["precision_per_class"], width,
                    label=r["model_name"], color=PALETTE[i],
                    edgecolor="black", linewidth=0.5)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(CLASS_NAMES, fontsize=9)
    axes[2].set_ylabel("Precision")
    axes[2].set_title("Per-Class Precision")
    axes[2].set_ylim(0, 1.1)
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {save_path}")


# ---------------------------------------------------------------------------
# 2. Confusion matrices (one per model, side by side)
# ---------------------------------------------------------------------------
def plot_confusion_matrices(
    results_list: List[Dict[str, Any]],
    save_path: Optional[str] = None,
) -> None:
    """Plot normalised confusion matrices for each model."""
    _ensure_output_dir()
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "confusion_matrices.png")

    n = len(results_list)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, results_list):
        cm = r["confusion_matrix"]
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        sns.heatmap(
            cm_norm, annot=True, fmt=".2%", cmap="YlOrRd",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=ax, cbar=True, linewidths=0.5,
            annot_kws={"fontsize": 10},
        )
        # Overlay raw counts
        for i in range(3):
            for j in range(3):
                ax.text(j + 0.5, i + 0.72, f"(n={cm[i, j]})",
                        ha="center", va="center", fontsize=7, color="gray")
        ax.set_title(f"{r['model_name']}", fontsize=12, fontweight="bold")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {save_path}")


# ---------------------------------------------------------------------------
# 3. Alert reduction analysis
# ---------------------------------------------------------------------------
def plot_alert_reduction(
    results_list: List[Dict[str, Any]],
    baseline_escalation: float = 1.0,
    save_path: Optional[str] = None,
) -> None:
    """Show how each model reduces the volume of alerts analysts must review."""
    _ensure_output_dir()
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "alert_reduction.png")

    model_names = ["No ML\n(all escalated)"] + [r["model_name"] for r in results_list]
    escalation_rates = [baseline_escalation] + [
        1.0 - r["alert_reduction_rate"] for r in results_list
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#d9534f"] + [PALETTE[i] for i in range(len(results_list))]
    bars = ax.bar(model_names, escalation_rates, color=colors,
                  edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, escalation_rates):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f"{val:.1%}", ha="center", fontsize=11, fontweight="bold")

    ax.set_ylabel("Escalation Rate (fraction requiring analyst review)")
    ax.set_title("Alert Fatigue Reduction: Escalation Rate by Model")
    ax.set_ylim(0, 1.25)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50% line")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {save_path}")


# ---------------------------------------------------------------------------
# 4. Cost analysis
# ---------------------------------------------------------------------------
def plot_cost_analysis(
    results_list: List[Dict[str, Any]],
    baseline_cost: float = 0.0,
    save_path: Optional[str] = None,
) -> None:
    """Compare total misclassification costs across models vs. baseline."""
    _ensure_output_dir()
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "cost_analysis.png")

    model_names = ["Baseline\n(all TP)"] + [r["model_name"] for r in results_list]
    costs = [baseline_cost] + [r["total_cost"] for r in results_list]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#d9534f"] + [PALETTE[i] for i in range(len(results_list))]
    bars = ax.barh(model_names, costs, color=colors,
                   edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, costs):
        ax.text(val + max(costs)*0.01, bar.get_y() + bar.get_height()/2,
                f"{val:,.0f}", va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Total Misclassification Cost")
    ax.set_title("Cost-Sensitive Evaluation: Total Cost by Model")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {save_path}")


# ---------------------------------------------------------------------------
# 5. SHAP feature importance
# ---------------------------------------------------------------------------
def plot_shap_importance(
    shap_values: np.ndarray,
    feature_names: List[str],
    class_idx: int = 2,
    top_k: int = 15,
    save_path: Optional[str] = None,
) -> None:
    """Bar plot of mean |SHAP| values for a given class (default: True Positive)."""
    _ensure_output_dir()
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "shap_importance.png")

    # shap_values shape: (n_samples, n_features) for the target class
    if isinstance(shap_values, list):
        vals = shap_values[class_idx]
    elif shap_values.ndim == 3:
        vals = shap_values[:, :, class_idx]
    else:
        vals = shap_values

    mean_abs = np.abs(vals).mean(axis=0)
    # If feature_names includes anomaly_score appended column
    if len(feature_names) < len(mean_abs):
        feature_names = list(feature_names) + ["anomaly_score"]

    indices = np.argsort(mean_abs)[-top_k:]
    top_features = [feature_names[i] for i in indices]
    top_values = mean_abs[indices]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_features, top_values, color=PALETTE[3],
            edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title(f"Top-{top_k} Feature Importance (Class: {CLASS_NAMES[class_idx]})")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {save_path}")


# ---------------------------------------------------------------------------
# 6. Combined summary figure (for LaTeX report)
# ---------------------------------------------------------------------------
def plot_summary_figure(
    results_list: List[Dict[str, Any]],
    shap_values: np.ndarray,
    feature_names: List[str],
    baseline_cost: float = 0.0,
    save_path: Optional[str] = None,
) -> None:
    """2x2 summary figure combining key visualisations."""
    _ensure_output_dir()
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "summary_figure.png")

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # (0,0) F-beta comparison
    ax1 = fig.add_subplot(gs[0, 0])
    names = [r["model_name"] for r in results_list]
    fbetas = [r["fbeta_macro"] for r in results_list]
    bars = ax1.bar(names, fbetas, color=PALETTE[:len(names)],
                   edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, fbetas):
        ax1.text(b.get_x() + b.get_width()/2, v + 0.02,
                 f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax1.set_ylabel(r"$F_{\beta=2}$ (macro)")
    ax1.set_title(r"(a) $F_{\beta=2}$ Score Comparison")
    ax1.set_ylim(0, 1)

    # (0,1) Confusion matrix of best model
    ax2 = fig.add_subplot(gs[0, 1])
    best = max(results_list, key=lambda r: r["fbeta_macro"])
    cm = best["confusion_matrix"]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="YlOrRd",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax2, linewidths=0.5)
    ax2.set_title(f"(b) Confusion Matrix ({best['model_name']})")
    ax2.set_ylabel("True Label")
    ax2.set_xlabel("Predicted Label")

    # (1,0) Cost comparison
    ax3 = fig.add_subplot(gs[1, 0])
    cost_names = ["Baseline"] + names
    costs = [baseline_cost] + [r["total_cost"] for r in results_list]
    c_colors = ["#d9534f"] + [PALETTE[i] for i in range(len(names))]
    bars3 = ax3.barh(cost_names, costs, color=c_colors,
                     edgecolor="black", linewidth=0.5)
    for b, v in zip(bars3, costs):
        ax3.text(v + max(costs)*0.01, b.get_y() + b.get_height()/2,
                 f"{v:,.0f}", va="center", fontsize=9)
    ax3.set_xlabel("Total Cost")
    ax3.set_title("(c) Misclassification Cost")
    ax3.invert_yaxis()

    # (1,1) SHAP importance
    ax4 = fig.add_subplot(gs[1, 1])
    if isinstance(shap_values, list):
        vals = shap_values[2]
    elif shap_values.ndim == 3:
        vals = shap_values[:, :, 2]
    else:
        vals = shap_values
    mean_abs = np.abs(vals).mean(axis=0)
    fn = list(feature_names)
    if len(fn) < len(mean_abs):
        fn = fn + ["anomaly_score"]
    top_k = min(10, len(fn))
    idx = np.argsort(mean_abs)[-top_k:]
    ax4.barh([fn[i] for i in idx], mean_abs[idx],
             color=PALETTE[3], edgecolor="black", linewidth=0.5)
    ax4.set_xlabel("Mean |SHAP Value|")
    ax4.set_title("(d) SHAP Feature Importance (True Positive)")

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {save_path}")
