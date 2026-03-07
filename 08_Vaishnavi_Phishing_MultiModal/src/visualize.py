"""
visualize.py - Visualization Module for Multi-Modal Phishing Detection
=======================================================================
Project 8: Explainable Multi-Modal Phishing Detection
Student: Vaishnavi Purohit (24260339)

Plotting functions:
  - plot_model_comparison: bar chart comparing all model metrics
  - plot_modality_importance: SHAP-based modality contribution
  - plot_confusion_matrix: heatmap confusion matrix
  - plot_xai_comparison: side-by-side SHAP vs LIME feature importance
  - plot_shap_summary: SHAP beeswarm/bar summary
  - plot_consistency: visual comparison of SHAP-LIME rankings
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import feature name lists for modality coloring
try:
    from data_loader import URL_FEATURE_NAMES, CONTENT_FEATURE_NAMES, EXTERNAL_FEATURE_NAMES
except ImportError:
    URL_FEATURE_NAMES = []
    CONTENT_FEATURE_NAMES = []
    EXTERNAL_FEATURE_NAMES = []

_URL_SET = set(URL_FEATURE_NAMES)
_CONTENT_SET = set(CONTENT_FEATURE_NAMES)
_EXTERNAL_SET = set(EXTERNAL_FEATURE_NAMES)


def _get_modality_color(feat_name: str) -> str:
    """Return color based on which modality a feature belongs to."""
    if feat_name in _URL_SET:
        return "#4C72B0"   # blue for URL
    elif feat_name in _CONTENT_SET:
        return "#55A868"   # green for Content
    else:
        return "#C44E52"   # red for External


# Global style
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})


def _save_fig(fig, filepath: str) -> None:
    """Save figure and close."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [Saved] {filepath}")


# ---------------------------------------------------------------------------
# 1. Model comparison
# ---------------------------------------------------------------------------
def plot_model_comparison(all_metrics: list, output_dir: str) -> str:
    """Bar chart comparing Accuracy, Precision, Recall, F1, AUC across
    all models."""
    models = [m["model"] for m in all_metrics]
    metric_keys = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]

    data = []
    for m in all_metrics:
        for key, label in zip(metric_keys, metric_labels):
            val = m.get(key)
            if val is not None:
                data.append({"Model": m["model"], "Metric": label,
                             "Value": val})

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(12, 5))
    palette = sns.color_palette("Set2", n_colors=len(metric_labels))

    x = np.arange(len(models))
    width = 0.15

    for i, (metric_label, color) in enumerate(zip(metric_labels, palette)):
        subset = df[df["Metric"] == metric_label]
        vals = [subset[subset["Model"] == m]["Value"].values[0]
                if len(subset[subset["Model"] == m]) > 0 else 0
                for m in models]
        bars = ax.bar(x + i * width, vals, width, label=metric_label,
                      color=color, edgecolor="gray", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=6,
                    rotation=45)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.12)
    ax.set_title("Model Performance Comparison: Single-Modal vs. Late Fusion")
    ax.legend(loc="upper left", fontsize=8, ncol=len(metric_labels))
    ax.grid(axis="y", alpha=0.3)

    filepath = os.path.join(output_dir, "model_comparison.png")
    _save_fig(fig, filepath)
    return filepath


# ---------------------------------------------------------------------------
# 2. Modality importance
# ---------------------------------------------------------------------------
def plot_modality_importance(shap_mod_imp: pd.DataFrame,
                            lime_mod_imp: pd.DataFrame,
                            output_dir: str) -> str:
    """Side-by-side bar chart of modality importance from SHAP and LIME."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # SHAP modality importance
    ax = axes[0]
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    bars = ax.barh(shap_mod_imp["modality"], shap_mod_imp["total_abs_shap"],
                   color=colors[:len(shap_mod_imp)], edgecolor="gray")
    ax.set_xlabel("Total |SHAP| (mean per sample)")
    ax.set_title("SHAP Modality Importance")
    for bar, v in zip(bars, shap_mod_imp["total_abs_shap"]):
        ax.text(v + 0.001, bar.get_y() + bar.get_height()/2,
                f"{v:.4f}", va="center", fontsize=9)
    ax.invert_yaxis()

    # LIME modality importance
    ax = axes[1]
    bars = ax.barh(lime_mod_imp["modality"], lime_mod_imp["total_abs_lime"],
                   color=colors[:len(lime_mod_imp)], edgecolor="gray")
    ax.set_xlabel("Total |LIME weight| (mean per sample)")
    ax.set_title("LIME Modality Importance")
    for bar, v in zip(bars, lime_mod_imp["total_abs_lime"]):
        ax.text(v + 0.001, bar.get_y() + bar.get_height()/2,
                f"{v:.4f}", va="center", fontsize=9)
    ax.invert_yaxis()

    fig.suptitle("Which Modality Contributes Most to Phishing Detection?",
                 fontsize=13, y=1.02)
    fig.tight_layout()

    filepath = os.path.join(output_dir, "modality_importance.png")
    _save_fig(fig, filepath)
    return filepath


# ---------------------------------------------------------------------------
# 3. Confusion matrices
# ---------------------------------------------------------------------------
def plot_confusion_matrix(all_metrics: list, output_dir: str) -> str:
    """Grid of confusion matrix heatmaps for all models."""
    n = len(all_metrics)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, m in enumerate(all_metrics):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        cm = m["confusion_matrix"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Legit", "Phish"],
                    yticklabels=["Legit", "Phish"],
                    cbar=False, linewidths=0.5)
        ax.set_title(m["model"], fontsize=9)
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")

    # Hide unused subplots
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].set_visible(False)

    fig.suptitle("Confusion Matrices", fontsize=13)
    fig.tight_layout()

    filepath = os.path.join(output_dir, "confusion_matrices.png")
    _save_fig(fig, filepath)
    return filepath


# ---------------------------------------------------------------------------
# 4. XAI comparison (SHAP vs LIME feature importance)
# ---------------------------------------------------------------------------
def plot_xai_comparison(shap_feat_imp: pd.DataFrame,
                        lime_feat_imp: pd.DataFrame,
                        output_dir: str, top_k: int = 15) -> str:
    """Side-by-side horizontal bar charts of SHAP vs LIME top-k features."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # SHAP top-k
    ax = axes[0]
    shap_top = shap_feat_imp.head(top_k).iloc[::-1]
    colors = [_get_modality_color(f) for f in shap_top["feature"]]
    ax.barh(shap_top["feature"], shap_top["mean_abs_shap"],
            color=colors, edgecolor="gray", linewidth=0.5)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"SHAP Top-{top_k} Features")
    ax.tick_params(axis="y", labelsize=8)

    # LIME top-k
    ax = axes[1]
    lime_top = lime_feat_imp.head(top_k).iloc[::-1]
    colors = [_get_modality_color(f) for f in lime_top["feature"]]
    ax.barh(lime_top["feature"], lime_top["mean_abs_lime"],
            color=colors, edgecolor="gray", linewidth=0.5)
    ax.set_xlabel("Mean |LIME weight|")
    ax.set_title(f"LIME Top-{top_k} Features")
    ax.tick_params(axis="y", labelsize=8)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4C72B0", label="URL"),
        Patch(facecolor="#55A868", label="Content"),
        Patch(facecolor="#C44E52", label="External"),
    ]
    fig.legend(handles=legend_elements, loc="upper center",
               ncol=3, fontsize=9, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("SHAP vs LIME Feature Importance Comparison",
                 fontsize=13, y=1.06)
    fig.tight_layout()

    filepath = os.path.join(output_dir, "xai_comparison.png")
    _save_fig(fig, filepath)
    return filepath


# ---------------------------------------------------------------------------
# 5. SHAP summary (beeswarm-style bar plot)
# ---------------------------------------------------------------------------
def plot_shap_summary(shap_values: np.ndarray, X: np.ndarray,
                      feature_names: list, output_dir: str) -> str:
    """SHAP bar summary plot for the fusion model."""
    fig, ax = plt.subplots(figsize=(8, 7))

    # Mean absolute SHAP
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    sorted_idx = np.argsort(mean_abs)

    colors = [_get_modality_color(feature_names[i]) for i in sorted_idx]
    ax.barh([feature_names[i] for i in sorted_idx],
            mean_abs[sorted_idx], color=colors, edgecolor="gray",
            linewidth=0.3)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("SHAP Feature Importance (All Features)")
    ax.tick_params(axis="y", labelsize=7)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4C72B0", label="URL"),
        Patch(facecolor="#55A868", label="Content"),
        Patch(facecolor="#C44E52", label="External"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    fig.tight_layout()

    filepath = os.path.join(output_dir, "shap_summary.png")
    _save_fig(fig, filepath)
    return filepath


# ---------------------------------------------------------------------------
# 6. Consistency visualization
# ---------------------------------------------------------------------------
def plot_consistency(shap_feat_imp: pd.DataFrame,
                     lime_feat_imp: pd.DataFrame,
                     consistency: dict,
                     output_dir: str) -> str:
    """Scatter plot of SHAP rank vs LIME rank for all features,
    highlighting top-k shared features."""
    # Merge SHAP and LIME rankings
    shap_ranked = shap_feat_imp.copy()
    shap_ranked["shap_rank"] = range(1, len(shap_ranked) + 1)

    lime_ranked = lime_feat_imp.copy()
    lime_ranked["lime_rank"] = range(1, len(lime_ranked) + 1)

    merged = shap_ranked[["feature", "shap_rank"]].merge(
        lime_ranked[["feature", "lime_rank"]], on="feature"
    )

    top_k = consistency["top_k"]
    shared = set(consistency["shared_features"])

    fig, ax = plt.subplots(figsize=(7, 6))

    # All features
    ax.scatter(merged["shap_rank"], merged["lime_rank"],
               alpha=0.4, color="gray", s=30, label="Other features")

    # Shared top-k features
    shared_mask = merged["feature"].isin(shared)
    ax.scatter(merged.loc[shared_mask, "shap_rank"],
               merged.loc[shared_mask, "lime_rank"],
               color="red", s=60, zorder=5,
               label=f"Shared top-{top_k}")

    # Annotate shared features
    for _, row in merged[shared_mask].iterrows():
        ax.annotate(row["feature"], (row["shap_rank"], row["lime_rank"]),
                    fontsize=6, xytext=(3, 3),
                    textcoords="offset points")

    # Perfect agreement line
    max_rank = max(merged["shap_rank"].max(), merged["lime_rank"].max())
    ax.plot([1, max_rank], [1, max_rank], "k--", alpha=0.3,
            label="Perfect agreement")

    ax.set_xlabel("SHAP Rank")
    ax.set_ylabel("LIME Rank")
    ax.set_title(f"SHAP vs LIME Rank Agreement\n"
                 f"Jaccard={consistency['jaccard_index']:.3f}, "
                 f"Spearman r={consistency['spearman_correlation']:.3f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    fig.tight_layout()

    filepath = os.path.join(output_dir, "rank_consistency.png")
    _save_fig(fig, filepath)
    return filepath


# ---------------------------------------------------------------------------
# Master plotting function
# ---------------------------------------------------------------------------
def generate_all_plots(all_metrics: list, xai_results: dict,
                       X_test: np.ndarray, feature_names: list,
                       output_dir: str) -> dict:
    """Generate all plots and return dict of file paths."""
    print("\n" + "="*60)
    print("  GENERATING VISUALIZATIONS")
    print("="*60)

    paths = {}
    paths["model_comparison"] = plot_model_comparison(all_metrics, output_dir)
    paths["modality_importance"] = plot_modality_importance(
        xai_results["shap_modality_importance"],
        xai_results["lime_modality_importance"],
        output_dir,
    )
    paths["confusion_matrices"] = plot_confusion_matrix(all_metrics, output_dir)
    paths["xai_comparison"] = plot_xai_comparison(
        xai_results["shap_feature_importance"],
        xai_results["lime_feature_importance"],
        output_dir,
    )
    paths["shap_summary"] = plot_shap_summary(
        xai_results["shap_values"], X_test, feature_names, output_dir
    )
    paths["rank_consistency"] = plot_consistency(
        xai_results["shap_feature_importance"],
        xai_results["lime_feature_importance"],
        xai_results["consistency"],
        output_dir,
    )

    print(f"\n  All {len(paths)} figures saved to {output_dir}/")
    return paths
