"""
visualize.py - Plotting Utilities for Log Anomaly Detection
============================================================
Produces four publication-quality figures:
    1. model_comparison.png   - Grouped bar chart (F1, Accuracy, Precision, Recall)
    2. confusion_matrix.png   - Heatmap for the best-performing model
    3. latency_comparison.png - Horizontal bar chart of train / predict latency
    4. embedding_tsne.png     - 2-D t-SNE scatter of log embeddings coloured by label

All figures are saved to *output_dir* and are sized for inclusion in an
IEEEtran two-column layout.

Author : Tejas Vijay Mariyappagoudar (x24213829)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


# Global style
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})

COLOURS = sns.color_palette("muted", n_colors=10)


# ---------------------------------------------------------------------------
# 1. Model comparison bar chart
# ---------------------------------------------------------------------------

def plot_model_comparison(results: list, output_dir: str = "figures") -> str:
    """
    Grouped bar chart comparing Accuracy, Precision, Recall, F1 across models.

    Parameters
    ----------
    results : list of ModelResult objects (from model.py)
    output_dir : directory to save the figure

    Returns
    -------
    path : str  - absolute path of saved figure
    """
    os.makedirs(output_dir, exist_ok=True)

    model_names = [r.name for r in results]
    metrics = {
        "Accuracy":  [r.accuracy for r in results],
        "Precision": [r.precision for r in results],
        "Recall":    [r.recall for r in results],
        "F1 Score":  [r.f1 for r in results],
    }

    x = np.arange(len(model_names))
    n_metrics = len(metrics)
    bar_width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, (metric_name, values) in enumerate(metrics.items()):
        offset = (idx - n_metrics / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, values, bar_width, label=metric_name,
                      color=COLOURS[idx], edgecolor="white", linewidth=0.5)
        # Value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha="right")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right")
    fig.tight_layout()

    path = os.path.join(output_dir, "model_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return os.path.abspath(path)


# ---------------------------------------------------------------------------
# 2. Confusion matrix heatmap
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    result,
    output_dir: str = "figures",
    filename: str = "confusion_matrix.png",
) -> str:
    """
    Heatmap of the confusion matrix for a single ModelResult.
    """
    os.makedirs(output_dir, exist_ok=True)

    cm = result.confusion
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
        ax=ax,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix: {result.name}")
    fig.tight_layout()

    path = os.path.join(output_dir, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return os.path.abspath(path)


# ---------------------------------------------------------------------------
# 3. Latency comparison
# ---------------------------------------------------------------------------

def plot_latency_comparison(results: list, output_dir: str = "figures") -> str:
    """
    Horizontal stacked bar chart showing training and prediction latency
    for each model.
    """
    os.makedirs(output_dir, exist_ok=True)

    model_names = [r.name for r in results]
    train_times = [r.train_time_sec for r in results]
    pred_times  = [r.predict_time_sec for r in results]

    y = np.arange(len(model_names))
    bar_height = 0.5

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(y, train_times, bar_height, label="Training", color=COLOURS[0],
            edgecolor="white", linewidth=0.5)
    ax.barh(y, pred_times, bar_height, left=train_times, label="Prediction",
            color=COLOURS[1], edgecolor="white", linewidth=0.5)

    # Annotate total
    for i in range(len(model_names)):
        total = train_times[i] + pred_times[i]
        ax.text(total + 0.005, y[i], f"{total:.3f}s", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(model_names)
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Latency Comparison (CPU-Only)")
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()

    path = os.path.join(output_dir, "latency_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return os.path.abspath(path)


# ---------------------------------------------------------------------------
# 4. t-SNE embedding visualisation
# ---------------------------------------------------------------------------

def plot_embedding_tsne(
    X: np.ndarray,
    labels: np.ndarray,
    title: str = "t-SNE of Simulated Transformer Embeddings",
    output_dir: str = "figures",
    filename: str = "embedding_tsne.png",
    perplexity: int = 30,
    seed: int = 42,
) -> str:
    """
    2-D t-SNE scatter plot coloured by normal / anomaly label.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Sub-sample for speed if dataset is large
    max_points = 3000
    if X.shape[0] > max_points:
        rng = np.random.RandomState(seed)
        idx = rng.choice(X.shape[0], max_points, replace=False)
        X_sub = X[idx]
        labels_sub = labels[idx]
    else:
        X_sub = X
        labels_sub = labels

    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, X_sub.shape[0] - 1),
        random_state=seed,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
    )
    coords = tsne.fit_transform(X_sub)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # Normal
    mask_n = labels_sub == 0
    ax.scatter(coords[mask_n, 0], coords[mask_n, 1],
               s=8, alpha=0.5, c=[COLOURS[0]], label="Normal", rasterized=True)
    # Anomaly
    mask_a = labels_sub == 1
    ax.scatter(coords[mask_a, 0], coords[mask_a, 1],
               s=14, alpha=0.7, c=[COLOURS[3]], label="Anomaly",
               marker="x", linewidths=0.8, rasterized=True)

    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_title(title)
    ax.legend(markerscale=2)
    fig.tight_layout()

    path = os.path.join(output_dir, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return os.path.abspath(path)


# ---------------------------------------------------------------------------
# Generate all figures at once
# ---------------------------------------------------------------------------

def generate_all_figures(
    results: list,
    X_transformer: np.ndarray,
    labels: np.ndarray,
    output_dir: str = "figures",
) -> dict:
    """
    Convenience function called from main.py to produce every figure.

    Returns a dict mapping figure name -> absolute path.
    """
    paths = {}
    print("\nGenerating figures ...")

    paths["model_comparison"] = plot_model_comparison(results, output_dir)

    # Confusion matrix for the best model by F1
    best = max(results, key=lambda r: r.f1)
    paths["confusion_matrix"] = plot_confusion_matrix(best, output_dir)

    paths["latency_comparison"] = plot_latency_comparison(results, output_dir)

    paths["embedding_tsne"] = plot_embedding_tsne(
        X_transformer, labels, output_dir=output_dir,
    )

    return paths
