"""
XAI module: SHAP and LIME explainers with consistency and fidelity metrics.
"""

import time
import numpy as np
from scipy.stats import kendalltau
import shap
import lime
import lime.lime_tabular


def compute_shap_importance(model, X_train, X_explain, feature_names,
                            model_name='Model'):
    """Compute SHAP feature importance values."""
    print(f"    Computing SHAP for {model_name}...")
    start = time.time()

    if model_name in ['Random Forest', 'XGBoost', 'LightGBM']:
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(
            model.predict_proba, shap.sample(X_train, 100)
        )

    shap_values = explainer.shap_values(X_explain)

    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        shap_vals = np.abs(shap_values[1])  # class 1 (attack)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_vals = np.abs(shap_values[:, :, 1])  # (samples, features, classes)
    else:
        shap_vals = np.abs(shap_values)

    # Ensure 2D: (n_samples, n_features)
    if shap_vals.ndim == 1:
        shap_vals = shap_vals.reshape(1, -1)

    # Trim to match input features if needed
    n_features = X_explain.shape[1]
    if shap_vals.shape[1] > n_features:
        shap_vals = shap_vals[:, :n_features]

    mean_importance = shap_vals.mean(axis=0)
    shap_time = time.time() - start

    print(f"    SHAP time: {shap_time:.2f}s")
    return mean_importance, shap_time, shap_vals


def compute_lime_importance(model, X_train, X_explain, feature_names,
                            model_name='Model'):
    """Compute LIME feature importance values."""
    print(f"    Computing LIME for {model_name}...")
    start = time.time()

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train, feature_names=feature_names,
        class_names=['Benign', 'Malicious'],
        mode='classification', random_state=42
    )

    n_explain = len(X_explain)
    importance_matrix = np.zeros((n_explain, len(feature_names)))

    for i in range(n_explain):
        exp = lime_explainer.explain_instance(
            X_explain[i], model.predict_proba,
            num_features=len(feature_names), num_samples=500
        )
        feature_map = dict(exp.as_list())
        for j, fname in enumerate(feature_names):
            for key, val in feature_map.items():
                if fname in key:
                    importance_matrix[i, j] = abs(val)
                    break

    mean_importance = importance_matrix.mean(axis=0)
    lime_time = time.time() - start

    print(f"    LIME time: {lime_time:.2f}s")
    return mean_importance, lime_time


def consistency_index(shap_importance, lime_importance):
    """
    Compute Consistency Index: Kendall's tau correlation between
    SHAP and LIME feature importance rankings.
    """
    shap_ranks = np.argsort(-shap_importance)
    lime_ranks = np.argsort(-lime_importance)

    tau, p_value = kendalltau(shap_ranks, lime_ranks)
    return tau, p_value


def explanation_fidelity(model, X_test, y_test, shap_importance,
                         n_remove_steps=5):
    """
    Compute Explanation Fidelity: correlation between feature importance
    and accuracy degradation when features are removed.
    """
    from sklearn.metrics import f1_score as f1_metric

    base_preds = model.predict(X_test)
    base_f1 = f1_metric(y_test, base_preds, zero_division=0)

    sorted_features = np.argsort(-shap_importance)
    n_features = len(shap_importance)
    step_size = max(1, n_features // n_remove_steps)

    degradations = []
    n_removed_list = []

    X_modified = X_test.copy()
    for step in range(1, n_remove_steps + 1):
        start_idx = (step - 1) * step_size
        end_idx = min(step * step_size, n_features)
        features_to_remove = sorted_features[start_idx:end_idx]

        for f_idx in features_to_remove:
            X_modified[:, f_idx] = 0.0

        preds = model.predict(X_modified)
        f1_now = f1_metric(y_test, preds, zero_division=0)
        degradation = base_f1 - f1_now
        degradations.append(degradation)
        n_removed_list.append(end_idx)

    # Fidelity = correlation between removal order and degradation
    if len(degradations) > 1 and np.std(degradations) > 0:
        correlation = np.corrcoef(n_removed_list, degradations)[0, 1]
    else:
        correlation = 0.0

    return correlation, degradations, n_removed_list


def computational_overhead_ratio(explain_time, predict_time):
    """Compute Computational Overhead Ratio: explanation time / prediction time."""
    if predict_time > 0:
        return explain_time / predict_time
    return float('inf')


def run_xai_analysis(model, X_train, X_test, y_test, feature_names,
                     model_name='Model', n_explain=100):
    """Run full XAI analysis for a model: SHAP, LIME, consistency, fidelity."""
    print(f"\n  XAI Analysis: {model_name}")
    print("  " + "-" * 50)

    # Use subset for explanation (computational efficiency)
    explain_idx = np.random.RandomState(42).choice(
        len(X_test), min(n_explain, len(X_test)), replace=False
    )
    X_explain = X_test[explain_idx]

    # SHAP
    shap_importance, shap_time, shap_vals = compute_shap_importance(
        model, X_train, X_explain, feature_names, model_name
    )

    # LIME
    lime_importance, lime_time = compute_lime_importance(
        model, X_train, X_explain, feature_names, model_name
    )

    # Consistency Index
    tau, p_val = consistency_index(shap_importance, lime_importance)
    print(f"    Consistency Index (Kendall's tau): {tau:.4f} (p={p_val:.4f})")

    # Explanation Fidelity
    fidelity, degradations, n_removed = explanation_fidelity(
        model, X_test, y_test, shap_importance
    )
    print(f"    Explanation Fidelity: {fidelity:.4f}")

    # Predict time for overhead ratio
    start = time.time()
    model.predict(X_explain)
    pred_time = time.time() - start

    shap_overhead = computational_overhead_ratio(shap_time, pred_time)
    lime_overhead = computational_overhead_ratio(lime_time, pred_time)
    print(f"    SHAP Overhead Ratio: {shap_overhead:.1f}x")
    print(f"    LIME Overhead Ratio: {lime_overhead:.1f}x")

    return {
        'shap_importance': shap_importance,
        'lime_importance': lime_importance,
        'shap_values': shap_vals,
        'shap_time': shap_time,
        'lime_time': lime_time,
        'consistency_tau': tau,
        'consistency_pval': p_val,
        'fidelity': fidelity,
        'fidelity_degradations': degradations,
        'fidelity_n_removed': n_removed,
        'shap_overhead': shap_overhead,
        'lime_overhead': lime_overhead,
    }
