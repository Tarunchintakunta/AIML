"""
explainers.py - Explainable AI for Multi-Modal Phishing Detection
=================================================================
Project 8: Explainable Multi-Modal Phishing Detection
Student: Vaishnavi Purohit (24260339)

Implements:
  - SHAP TreeExplainer for tree-based models
  - LIME tabular explanations
  - Modality importance analysis (which modality contributes most)
  - SHAP-LIME consistency index

References:
  Al-Subaiey et al. 2024; Alhuzali et al. 2025; Patra et al. 2025
"""

import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
from typing import Optional


# ---------------------------------------------------------------------------
# SHAP explanations
# ---------------------------------------------------------------------------
class SHAPExplainer:
    """SHAP TreeExplainer wrapper for tree-based fusion models."""

    def __init__(self, model, feature_names: list,
                 modality_ranges: Optional[dict] = None):
        """
        Parameters
        ----------
        model : fitted tree-based model (RF, XGB, LGBM) with .predict method
        feature_names : list of all feature names
        modality_ranges : dict mapping modality name to (start, end) column
                          indices. Used for modality importance.
        """
        self.model = model
        self.feature_names = feature_names
        self.modality_ranges = modality_ranges or {
            "text": (0, 20),
            "url": (20, 30),
            "temporal": (30, 35),
        }
        # Create SHAP explainer
        self.explainer = shap.TreeExplainer(model)
        self.shap_values = None

    def compute_shap_values(self, X: np.ndarray) -> np.ndarray:
        """Compute SHAP values for the given data.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            SHAP values for the positive class (phishing).
        """
        sv = self.explainer.shap_values(X)

        # Handle different SHAP output formats
        if isinstance(sv, list):
            # For RF: list of [class0_shap, class1_shap]
            self.shap_values = sv[1]
        elif isinstance(sv, np.ndarray) and sv.ndim == 3:
            self.shap_values = sv[:, :, 1]
        else:
            self.shap_values = sv

        return self.shap_values

    def get_feature_importance(self) -> pd.DataFrame:
        """Return mean absolute SHAP values per feature."""
        if self.shap_values is None:
            raise ValueError("Call compute_shap_values() first.")
        mean_abs = np.mean(np.abs(self.shap_values), axis=0)
        df = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        return df

    def get_modality_importance(self) -> pd.DataFrame:
        """Aggregate SHAP values by modality."""
        if self.shap_values is None:
            raise ValueError("Call compute_shap_values() first.")

        results = []
        total_abs = np.mean(np.abs(self.shap_values))
        for mod_name, (start, end) in self.modality_ranges.items():
            mod_shap = self.shap_values[:, start:end]
            mod_mean = np.mean(np.abs(mod_shap))
            n_feats = end - start
            results.append({
                "modality": mod_name,
                "mean_abs_shap": mod_mean,
                "total_abs_shap": np.sum(np.abs(mod_shap)) / len(self.shap_values),
                "n_features": n_feats,
                "normalized_importance": mod_mean / total_abs if total_abs > 0 else 0,
            })

        df = pd.DataFrame(results).sort_values(
            "total_abs_shap", ascending=False
        ).reset_index(drop=True)
        return df

    def explain_instance(self, x: np.ndarray, top_k: int = 10) -> pd.DataFrame:
        """Explain a single prediction with SHAP.

        Parameters
        ----------
        x : 1D array for one sample
        top_k : number of top features to return
        """
        sv = self.explainer.shap_values(x.reshape(1, -1))
        if isinstance(sv, list):
            vals = sv[1][0]
        elif isinstance(sv, np.ndarray) and sv.ndim == 3:
            vals = sv[0, :, 1]
        else:
            vals = sv[0]

        df = pd.DataFrame({
            "feature": self.feature_names,
            "shap_value": vals,
            "abs_shap": np.abs(vals),
        }).sort_values("abs_shap", ascending=False).head(top_k)
        return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# LIME explanations
# ---------------------------------------------------------------------------
class LIMEExplainer:
    """LIME tabular explainer for phishing detection models."""

    def __init__(self, model, X_train: np.ndarray,
                 feature_names: list,
                 modality_ranges: Optional[dict] = None):
        """
        Parameters
        ----------
        model : fitted model with predict_proba method
        X_train : training data for background distribution
        feature_names : list of all feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.modality_ranges = modality_ranges or {
            "text": (0, 20),
            "url": (20, 30),
            "temporal": (30, 35),
        }
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=feature_names,
            class_names=["Legitimate", "Phishing"],
            mode="classification",
            random_state=42,
        )

    def explain_instance(self, x: np.ndarray, num_features: int = 10,
                         num_samples: int = 1000) -> dict:
        """Explain a single prediction with LIME.

        Returns
        -------
        dict with 'explanation' (LIME object), 'feature_weights' (DataFrame)
        """
        exp = self.explainer.explain_instance(
            x, self.model.predict_proba,
            num_features=num_features,
            num_samples=num_samples,
        )

        # Extract feature weights
        feat_weights = exp.as_list()
        df = pd.DataFrame(feat_weights, columns=["feature_rule", "weight"])
        df["abs_weight"] = df["weight"].abs()
        df = df.sort_values("abs_weight", ascending=False).reset_index(drop=True)

        return {"explanation": exp, "feature_weights": df}

    def batch_explain(self, X: np.ndarray, n_instances: int = 50,
                      num_features: int = 35,
                      num_samples: int = 500) -> np.ndarray:
        """Compute LIME importance for multiple instances.

        Returns array of shape (n_instances, n_features) with signed weights.
        """
        n = min(n_instances, len(X))
        importance_matrix = np.zeros((n, len(self.feature_names)))

        for i in range(n):
            exp = self.explainer.explain_instance(
                X[i], self.model.predict_proba,
                num_features=num_features,
                num_samples=num_samples,
            )
            # Map feature index to weight
            feat_map = exp.local_exp.get(1, exp.local_exp.get(0, []))
            for feat_idx, weight in feat_map:
                if feat_idx < len(self.feature_names):
                    importance_matrix[i, feat_idx] = weight

        return importance_matrix

    def get_feature_importance(self, X: np.ndarray,
                               n_instances: int = 50,
                               _precomputed: np.ndarray = None) -> pd.DataFrame:
        """Average absolute LIME weights across instances."""
        imp = _precomputed if _precomputed is not None else self.batch_explain(X, n_instances=n_instances)
        mean_abs = np.mean(np.abs(imp), axis=0)
        df = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_lime": mean_abs,
        }).sort_values("mean_abs_lime", ascending=False).reset_index(drop=True)
        return df

    def get_modality_importance(self, X: np.ndarray,
                                n_instances: int = 50,
                                _precomputed: np.ndarray = None) -> pd.DataFrame:
        """Aggregate LIME importance by modality."""
        imp = _precomputed if _precomputed is not None else self.batch_explain(X, n_instances=n_instances)
        mean_abs_all = np.mean(np.abs(imp))

        results = []
        for mod_name, (start, end) in self.modality_ranges.items():
            mod_imp = imp[:, start:end]
            mod_mean = np.mean(np.abs(mod_imp))
            results.append({
                "modality": mod_name,
                "mean_abs_lime": mod_mean,
                "total_abs_lime": np.sum(np.abs(mod_imp)) / len(imp),
                "n_features": end - start,
                "normalized_importance": mod_mean / mean_abs_all if mean_abs_all > 0 else 0,
            })

        df = pd.DataFrame(results).sort_values(
            "total_abs_lime", ascending=False
        ).reset_index(drop=True)
        return df


# ---------------------------------------------------------------------------
# Consistency between SHAP and LIME
# ---------------------------------------------------------------------------
def compute_consistency_index(shap_importance: pd.DataFrame,
                              lime_importance: pd.DataFrame,
                              top_k: int = 15) -> dict:
    """Compute agreement between SHAP and LIME top-k feature rankings.

    Metrics:
      - Jaccard index: overlap of top-k feature sets
      - Rank correlation: Spearman correlation of shared features' ranks
      - Modality agreement: whether both methods agree on modality ordering

    Parameters
    ----------
    shap_importance : DataFrame with columns ['feature', 'mean_abs_shap']
    lime_importance : DataFrame with columns ['feature', 'mean_abs_lime']
    top_k : number of top features to compare

    Returns
    -------
    dict with consistency metrics
    """
    shap_top = set(shap_importance.head(top_k)["feature"].values)
    lime_top = set(lime_importance.head(top_k)["feature"].values)

    # Jaccard index
    intersection = shap_top & lime_top
    union = shap_top | lime_top
    jaccard = len(intersection) / len(union) if len(union) > 0 else 0.0

    # Rank correlation on overlapping features
    if len(intersection) > 1:
        shap_ranks = {f: i for i, f in enumerate(
            shap_importance["feature"].values)}
        lime_ranks = {f: i for i, f in enumerate(
            lime_importance["feature"].values)}
        shared = sorted(intersection)
        shap_r = np.array([shap_ranks[f] for f in shared], dtype=float)
        lime_r = np.array([lime_ranks[f] for f in shared], dtype=float)

        # Spearman correlation
        from scipy.stats import spearmanr
        corr, p_val = spearmanr(shap_r, lime_r)
    else:
        corr = float("nan")
        p_val = float("nan")

    results = {
        "top_k": top_k,
        "jaccard_index": jaccard,
        "overlap_count": len(intersection),
        "spearman_correlation": corr,
        "spearman_p_value": p_val,
        "shap_top_features": sorted(shap_top),
        "lime_top_features": sorted(lime_top),
        "shared_features": sorted(intersection),
    }

    return results


def print_consistency_report(consistency: dict) -> None:
    """Pretty-print the SHAP-LIME consistency report."""
    print("\n" + "="*60)
    print("  SHAP-LIME CONSISTENCY REPORT")
    print("="*60)
    print(f"  Top-K             : {consistency['top_k']}")
    print(f"  Jaccard Index     : {consistency['jaccard_index']:.4f}")
    print(f"  Overlap Count     : {consistency['overlap_count']}")
    print(f"  Spearman Corr     : {consistency['spearman_correlation']:.4f}")
    print(f"  Spearman p-value  : {consistency['spearman_p_value']:.4f}")
    print(f"\n  Shared top features:")
    for f in consistency['shared_features']:
        print(f"    - {f}")


# ---------------------------------------------------------------------------
# Full XAI pipeline
# ---------------------------------------------------------------------------
def run_xai_pipeline(model, X_train: np.ndarray, X_test: np.ndarray,
                     feature_names: list,
                     n_lime_instances: int = 50,
                     shap_background_size: int = 200) -> dict:
    """Run complete XAI analysis: SHAP + LIME + consistency.

    Parameters
    ----------
    model : fitted tree-based model (must have .predict_proba)
    X_train : training features
    X_test : test features (used for SHAP values)
    feature_names : list of feature names
    n_lime_instances : number of instances for LIME batch
    shap_background_size : number of background samples for SHAP

    Returns
    -------
    dict with SHAP explainer, LIME explainer, importances, consistency
    """
    print("\n" + "="*60)
    print("  RUNNING XAI PIPELINE")
    print("="*60)

    # ---- SHAP ----
    print("\n[XAI] Computing SHAP values...")
    # Use underlying model if wrapped in our class
    raw_model = model.model if hasattr(model, "model") else model
    shap_exp = SHAPExplainer(raw_model, feature_names)

    # Use a subsample for SHAP to keep it tractable
    n_shap = min(shap_background_size, len(X_test))
    shap_exp.compute_shap_values(X_test[:n_shap])
    shap_feat_imp = shap_exp.get_feature_importance()
    shap_mod_imp = shap_exp.get_modality_importance()

    print("\n  SHAP Feature Importance (top 10):")
    print(shap_feat_imp.head(10).to_string(index=False))
    print("\n  SHAP Modality Importance:")
    print(shap_mod_imp.to_string(index=False))

    # ---- LIME ----
    print(f"\n[XAI] Computing LIME explanations ({n_lime_instances} instances)...")
    predict_fn = model if hasattr(model, "predict_proba") else model.model
    lime_exp = LIMEExplainer(predict_fn, X_train, feature_names)
    # Compute LIME batch once, reuse for both feature and modality importance
    lime_batch = lime_exp.batch_explain(X_test, n_instances=n_lime_instances)
    lime_feat_imp = lime_exp.get_feature_importance(
        X_test, n_instances=n_lime_instances, _precomputed=lime_batch
    )
    lime_mod_imp = lime_exp.get_modality_importance(
        X_test, n_instances=n_lime_instances, _precomputed=lime_batch
    )

    print("\n  LIME Feature Importance (top 10):")
    print(lime_feat_imp.head(10).to_string(index=False))
    print("\n  LIME Modality Importance:")
    print(lime_mod_imp.to_string(index=False))

    # ---- Consistency ----
    consistency = compute_consistency_index(shap_feat_imp, lime_feat_imp)
    print_consistency_report(consistency)

    return {
        "shap_explainer": shap_exp,
        "lime_explainer": lime_exp,
        "shap_feature_importance": shap_feat_imp,
        "lime_feature_importance": lime_feat_imp,
        "shap_modality_importance": shap_mod_imp,
        "lime_modality_importance": lime_mod_imp,
        "shap_values": shap_exp.shap_values,
        "consistency": consistency,
    }
