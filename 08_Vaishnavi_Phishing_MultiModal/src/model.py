"""
model.py - Multi-Modal Phishing Detection Models
=================================================
Project 8: Explainable Multi-Modal Phishing Detection
Student: Vaishnavi Purohit (24260339)

Implements:
  - Single-modality baselines (Random Forest per modality)
  - Late Fusion with Random Forest (concatenated features)
  - Late Fusion with XGBoost
  - Late Fusion with LightGBM
  - Probability-level late fusion (average of per-modality probabilities)

References:
  Ghadami & Rahebi 2025; Mahmoud & Nallasivan 2025; Vulfin et al. 2026
"""

import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------
def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray,
                   y_prob: np.ndarray = None, model_name: str = "") -> dict:
    """Compute standard binary classification metrics."""
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
    else:
        metrics["auc_roc"] = None
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)
    return metrics


def print_metrics(m: dict) -> None:
    """Pretty-print evaluation metrics."""
    print(f"\n{'='*50}")
    print(f"  Model: {m['model']}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {m['accuracy']:.4f}")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  F1 Score  : {m['f1']:.4f}")
    if m['auc_roc'] is not None:
        print(f"  AUC-ROC   : {m['auc_roc']:.4f}")
    print(f"  Confusion Matrix:")
    cm = m['confusion_matrix']
    print(f"    TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
    print(f"    FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")


# ---------------------------------------------------------------------------
# Single-modality baselines
# ---------------------------------------------------------------------------
class SingleModalityModel:
    """Train and evaluate a Random Forest on one modality."""

    def __init__(self, modality_name: str, n_estimators: int = 200,
                 random_state: int = 42):
        self.modality_name = modality_name
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=12,
            min_samples_split=5,
            random_state=random_state,
            n_jobs=-1,
        )
        self.is_fitted = False

    def fit(self, X_train, y_train):
        t0 = time.time()
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        elapsed = time.time() - t0
        print(f"[SingleModal-{self.modality_name}] Trained in {elapsed:.2f}s")
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test) -> dict:
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)[:, 1]
        return evaluate_model(y_test, y_pred, y_prob,
                              f"RF-{self.modality_name}")


# ---------------------------------------------------------------------------
# Late Fusion models
# ---------------------------------------------------------------------------
class LateFusionRF:
    """Late fusion: concatenate all modality features, train a single RF."""

    def __init__(self, n_estimators: int = 300, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=15,
            min_samples_split=4,
            random_state=random_state,
            n_jobs=-1,
        )
        self.name = "LateFusion-RF"
        self.is_fitted = False

    def fit(self, X_train, y_train):
        t0 = time.time()
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        print(f"[{self.name}] Trained in {time.time()-t0:.2f}s")
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test) -> dict:
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)[:, 1]
        return evaluate_model(y_test, y_pred, y_prob, self.name)


class LateFusionXGB:
    """Late fusion with XGBoost."""

    def __init__(self, n_estimators: int = 300, random_state: int = 42):
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
            n_jobs=-1,
        )
        self.name = "LateFusion-XGB"
        self.is_fitted = False

    def fit(self, X_train, y_train):
        t0 = time.time()
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        print(f"[{self.name}] Trained in {time.time()-t0:.2f}s")
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test) -> dict:
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)[:, 1]
        return evaluate_model(y_test, y_pred, y_prob, self.name)


class LateFusionLGBM:
    """Late fusion with LightGBM."""

    def __init__(self, n_estimators: int = 300, random_state: int = 42):
        self.model = LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            verbose=-1,
            n_jobs=-1,
        )
        self.name = "LateFusion-LGBM"
        self.is_fitted = False

    def fit(self, X_train, y_train):
        t0 = time.time()
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        print(f"[{self.name}] Trained in {time.time()-t0:.2f}s")
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test) -> dict:
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)[:, 1]
        return evaluate_model(y_test, y_pred, y_prob, self.name)


class ProbabilityFusion:
    """Probability-level late fusion: average predicted probabilities
    from per-modality models, then threshold at 0.5."""

    def __init__(self):
        self.name = "ProbFusion-Avg"
        self.modality_models = {}
        self.is_fitted = False

    def fit(self, modality_models: dict):
        """Accept already-fitted SingleModalityModel instances.

        Parameters
        ----------
        modality_models : dict
            e.g. {'text': model, 'url': model, 'temporal': model}
        """
        self.modality_models = modality_models
        self.is_fitted = True
        return self

    def predict_proba_combined(self, X_dict: dict) -> np.ndarray:
        """Average probabilities across modalities.

        Parameters
        ----------
        X_dict : dict
            Keyed by modality name, values are feature arrays.
        """
        probs = []
        for mod_name, model in self.modality_models.items():
            p = model.predict_proba(X_dict[mod_name])[:, 1]
            probs.append(p)
        return np.mean(probs, axis=0)

    def predict(self, X_dict: dict) -> np.ndarray:
        avg_prob = self.predict_proba_combined(X_dict)
        return (avg_prob >= 0.5).astype(int)

    def evaluate(self, X_dict: dict, y_test: np.ndarray) -> dict:
        y_pred = self.predict(X_dict)
        y_prob = self.predict_proba_combined(X_dict)
        return evaluate_model(y_test, y_pred, y_prob, self.name)


# ---------------------------------------------------------------------------
# Train all models
# ---------------------------------------------------------------------------
def train_all_models(splits: dict, random_state: int = 42) -> dict:
    """Train all single-modality and fusion models.

    Parameters
    ----------
    splits : dict
        Output of data_loader.prepare_splits().

    Returns
    -------
    dict with keys: 'single_models', 'fusion_models', 'all_metrics'
    """
    print("\n" + "="*60)
    print("  TRAINING MODELS")
    print("="*60)

    # ---- Single-modality baselines ----
    single_models = {}
    single_metrics = []
    for mod in ["url", "content", "external"]:
        sm = SingleModalityModel(mod, random_state=random_state)
        sm.fit(splits[mod]["X_train"], splits[mod]["y_train"])
        m = sm.evaluate(splits[mod]["X_test"], splits[mod]["y_test"])
        print_metrics(m)
        single_models[mod] = sm
        single_metrics.append(m)

    # ---- Feature-level late fusion ----
    fusion_models = {}
    fusion_metrics = []

    # RF fusion
    rf_fusion = LateFusionRF(random_state=random_state)
    rf_fusion.fit(splits["combined"]["X_train"],
                  splits["combined"]["y_train"])
    m = rf_fusion.evaluate(splits["combined"]["X_test"],
                           splits["combined"]["y_test"])
    print_metrics(m)
    fusion_models["rf"] = rf_fusion
    fusion_metrics.append(m)

    # XGBoost fusion
    xgb_fusion = LateFusionXGB(random_state=random_state)
    xgb_fusion.fit(splits["combined"]["X_train"],
                   splits["combined"]["y_train"])
    m = xgb_fusion.evaluate(splits["combined"]["X_test"],
                            splits["combined"]["y_test"])
    print_metrics(m)
    fusion_models["xgb"] = xgb_fusion
    fusion_metrics.append(m)

    # LightGBM fusion
    lgbm_fusion = LateFusionLGBM(random_state=random_state)
    lgbm_fusion.fit(splits["combined"]["X_train"],
                    splits["combined"]["y_train"])
    m = lgbm_fusion.evaluate(splits["combined"]["X_test"],
                             splits["combined"]["y_test"])
    print_metrics(m)
    fusion_models["lgbm"] = lgbm_fusion
    fusion_metrics.append(m)

    # Probability-level fusion
    prob_fusion = ProbabilityFusion()
    prob_fusion.fit(single_models)
    X_test_dict = {mod: splits[mod]["X_test"] for mod in
                   ["url", "content", "external"]}
    m = prob_fusion.evaluate(X_test_dict, splits["combined"]["y_test"])
    print_metrics(m)
    fusion_models["prob"] = prob_fusion
    fusion_metrics.append(m)

    all_metrics = single_metrics + fusion_metrics

    return {
        "single_models": single_models,
        "fusion_models": fusion_models,
        "all_metrics": all_metrics,
    }
