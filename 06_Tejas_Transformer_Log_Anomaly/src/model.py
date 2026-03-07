"""
model.py - Anomaly Detection Models with Latency Measurement
=============================================================
Implements five detection pipelines:
    1. TF-IDF  + Isolation Forest
    2. TF-IDF  + One-Class SVM
    3. TF-IDF  + Local Outlier Factor (LOF)
    4. Simulated Transformer Embeddings + Isolation Forest
    5. Simulated Transformer Embeddings + LOF

Each pipeline exposes a common train/predict/evaluate interface and records
wall-clock latency for training and inference.

Author : Tejas Vijay Mariyappagoudar (x24213829)
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    """Stores evaluation results for a single model."""
    name: str
    feature_method: str
    detector_name: str
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    confusion: Optional[np.ndarray] = None
    train_time_sec: float = 0.0
    predict_time_sec: float = 0.0
    total_latency_sec: float = 0.0
    predictions: Optional[np.ndarray] = None
    report: str = ""

    def as_dict(self) -> Dict:
        return {
            "name": self.name,
            "feature_method": self.feature_method,
            "detector": self.detector_name,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "train_time_sec": round(self.train_time_sec, 4),
            "predict_time_sec": round(self.predict_time_sec, 4),
            "total_latency_sec": round(self.total_latency_sec, 4),
        }


# ---------------------------------------------------------------------------
# Detector wrappers
# ---------------------------------------------------------------------------

def _build_detector(name: str, contamination: float = 0.15):
    """
    Instantiate an unsupervised anomaly detector.

    All three detectors use the scikit-learn API and share a common
    contamination parameter that approximates the expected anomaly ratio.
    """
    if name == "isolation_forest":
        return IsolationForest(
            n_estimators=200,
            contamination=contamination,
            max_samples="auto",
            random_state=42,
            n_jobs=-1,
        )
    elif name == "ocsvm":
        # nu roughly maps to contamination
        return OneClassSVM(
            kernel="rbf",
            gamma="scale",
            nu=contamination,
        )
    elif name == "lof":
        return LocalOutlierFactor(
            n_neighbors=20,
            contamination=contamination,
            novelty=False,   # transductive mode
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown detector: {name}")


def _convert_predictions(raw_preds: np.ndarray) -> np.ndarray:
    """
    sklearn anomaly detectors return  +1 (inlier)  / -1 (outlier).
    Convert to  0 (normal)  /  1 (anomaly).
    """
    return (raw_preds == -1).astype(int)


# ---------------------------------------------------------------------------
# Train + evaluate
# ---------------------------------------------------------------------------

def train_and_evaluate(
    X_train: np.ndarray,
    y_true: np.ndarray,
    detector_name: str,
    feature_method: str,
    contamination: float = 0.15,
    model_label: str = "",
) -> ModelResult:
    """
    Train an unsupervised anomaly detector and evaluate against known labels.

    Parameters
    ----------
    X_train : np.ndarray, shape (n, d)
        Feature matrix (either TF-IDF or transformer embeddings).
    y_true : np.ndarray, shape (n,)
        Ground-truth labels (0 = normal, 1 = anomaly).
    detector_name : str
        One of "isolation_forest", "ocsvm", "lof".
    feature_method : str
        Descriptive tag, e.g. "tfidf" or "transformer".
    contamination : float
        Expected anomaly ratio.
    model_label : str
        Human-readable name for reports.

    Returns
    -------
    ModelResult
    """
    if not model_label:
        model_label = f"{feature_method}+{detector_name}"

    result = ModelResult(
        name=model_label,
        feature_method=feature_method,
        detector_name=detector_name,
    )

    detector = _build_detector(detector_name, contamination=contamination)

    # --- Training --------------------------------------------------------
    t0 = time.perf_counter()
    if detector_name == "lof":
        # LOF in transductive mode: fit_predict in one step
        raw = detector.fit_predict(X_train)
    else:
        detector.fit(X_train)
    result.train_time_sec = time.perf_counter() - t0

    # --- Prediction ------------------------------------------------------
    t1 = time.perf_counter()
    if detector_name == "lof":
        preds = _convert_predictions(raw)
    else:
        preds = _convert_predictions(detector.predict(X_train))
    result.predict_time_sec = time.perf_counter() - t1

    result.total_latency_sec = result.train_time_sec + result.predict_time_sec

    # --- Metrics ---------------------------------------------------------
    result.predictions = preds
    result.accuracy  = accuracy_score(y_true, preds)
    result.precision = precision_score(y_true, preds, zero_division=0)
    result.recall    = recall_score(y_true, preds, zero_division=0)
    result.f1        = f1_score(y_true, preds, zero_division=0)
    result.confusion = confusion_matrix(y_true, preds)
    result.report    = classification_report(y_true, preds, target_names=["Normal", "Anomaly"])

    return result


# ---------------------------------------------------------------------------
# Pipeline: run all five model configurations
# ---------------------------------------------------------------------------

PIPELINE_CONFIGS = [
    {"feature_method": "tfidf",       "detector_name": "isolation_forest", "label": "TF-IDF + Isolation Forest"},
    {"feature_method": "tfidf",       "detector_name": "ocsvm",           "label": "TF-IDF + One-Class SVM"},
    {"feature_method": "tfidf",       "detector_name": "lof",             "label": "TF-IDF + LOF"},
    {"feature_method": "transformer", "detector_name": "isolation_forest", "label": "Transformer + Isolation Forest"},
    {"feature_method": "transformer", "detector_name": "lof",             "label": "Transformer + LOF"},
]


def run_all_models(
    X_tfidf: np.ndarray,
    X_transformer: np.ndarray,
    y_true: np.ndarray,
    contamination: float = 0.15,
) -> list:
    """
    Execute all five pipelines and return a list of ModelResult objects.
    """
    feature_map = {
        "tfidf": X_tfidf,
        "transformer": X_transformer,
    }

    results = []
    for cfg in PIPELINE_CONFIGS:
        X = feature_map[cfg["feature_method"]]
        res = train_and_evaluate(
            X_train=X,
            y_true=y_true,
            detector_name=cfg["detector_name"],
            feature_method=cfg["feature_method"],
            contamination=contamination,
            model_label=cfg["label"],
        )
        results.append(res)
        print(f"  [{res.name}]  F1={res.f1:.4f}  Acc={res.accuracy:.4f}  "
              f"Latency={res.total_latency_sec:.4f}s")

    return results


# ---------------------------------------------------------------------------
# CLI quick-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    from data_loader import load_real_datasets, prepare_features

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), "data")

    df = load_real_datasets(data_dir=data_dir, seed=42)
    X_tfidf, _ = prepare_features(df, method="tfidf")
    X_trans, _  = prepare_features(df, method="transformer")
    y = df["label"].values

    contamination = max(0.01, min(0.5, y.mean()))
    print(f"Running all models (contamination={contamination:.4f}) ...\n")
    results = run_all_models(X_tfidf, X_trans, y, contamination=contamination)

    print("\n--- Best model by F1 ---")
    best = max(results, key=lambda r: r.f1)
    print(f"{best.name}  F1={best.f1:.4f}")
