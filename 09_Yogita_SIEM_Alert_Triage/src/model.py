"""
model.py - Cost-Sensitive Ensemble Models for SIEM Alert Triage
================================================================
Individual classifiers:
    1. Random Forest (cost-sensitive via class_weight)
    2. XGBoost (cost-sensitive via sample_weight)
    3. Isolation Forest (anomaly scores as features)

Stacking ensemble with Logistic Regression meta-learner.

Cost Matrix:
    - FN cost (missing a True Positive) = 10x FP cost
    - Evaluated with F-beta score (beta=2, recall-heavy)
"""

import numpy as np
import warnings
from typing import Dict, Any, Optional

from sklearn.ensemble import (
    RandomForestClassifier,
    IsolationForest,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    fbeta_score,
    make_scorer,
    precision_recall_fscore_support,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Cost configuration
# ---------------------------------------------------------------------------
# Class mapping: 0=FP, 1=Indeterminate, 2=TP
# FN cost for TP is 10x the FP cost.  We encode this as class weights.
COST_MATRIX = {
    0: 1.0,   # False Positive  - low misclassification cost
    1: 3.0,   # Indeterminate   - moderate cost
    2: 10.0,  # True Positive   - highest cost when missed (FN)
}

BETA = 2  # F-beta with beta=2 favours recall


def _compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """Map class labels to per-sample cost weights."""
    return np.array([COST_MATRIX[int(label)] for label in y])


# ---------------------------------------------------------------------------
# Individual model builders
# ---------------------------------------------------------------------------
def build_random_forest(random_state: int = 42) -> RandomForestClassifier:
    """Cost-sensitive Random Forest with class_weight parameter."""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight={0: 1.0, 1: 3.0, 2: 10.0},
        random_state=random_state,
        n_jobs=-1,
    )


def build_xgboost(random_state: int = 42) -> XGBClassifier:
    """XGBoost with multi-class objective.  Cost-sensitivity is applied
    through sample_weight during .fit()."""
    return XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )


def build_isolation_forest(random_state: int = 42) -> IsolationForest:
    """Isolation Forest for anomaly scoring (unsupervised)."""
    return IsolationForest(
        n_estimators=200,
        contamination=0.05,
        max_features=1.0,
        random_state=random_state,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# Anomaly feature augmentation
# ---------------------------------------------------------------------------
def augment_with_anomaly_scores(
    iso_forest: IsolationForest,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple:
    """Fit Isolation Forest on training data and append anomaly scores
    as an extra feature to all splits."""
    iso_forest.fit(X_train)
    score_train = iso_forest.decision_function(X_train).reshape(-1, 1)
    score_val = iso_forest.decision_function(X_val).reshape(-1, 1)
    score_test = iso_forest.decision_function(X_test).reshape(-1, 1)
    return (
        np.hstack([X_train, score_train]),
        np.hstack([X_val, score_val]),
        np.hstack([X_test, score_test]),
    )


# ---------------------------------------------------------------------------
# Stacking ensemble
# ---------------------------------------------------------------------------
def build_stacking_ensemble(random_state: int = 42) -> StackingClassifier:
    """Stacking classifier: RF + XGBoost as base, LR meta-learner.
    Uses cross-validated predictions to avoid leakage."""
    estimators = [
        ("rf", build_random_forest(random_state)),
        ("xgb", build_xgboost(random_state)),
    ]
    meta_learner = LogisticRegression(
        max_iter=1000,
        class_weight={0: 1.0, 1: 3.0, 2: 10.0},
        multi_class="multinomial",
        solver="lbfgs",
        random_state=random_state,
    )
    return StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=5,
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def train_model(model, X_train, y_train, model_name: str = "model"):
    """Train a single model, applying sample weights for cost-sensitivity
    where the API supports it."""
    sample_weights = _compute_sample_weights(y_train)

    if model_name in ("xgb", "xgboost"):
        model.fit(X_train, y_train, sample_weight=sample_weights)
    elif model_name in ("rf", "random_forest"):
        # RF uses class_weight natively; fit normally
        model.fit(X_train, y_train)
    elif model_name in ("stacking", "ensemble"):
        model.fit(X_train, y_train)
    else:
        try:
            model.fit(X_train, y_train, sample_weight=sample_weights)
        except TypeError:
            model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    class_names=None,
) -> Dict[str, Any]:
    """Comprehensive evaluation returning metrics dict."""
    if class_names is None:
        class_names = ["False Positive", "Indeterminate", "True Positive"]

    y_pred = model.predict(X_test)

    # F-beta (macro + per-class)
    fbeta_macro = fbeta_score(y_test, y_pred, beta=BETA, average="macro")
    fbeta_weighted = fbeta_score(y_test, y_pred, beta=BETA, average="weighted")
    fbeta_per_class = fbeta_score(y_test, y_pred, beta=BETA, average=None)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None,
    )

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=class_names, digits=4,
    )

    # Cost-based evaluation
    cost = _compute_total_cost(y_test, y_pred)

    # Alert reduction: proportion of alerts classified as FP
    alert_reduction = np.mean(y_pred == 0)

    results = {
        "model_name": model_name,
        "y_pred": y_pred,
        "fbeta_macro": fbeta_macro,
        "fbeta_weighted": fbeta_weighted,
        "fbeta_per_class": fbeta_per_class,
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1,
        "support_per_class": support,
        "confusion_matrix": cm,
        "classification_report": report,
        "total_cost": cost,
        "alert_reduction_rate": alert_reduction,
    }
    return results


def _compute_total_cost(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute asymmetric misclassification cost.

    Cost model:
        - Correctly classified: 0
        - FP classified as Indet: 0.5  (minor analyst overhead)
        - FP classified as TP:   1.0  (unnecessary investigation)
        - Indet classified as FP: 2.0 (possible missed threat)
        - Indet classified as TP: 0.5 (over-escalation, tolerable)
        - TP classified as FP:  10.0  (missed real attack -- critical)
        - TP classified as Indet: 5.0 (delayed response)
    """
    cost_matrix = np.array([
        #  pred-FP  pred-Ind  pred-TP
        [  0.0,     0.5,      1.0 ],   # true FP
        [  2.0,     0.0,      0.5 ],   # true Indet
        [ 10.0,     5.0,      0.0 ],   # true TP
    ])
    total = 0.0
    for t, p in zip(y_true, y_pred):
        total += cost_matrix[int(t), int(p)]
    return total


def compute_baseline_cost(y_test: np.ndarray) -> float:
    """Cost if every alert is escalated (predicted TP).
    This is the 'no-ML' baseline."""
    y_pred_all_tp = np.full_like(y_test, 2)
    return _compute_total_cost(y_test, y_pred_all_tp)


def print_results(results: Dict[str, Any]) -> None:
    """Pretty-print evaluation results to console."""
    print(f"\n{'='*60}")
    print(f" {results['model_name']} -- Evaluation Results")
    print(f"{'='*60}")
    print(results["classification_report"])
    print(f"  F-beta (beta=2, macro):    {results['fbeta_macro']:.4f}")
    print(f"  F-beta (beta=2, weighted): {results['fbeta_weighted']:.4f}")
    print(f"  Total Misclassification Cost: {results['total_cost']:.1f}")
    print(f"  Alert Reduction Rate:     {results['alert_reduction_rate']:.2%}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Quick standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_loader import generate_synthetic_siem_data, prepare_data

    df = generate_synthetic_siem_data(n_samples=5000, random_state=42)
    data = prepare_data(df)

    print("Training Random Forest ...")
    rf = build_random_forest()
    rf = train_model(rf, data["X_train"], data["y_train"], "rf")
    res = evaluate_model(rf, data["X_test"], data["y_test"], "Random Forest")
    print_results(res)
