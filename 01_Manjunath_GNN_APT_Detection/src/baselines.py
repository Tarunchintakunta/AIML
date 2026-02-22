"""
Baseline classifiers for comparison with GraphSAGE.
Implements Random Forest, XGBoost, and MLP baselines using scikit-learn.
"""

import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             roc_auc_score)


def evaluate_baseline(y_true, y_pred, y_prob=None):
    """Compute standard evaluation metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'report': classification_report(y_true, y_pred,
                                        target_names=['Benign', 'Malicious'],
                                        zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
    }
    if y_prob is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['auc_roc'] = 0.0
    return metrics


def run_random_forest(data):
    """Train and evaluate a Random Forest baseline."""
    X_train = data.x[data.train_mask].numpy()
    y_train = data.y[data.train_mask].numpy()
    X_test = data.x[data.test_mask].numpy()
    y_test = data.y[data.test_mask].numpy()

    print("Training Random Forest baseline...")
    start = time.time()

    clf = RandomForestClassifier(n_estimators=200, max_depth=20,
                                 random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    metrics = evaluate_baseline(y_test, y_pred, y_prob)
    metrics['training_time'] = train_time

    print(f"  Random Forest - F1: {metrics['f1']:.4f}, "
          f"Accuracy: {metrics['accuracy']:.4f}, Time: {train_time:.2f}s")
    return metrics


def run_xgboost(data):
    """Train and evaluate an XGBoost baseline."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("  XGBoost not installed. Skipping.")
        return None

    X_train = data.x[data.train_mask].numpy()
    y_train = data.y[data.train_mask].numpy()
    X_test = data.x[data.test_mask].numpy()
    y_test = data.y[data.test_mask].numpy()

    print("Training XGBoost baseline...")
    start = time.time()

    clf = XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                        random_state=42, use_label_encoder=False,
                        eval_metric='logloss', n_jobs=-1)
    clf.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    metrics = evaluate_baseline(y_test, y_pred, y_prob)
    metrics['training_time'] = train_time

    print(f"  XGBoost - F1: {metrics['f1']:.4f}, "
          f"Accuracy: {metrics['accuracy']:.4f}, Time: {train_time:.2f}s")
    return metrics


def run_mlp(data):
    """Train and evaluate an MLP baseline."""
    X_train = data.x[data.train_mask].numpy()
    y_train = data.y[data.train_mask].numpy()
    X_test = data.x[data.test_mask].numpy()
    y_test = data.y[data.test_mask].numpy()

    print("Training MLP baseline...")
    start = time.time()

    clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200,
                        random_state=42, early_stopping=True,
                        validation_fraction=0.15)
    clf.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    metrics = evaluate_baseline(y_test, y_pred, y_prob)
    metrics['training_time'] = train_time

    print(f"  MLP - F1: {metrics['f1']:.4f}, "
          f"Accuracy: {metrics['accuracy']:.4f}, Time: {train_time:.2f}s")
    return metrics


def run_all_baselines(data):
    """Run all baseline models and return results."""
    results = {}
    results['random_forest'] = run_random_forest(data)
    results['xgboost'] = run_xgboost(data)
    results['mlp'] = run_mlp(data)
    return results
