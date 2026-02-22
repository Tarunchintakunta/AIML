"""
Ensemble models for NIDS: Bagging (Random Forest), Boosting (XGBoost, LightGBM),
and Stacking ensemble.
"""

import time
import numpy as np
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def create_random_forest(n_estimators=200, max_depth=20):
    """Bagging: Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=42, n_jobs=-1, class_weight='balanced'
    )


def create_xgboost(n_estimators=200, max_depth=8, lr=0.1):
    """Boosting: XGBoost classifier."""
    return XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=lr, random_state=42,
        eval_metric='logloss', verbosity=0
    )


def create_lightgbm(n_estimators=200, max_depth=8, lr=0.1):
    """Boosting: LightGBM classifier."""
    return LGBMClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=lr, random_state=42, verbose=-1,
        class_weight='balanced'
    )


def create_stacking_ensemble():
    """Stacking: RF + XGBoost + LightGBM with Logistic Regression meta-learner."""
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=15,
                                       random_state=42, n_jobs=-1)),
        ('xgb', XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                               random_state=42, eval_metric='logloss',
                               verbosity=0)),
        ('lgbm', LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                 random_state=42, verbose=-1)),
    ]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=3, n_jobs=-1
    )


def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name='Model'):
    """Train model and return metrics with timing."""
    print(f"\n  Training {model_name}...")
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    start = time.time()
    preds = model.predict(X_test)
    predict_time = time.time() - start

    metrics = {
        'accuracy': accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds, zero_division=0),
        'recall': recall_score(y_test, preds, zero_division=0),
        'f1': f1_score(y_test, preds, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, preds),
        'report': classification_report(y_test, preds,
                                        target_names=['Benign', 'Malicious'],
                                        zero_division=0),
        'train_time': train_time,
        'predict_time': predict_time,
    }

    print(f"  {model_name}: Acc={metrics['accuracy']:.4f} | "
          f"F1={metrics['f1']:.4f} | Train={train_time:.2f}s | "
          f"Predict={predict_time:.4f}s")

    return model, metrics


def get_all_models():
    """Return dict of all ensemble models."""
    return {
        'Random Forest': create_random_forest(),
        'XGBoost': create_xgboost(),
        'LightGBM': create_lightgbm(),
        'Stacking': create_stacking_ensemble(),
    }
