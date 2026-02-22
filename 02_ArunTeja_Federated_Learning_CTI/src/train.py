"""
Federated training pipeline with differential privacy and Byzantine robustness.
"""

import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)

from model import (IntrusionDetectorMLP, fedavg_aggregate, krum_aggregate,
                   median_aggregate, add_dp_noise)


def train_local(model, features, labels, epochs=5, lr=0.01, batch_size=128):
    """Train a model locally on one client's data."""
    model.train()
    dataset = TensorDataset(torch.tensor(features), torch.tensor(labels))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()

    return model


@torch.no_grad()
def evaluate_model(model, features, labels):
    """Evaluate model on test data."""
    model.eval()
    X = torch.tensor(features)
    out = model(X)
    preds = out.argmax(dim=1).numpy()

    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'f1': f1_score(labels, preds, zero_division=0),
        'report': classification_report(labels, preds,
                                        target_names=['Benign', 'Malicious'],
                                        zero_division=0),
        'confusion_matrix': confusion_matrix(labels, preds),
    }


def run_federated_training(partitions, test_data, input_dim,
                           aggregation='fedavg', n_rounds=20,
                           local_epochs=5, lr=0.01, epsilon=None,
                           n_malicious=0):
    """
    Run full federated learning pipeline.

    Args:
        partitions: List of (features, labels) for each client.
        test_data: Tuple of (test_features, test_labels).
        input_dim: Number of input features.
        aggregation: 'fedavg', 'krum', or 'median'.
        n_rounds: Number of federated rounds.
        local_epochs: Local training epochs per round.
        lr: Learning rate.
        epsilon: Differential privacy epsilon (None = no DP).
        n_malicious: Number of assumed malicious clients (for Krum).

    Returns:
        global_model: Final aggregated model.
        history: Training history.
    """
    n_clients = len(partitions)
    test_features, test_labels = test_data

    global_model = IntrusionDetectorMLP(input_dim=input_dim)
    history = {'round': [], 'accuracy': [], 'f1': [], 'loss': []}

    print(f"\nFederated Learning: {aggregation.upper()} | "
          f"Clients: {n_clients} | Rounds: {n_rounds} | "
          f"DP epsilon: {epsilon or 'None'}")
    print("-" * 60)

    start_time = time.time()

    for r in range(1, n_rounds + 1):
        client_models = []
        client_sizes = []

        for i, (feats, labs) in enumerate(partitions):
            if len(feats) == 0:
                continue

            local_model = copy.deepcopy(global_model)
            local_model = train_local(local_model, feats, labs,
                                      epochs=local_epochs, lr=lr)

            if epsilon is not None:
                local_model = add_dp_noise(local_model, epsilon=epsilon)

            client_models.append(local_model)
            client_sizes.append(len(feats))

        if not client_models:
            continue

        # Aggregate
        if aggregation == 'fedavg':
            global_model = fedavg_aggregate(global_model, client_models, client_sizes)
        elif aggregation == 'krum':
            global_model = krum_aggregate(global_model, client_models, n_malicious)
        elif aggregation == 'median':
            global_model = median_aggregate(global_model, client_models)

        # Evaluate
        metrics = evaluate_model(global_model, test_features, test_labels)
        history['round'].append(r)
        history['accuracy'].append(metrics['accuracy'])
        history['f1'].append(metrics['f1'])

        if r % 5 == 0 or r == 1:
            print(f"  Round {r:3d} | Acc: {metrics['accuracy']:.4f} | "
                  f"F1: {metrics['f1']:.4f}")

    total_time = time.time() - start_time

    # Final evaluation
    final_metrics = evaluate_model(global_model, test_features, test_labels)
    print(f"\nFinal Results ({aggregation.upper()}):")
    print(f"  Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"  Precision: {final_metrics['precision']:.4f}")
    print(f"  Recall:    {final_metrics['recall']:.4f}")
    print(f"  F1-Score:  {final_metrics['f1']:.4f}")
    print(f"  Time:      {total_time:.2f}s")
    print(f"\n{final_metrics['report']}")

    final_metrics['training_time'] = total_time
    final_metrics['history'] = history

    return global_model, final_metrics


def run_centralised_baseline(partitions, test_data, input_dim, epochs=30, lr=0.01):
    """Train a centralised baseline (all data combined) for comparison."""
    all_features = np.vstack([p[0] for p in partitions if len(p[0]) > 0])
    all_labels = np.concatenate([p[1] for p in partitions if len(p[1]) > 0])
    test_features, test_labels = test_data

    model = IntrusionDetectorMLP(input_dim=input_dim)
    model = train_local(model, all_features, all_labels, epochs=epochs, lr=lr)

    metrics = evaluate_model(model, test_features, test_labels)
    print(f"\nCentralised Baseline:")
    print(f"  Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")

    return model, metrics
