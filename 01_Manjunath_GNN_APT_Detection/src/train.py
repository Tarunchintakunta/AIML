"""
Training and evaluation pipeline for GraphSAGE APT detection model.
Includes mini-batch training with neighbour sampling and full evaluation.
"""

import time
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             roc_auc_score)
from model import GraphSAGEDetector, GraphSAGEThreeLayer, create_neighbor_loader


def train_epoch(model, data, optimizer, device):
    """Train one epoch using full-batch approach."""
    model.train()
    optimizer.zero_grad()

    data = data.to(device)
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    return loss.item()


def train_epoch_minibatch(model, loader, optimizer, device):
    """Train one epoch using mini-batch neighbour sampling."""
    model.train()
    total_loss = 0
    total_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)

        # Only compute loss on seed nodes (batch_size)
        mask = batch.train_mask[:batch.batch_size]
        y = batch.y[:batch.batch_size]

        if mask.sum() == 0:
            continue

        loss = F.cross_entropy(out[:batch.batch_size][mask], y[mask])
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * mask.sum().item()
        total_nodes += mask.sum().item()

    return total_loss / max(total_nodes, 1)


@torch.no_grad()
def evaluate(model, data, mask_attr, device):
    """Evaluate model on the specified mask (val/test)."""
    model.eval()
    data = data.to(device)
    out = model(data.x, data.edge_index)
    mask = getattr(data, mask_attr)

    preds = out[mask].argmax(dim=1).cpu().numpy()
    labels = data.y[mask].cpu().numpy()
    probs = F.softmax(out[mask], dim=1)[:, 1].cpu().numpy()

    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'f1': f1_score(labels, preds, zero_division=0),
        'report': classification_report(labels, preds, target_names=['Benign', 'Malicious'],
                                        zero_division=0),
        'confusion_matrix': confusion_matrix(labels, preds),
    }

    try:
        metrics['auc_roc'] = roc_auc_score(labels, probs)
    except ValueError:
        metrics['auc_roc'] = 0.0

    return metrics


def run_training(data, model_class=GraphSAGEDetector, hidden_channels=128,
                 epochs=100, lr=0.01, weight_decay=5e-4, use_minibatch=False,
                 batch_size=512, patience=15):
    """
    Full training pipeline with early stopping.

    Args:
        data: PyTorch Geometric Data object.
        model_class: Model class to use.
        hidden_channels: Hidden layer size.
        epochs: Maximum training epochs.
        lr: Learning rate.
        weight_decay: L2 regularization.
        use_minibatch: Use neighbour sampling mini-batch training.
        batch_size: Mini-batch size.
        patience: Early stopping patience.

    Returns:
        model: Trained model.
        results: Dictionary of training history and test metrics.
    """
    device = torch.device('cpu')  # CPU-based as per research question
    in_channels = data.num_node_features

    model = model_class(in_channels=in_channels, hidden_channels=hidden_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if use_minibatch:
        train_loader = create_neighbor_loader(data, batch_size=batch_size,
                                              mask_attr='train_mask')

    best_val_f1 = 0
    best_model_state = None
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_f1': [], 'val_accuracy': []}

    print(f"Training {model_class.__name__} on {device}...")
    print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}, Features: {in_channels}")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        if use_minibatch:
            loss = train_epoch_minibatch(model, train_loader, optimizer, device)
        else:
            loss = train_epoch(model, data, optimizer, device)

        val_metrics = evaluate(model, data, 'val_mask', device)

        history['train_loss'].append(loss)
        history['val_f1'].append(val_metrics['f1'])
        history['val_accuracy'].append(val_metrics['accuracy'])

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Loss: {loss:.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    training_time = time.time() - start_time

    # Load best model and evaluate on test set
    if best_model_state:
        model.load_state_dict(best_model_state)

    test_metrics = evaluate(model, data, 'test_mask', device)

    print(f"\n{'='*60}")
    print(f"Training completed in {training_time:.2f}s")
    print(f"Test Results:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1-Score:  {test_metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {test_metrics['auc_roc']:.4f}")
    print(f"\nClassification Report:\n{test_metrics['report']}")
    print(f"Confusion Matrix:\n{test_metrics['confusion_matrix']}")

    results = {
        'history': history,
        'test_metrics': test_metrics,
        'training_time': training_time,
        'device': str(device),
    }

    return model, results
