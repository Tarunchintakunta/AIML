"""
Deep learning models for encrypted traffic classification:
CNN, LSTM, CNN-LSTM hybrid with attention, and Random Forest baseline.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)


class CNNClassifier(nn.Module):
    """1D-CNN for spatial feature extraction from flow statistics."""
    def __init__(self, input_dim=28, n_classes=7):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        # x: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class LSTMClassifier(nn.Module):
    """LSTM for temporal pattern learning from flow features."""
    def __init__(self, input_dim=28, hidden_dim=64, n_layers=2, n_classes=7):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim,
                           num_layers=n_layers, batch_first=True,
                           dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        # x: (batch, features) -> (batch, features, 1) as sequence
        x = x.unsqueeze(-1)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)


class AttentionLayer(nn.Module):
    """Simple attention mechanism for temporal weighting."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_dim)
        weights = torch.softmax(self.attn(lstm_output), dim=1)
        context = (weights * lstm_output).sum(dim=1)
        return context, weights


class CNNLSTMAttention(nn.Module):
    """Hybrid CNN-LSTM with attention for encrypted traffic classification."""
    def __init__(self, input_dim=28, cnn_channels=64, lstm_hidden=64,
                 n_classes=7):
        super().__init__()
        # CNN for spatial features
        self.conv1 = nn.Conv1d(1, cnn_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3,
                               padding=1)
        self.bn = nn.BatchNorm1d(cnn_channels)

        # LSTM for temporal patterns
        self.lstm = nn.LSTM(input_size=cnn_channels, hidden_size=lstm_hidden,
                           num_layers=2, batch_first=True, dropout=0.3)

        # Attention
        self.attention = AttentionLayer(lstm_hidden)

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        # x: (batch, features)
        x = x.unsqueeze(1)  # (batch, 1, features)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.bn(self.conv2(x)))
        # (batch, channels, features) -> (batch, features, channels)
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)
        context, attn_weights = self.attention(lstm_out)
        return self.fc(context)


def train_model(model, X_train, y_train, X_val, y_val, epochs=30,
                lr=0.001, batch_size=128):
    """Train a PyTorch model with early stopping."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=True)

    best_val_f1 = 0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()

        # Validation
        val_metrics = evaluate_model(model, X_val, y_val)
        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_macro']
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 10:
            break

    if best_state:
        model.load_state_dict(best_state)

    return model


@torch.no_grad()
def evaluate_model(model, X, y, class_names=None):
    """Evaluate PyTorch model."""
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    preds = model(X_tensor).argmax(dim=1).numpy()

    metrics = {
        'accuracy': accuracy_score(y, preds),
        'precision': precision_score(y, preds, average='macro', zero_division=0),
        'recall': recall_score(y, preds, average='macro', zero_division=0),
        'f1_macro': f1_score(y, preds, average='macro', zero_division=0),
        'confusion_matrix': confusion_matrix(y, preds),
    }

    if class_names:
        metrics['report'] = classification_report(
            y, preds, target_names=class_names, zero_division=0
        )

    return metrics


def measure_latency(model, X, n_runs=50):
    """Measure average inference latency per sample."""
    X_tensor = torch.tensor(X[:100], dtype=torch.float32)
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            model(X_tensor)

    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            model(X_tensor)
    total = time.time() - start

    return total / (n_runs * len(X_tensor))


def train_random_forest(X_train, y_train, X_test, y_test, class_names=None):
    """Random Forest baseline."""
    rf = RandomForestClassifier(n_estimators=200, max_depth=20,
                                random_state=42, n_jobs=-1)
    start = time.time()
    rf.fit(X_train, y_train)
    train_time = time.time() - start

    preds = rf.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds, average='macro', zero_division=0),
        'recall': recall_score(y_test, preds, average='macro', zero_division=0),
        'f1_macro': f1_score(y_test, preds, average='macro', zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, preds),
        'train_time': train_time,
    }

    if class_names:
        metrics['report'] = classification_report(
            y_test, preds, target_names=class_names, zero_division=0
        )

    return rf, metrics
