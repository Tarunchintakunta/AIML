"""
Neural network models for Federated Learning CTI project.
Includes lightweight MLP for CPU-only deployment and federated aggregation.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class IntrusionDetectorMLP(nn.Module):
    """Lightweight MLP for intrusion detection, optimised for CPU training."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, n_classes: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def fedavg_aggregate(global_model: nn.Module, client_models: list,
                     client_sizes: list) -> nn.Module:
    """
    Federated Averaging (FedAvg) aggregation.
    Weighted average of client model parameters by dataset size.
    """
    global_dict = global_model.state_dict()
    total_size = sum(client_sizes)

    for key in global_dict:
        global_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32)
        for model, size in zip(client_models, client_sizes):
            weight = size / total_size
            global_dict[key] += model.state_dict()[key].float() * weight

    global_model.load_state_dict(global_dict)
    return global_model


def krum_aggregate(global_model: nn.Module, client_models: list,
                   n_malicious: int = 0) -> nn.Module:
    """
    Krum Byzantine-robust aggregation.
    Selects the client model closest to the majority of other clients.
    """
    n_clients = len(client_models)
    n_select = n_clients - n_malicious - 2
    if n_select < 1:
        n_select = 1

    # Flatten parameters for each client
    flat_params = []
    for model in client_models:
        params = torch.cat([p.data.view(-1).float() for p in model.parameters()])
        flat_params.append(params)

    # Compute pairwise distances
    distances = torch.zeros(n_clients, n_clients)
    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            d = torch.norm(flat_params[i] - flat_params[j])
            distances[i][j] = d
            distances[j][i] = d

    # For each client, sum of distances to closest n_select clients
    scores = []
    for i in range(n_clients):
        sorted_d = torch.sort(distances[i])[0]
        scores.append(sorted_d[1:n_select + 1].sum().item())

    best_client = np.argmin(scores)
    global_model.load_state_dict(client_models[best_client].state_dict())
    return global_model


def median_aggregate(global_model: nn.Module, client_models: list) -> nn.Module:
    """
    Coordinate-wise median aggregation for Byzantine robustness.
    """
    global_dict = global_model.state_dict()

    for key in global_dict:
        stacked = torch.stack([m.state_dict()[key].float() for m in client_models])
        global_dict[key] = torch.median(stacked, dim=0).values

    global_model.load_state_dict(global_dict)
    return global_model


def add_dp_noise(model: nn.Module, epsilon: float = 1.0,
                 sensitivity: float = 1.0) -> nn.Module:
    """
    Add calibrated Gaussian noise for (epsilon, delta)-differential privacy.
    Uses the Gaussian mechanism with delta = 1e-5.
    """
    delta = 1e-5
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon

    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn_like(param) * sigma
            param.add_(noise)

    return model
