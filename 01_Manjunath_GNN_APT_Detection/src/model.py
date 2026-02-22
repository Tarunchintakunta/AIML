"""
GraphSAGE model with neighbour sampling for APT detection.
Implements both the GNN model and baseline classifiers for comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader


class GraphSAGEDetector(nn.Module):
    """
    Two-layer GraphSAGE model with neighbour sampling support.
    Designed for binary node classification (benign vs. malicious).
    """

    def __init__(self, in_channels: int, hidden_channels: int = 128,
                 out_channels: int = 2, dropout: float = 0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x

    def get_embeddings(self, x, edge_index):
        """Extract node embeddings before the classification layer."""
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x


class GraphSAGEThreeLayer(nn.Module):
    """Three-layer GraphSAGE for deeper neighbourhood aggregation."""

    def __init__(self, in_channels: int, hidden_channels: int = 128,
                 out_channels: int = 2, dropout: float = 0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels // 2)
        self.conv3 = SAGEConv(hidden_channels // 2, hidden_channels // 4)
        self.classifier = nn.Linear(hidden_channels // 4, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        x = self.classifier(x)
        return x


def create_neighbor_loader(data, batch_size: int = 512,
                           num_neighbors: list = None,
                           mask_attr: str = 'train_mask') -> NeighborLoader:
    """
    Create a NeighborLoader for mini-batch training with neighbour sampling.
    This enables scalable training on large graphs using standard CPU hardware.
    """
    if num_neighbors is None:
        num_neighbors = [15, 10]

    mask = getattr(data, mask_attr)
    input_nodes = mask.nonzero(as_tuple=False).view(-1)

    loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=input_nodes,
        shuffle=True if 'train' in mask_attr else False,
    )
    return loader
