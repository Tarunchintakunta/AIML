"""
Data loading and preprocessing module for CIC-IDS2017 and ToN-IoT datasets.
Constructs graph representations from network flow data for GraphSAGE-based APT detection.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data


def load_cicids2017(data_path: str) -> pd.DataFrame:
    """Load and concatenate CIC-IDS2017 CSV files."""
    csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_path}")

    frames = []
    for f in sorted(csv_files):
        df = pd.read_csv(os.path.join(data_path, f), low_memory=False)
        df.columns = df.columns.str.strip()
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    return data


def load_toniot(data_path: str) -> pd.DataFrame:
    """Load ToN-IoT network dataset."""
    csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_path}")

    frames = []
    for f in sorted(csv_files):
        df = pd.read_csv(os.path.join(data_path, f), low_memory=False)
        df.columns = df.columns.str.strip()
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    return data


def preprocess_dataframe(df: pd.DataFrame, label_col: str = 'Label',
                         max_samples: int = 100000) -> tuple:
    """
    Clean, encode, and scale the dataframe.

    Returns:
        features (np.ndarray): Scaled feature matrix.
        labels (np.ndarray): Binary labels (0=benign, 1=malicious).
        src_ips (np.ndarray): Source IP addresses for graph construction.
        dst_ips (np.ndarray): Destination IP addresses for graph construction.
    """
    df = df.copy()

    # Drop rows with NaN/Inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Subsample if too large
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)

    # Extract IP columns for graph edges
    src_col = None
    dst_col = None
    for col in df.columns:
        col_lower = col.lower()
        if 'src' in col_lower and 'ip' in col_lower:
            src_col = col
        elif 'dst' in col_lower and 'ip' in col_lower:
            dst_col = col
        elif col_lower == 'source ip':
            src_col = col
        elif col_lower == 'destination ip':
            dst_col = col

    src_ips = df[src_col].values if src_col else np.arange(len(df))
    dst_ips = df[dst_col].values if dst_col else np.arange(len(df))

    # Encode labels as binary
    if label_col in df.columns:
        labels = df[label_col].apply(
            lambda x: 0 if str(x).strip().upper() in ['BENIGN', 'NORMAL', '0'] else 1
        ).values
    else:
        raise KeyError(f"Label column '{label_col}' not found in dataframe.")

    # Drop non-numeric columns
    drop_cols = [label_col]
    if src_col:
        drop_cols.append(src_col)
    if dst_col:
        drop_cols.append(dst_col)

    for col in df.columns:
        if df[col].dtype == 'object' and col not in drop_cols:
            drop_cols.append(col)

    features_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # Convert to numeric
    features_df = features_df.apply(pd.to_numeric, errors='coerce')
    features_df.fillna(0, inplace=True)

    # Scale features
    scaler = StandardScaler()
    features = scaler.fit_transform(features_df.values)

    return features, labels, src_ips, dst_ips


def build_graph(features: np.ndarray, labels: np.ndarray,
                src_ips: np.ndarray, dst_ips: np.ndarray) -> Data:
    """
    Build a PyTorch Geometric graph from network flow data.

    Each flow becomes a node. Edges connect flows sharing the same
    source or destination IP (communication graph).
    """
    n = len(features)

    # Create IP-to-node mapping for edge construction
    ip_to_nodes = {}
    for i in range(n):
        src = str(src_ips[i])
        dst = str(dst_ips[i])
        ip_to_nodes.setdefault(src, []).append(i)
        ip_to_nodes.setdefault(dst, []).append(i)

    # Build edge list: connect nodes sharing an IP (limit neighbours for scalability)
    max_neighbours = 10
    edge_src, edge_dst = [], []
    for ip, nodes in ip_to_nodes.items():
        if len(nodes) > max_neighbours:
            nodes = np.random.choice(nodes, max_neighbours, replace=False).tolist()
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                edge_src.extend([nodes[i], nodes[j]])
                edge_dst.extend([nodes[j], nodes[i]])

    # If no edges created, add self-loops
    if not edge_src:
        edge_src = list(range(n))
        edge_dst = list(range(n))

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)

    data = Data(x=x, y=y, edge_index=edge_index)
    return data


def prepare_dataset(data_path: str, dataset_type: str = 'cicids2017',
                    label_col: str = 'Label', max_samples: int = 100000) -> Data:
    """
    End-to-end data preparation pipeline.

    Args:
        data_path: Path to the dataset CSV directory.
        dataset_type: 'cicids2017' or 'toniot'.
        label_col: Name of the label column.
        max_samples: Maximum number of samples to use.

    Returns:
        PyTorch Geometric Data object.
    """
    if dataset_type == 'cicids2017':
        df = load_cicids2017(data_path)
    elif dataset_type == 'toniot':
        df = load_toniot(data_path)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    features, labels, src_ips, dst_ips = preprocess_dataframe(
        df, label_col=label_col, max_samples=max_samples
    )

    graph = build_graph(features, labels, src_ips, dst_ips)

    # Create train/val/test masks
    n = graph.num_nodes
    indices = np.arange(n)
    train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42,
                                            stratify=labels)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42,
                                          stratify=labels[test_idx])

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    graph.train_mask = train_mask
    graph.val_mask = val_mask
    graph.test_mask = test_mask

    return graph


def generate_synthetic_data(n_nodes: int = 5000, n_features: int = 20,
                            attack_ratio: float = 0.2) -> Data:
    """
    Generate synthetic network flow graph data for testing.
    Useful when real datasets are not available.
    """
    np.random.seed(42)

    n_attack = int(n_nodes * attack_ratio)
    n_benign = n_nodes - n_attack

    # Benign features: centered around 0
    benign_features = np.random.randn(n_benign, n_features) * 0.5

    # Attack features: shifted distribution
    attack_features = np.random.randn(n_attack, n_features) * 0.8 + 1.5

    features = np.vstack([benign_features, attack_features])
    labels = np.array([0] * n_benign + [1] * n_attack)

    # Shuffle
    perm = np.random.permutation(n_nodes)
    features = features[perm]
    labels = labels[perm]

    # Build random graph edges (simulating IP-based connections)
    n_edges = n_nodes * 5
    src = np.random.randint(0, n_nodes, n_edges)
    dst = np.random.randint(0, n_nodes, n_edges)
    # Remove self-loops
    mask = src != dst
    src, dst = src[mask], dst[mask]

    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)

    data = Data(x=x, y=y, edge_index=edge_index)

    # Masks
    indices = np.arange(n_nodes)
    train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42,
                                            stratify=labels)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42,
                                          stratify=labels[test_idx])

    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data
