"""
Data loading and preprocessing for Federated Learning CTI project.
Supports UNSW-NB15, CIC-IDS2017, and NSL-KDD datasets.
Includes non-IID partitioning for federated learning simulation.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def load_dataset(data_path: str, dataset_name: str) -> pd.DataFrame:
    """Load a dataset from CSV files."""
    csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_path}")
    frames = [pd.read_csv(os.path.join(data_path, f), low_memory=False) for f in sorted(csv_files)]
    df = pd.concat(frames, ignore_index=True)
    df.columns = df.columns.str.strip()
    return df


def preprocess(df: pd.DataFrame, label_col: str = 'label',
               max_samples: int = 50000) -> tuple:
    """Clean and preprocess dataframe into features and labels."""
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)

    # Find label column case-insensitively
    label_candidates = [c for c in df.columns if c.lower() == label_col.lower()]
    if not label_candidates:
        label_candidates = [c for c in df.columns if 'label' in c.lower() or 'class' in c.lower()]
    if not label_candidates:
        raise KeyError(f"No label column found matching '{label_col}'")
    actual_label = label_candidates[0]

    # Binary encode
    labels = df[actual_label].apply(
        lambda x: 0 if str(x).strip().upper() in ['BENIGN', 'NORMAL', '0', 'NONE'] else 1
    ).values

    # Drop non-numeric
    drop_cols = [actual_label]
    for col in df.columns:
        if df[col].dtype == 'object' and col != actual_label:
            drop_cols.append(col)
    features_df = df.drop(columns=drop_cols, errors='ignore')
    features_df = features_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    scaler = StandardScaler()
    features = scaler.fit_transform(features_df.values)

    return features.astype(np.float32), labels.astype(np.int64)


def partition_non_iid(features: np.ndarray, labels: np.ndarray,
                      n_clients: int = 5, alpha: float = 0.5) -> list:
    """
    Partition data into non-IID splits using Dirichlet distribution.
    Simulates heterogeneous data across federated clients.

    Args:
        features: Feature matrix.
        labels: Label array.
        n_clients: Number of federated clients.
        alpha: Dirichlet concentration (lower = more non-IID).

    Returns:
        List of (features, labels) tuples for each client.
    """
    n_classes = len(np.unique(labels))
    client_data = [[] for _ in range(n_clients)]

    for c in range(n_classes):
        idx = np.where(labels == c)[0]
        np.random.shuffle(idx)

        proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
        proportions = (proportions * len(idx)).astype(int)
        # Fix rounding
        proportions[-1] = len(idx) - proportions[:-1].sum()

        start = 0
        for i in range(n_clients):
            end = start + proportions[i]
            client_data[i].extend(idx[start:end].tolist())
            start = end

    partitions = []
    for indices in client_data:
        if indices:
            partitions.append((features[indices], labels[indices]))
        else:
            partitions.append((np.empty((0, features.shape[1])), np.empty(0)))

    return partitions


def generate_synthetic_data(n_samples: int = 10000, n_features: int = 30,
                            n_clients: int = 5, attack_ratio: float = 0.3):
    """Generate synthetic network traffic data for FL testing."""
    np.random.seed(42)

    n_attack = int(n_samples * attack_ratio)
    n_benign = n_samples - n_attack

    benign = np.random.randn(n_benign, n_features) * 0.5
    attack = np.random.randn(n_attack, n_features) * 0.8 + 1.2

    features = np.vstack([benign, attack]).astype(np.float32)
    labels = np.array([0] * n_benign + [1] * n_attack, dtype=np.int64)

    perm = np.random.permutation(n_samples)
    features, labels = features[perm], labels[perm]

    partitions = partition_non_iid(features, labels, n_clients=n_clients)

    # Global test set
    test_features = np.random.randn(2000, n_features).astype(np.float32)
    test_labels = np.array([0] * 1400 + [1] * 600, dtype=np.int64)
    perm = np.random.permutation(2000)
    test_features, test_labels = test_features[perm], test_labels[perm]

    # Adjust test data distributions
    test_features[test_labels == 1] = test_features[test_labels == 1] * 0.8 + 1.2

    return partitions, (test_features, test_labels)
