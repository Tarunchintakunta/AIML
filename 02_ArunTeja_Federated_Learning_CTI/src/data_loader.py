"""
Data loading and preprocessing for Federated Learning CTI project.
Downloads and uses REAL datasets: KDD Cup 99 (via sklearn) and NSL-KDD.
Includes non-IID partitioning for federated learning simulation.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# NSL-KDD column definitions
# ---------------------------------------------------------------------------
NSL_KDD_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
    'label', 'difficulty_level',
]

NORMAL_LABELS = {'normal'}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_nslkdd(data_dir: str = 'data/nslkdd') -> tuple:
    """
    Download NSL-KDD train/test CSVs from the public GitHub mirror.

    Returns:
        (train_path, test_path) absolute file paths.
    """
    import urllib.request

    os.makedirs(data_dir, exist_ok=True)

    train_url = (
        "https://raw.githubusercontent.com/defcom17/NSL_KDD/"
        "master/KDDTrain%2B.txt"
    )
    test_url = (
        "https://raw.githubusercontent.com/defcom17/NSL_KDD/"
        "master/KDDTest%2B.txt"
    )

    train_path = os.path.join(data_dir, 'KDDTrain+.txt')
    test_path = os.path.join(data_dir, 'KDDTest+.txt')

    for url, dest in [(train_url, train_path), (test_url, test_path)]:
        if not os.path.exists(dest):
            print(f"  Downloading {os.path.basename(dest)} ...")
            urllib.request.urlretrieve(url, dest)
            print(f"  Saved to {dest}")
        else:
            print(f"  Already cached: {dest}")

    return train_path, test_path


def load_nslkdd(data_dir: str = 'data/nslkdd',
                max_samples: int = 50000) -> tuple:
    """
    Download (if needed) and load NSL-KDD into numpy arrays.

    Returns:
        (features, labels) where labels are binary (0=normal, 1=attack).
    """
    train_path, test_path = download_nslkdd(data_dir)

    train_df = pd.read_csv(train_path, header=None, names=NSL_KDD_COLUMNS)
    test_df = pd.read_csv(test_path, header=None, names=NSL_KDD_COLUMNS)
    df = pd.concat([train_df, test_df], ignore_index=True)

    # Drop the difficulty level column (last column, not a feature)
    df.drop(columns=['difficulty_level'], inplace=True, errors='ignore')

    # Binary label: normal -> 0, everything else -> 1
    df['label'] = df['label'].apply(
        lambda x: 0 if str(x).strip().lower() in NORMAL_LABELS else 1
    )

    # One-hot encode categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'label' in cat_cols:
        cat_cols.remove('label')
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    labels = df['label'].values.astype(np.int64)
    features_df = df.drop(columns=['label'])
    features_df = features_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Sub-sample if too large
    if len(features_df) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(features_df), max_samples, replace=False)
        features_df = features_df.iloc[idx]
        labels = labels[idx]

    scaler = StandardScaler()
    features = scaler.fit_transform(features_df.values).astype(np.float32)

    return features, labels


def load_kddcup99(max_samples: int = 50000) -> tuple:
    """
    Load KDD Cup 99 dataset via sklearn (auto-downloads on first call).

    Returns:
        (features, labels) where labels are binary (0=normal, 1=attack).
    """
    from sklearn.datasets import fetch_kddcup99

    print("  Fetching KDD Cup 99 (SF subset) via sklearn ...")
    kdd = fetch_kddcup99(subset='SF', as_frame=True)
    df = kdd.frame  # type: ignore[union-attr]

    # Binary label
    df['labels'] = df['labels'].apply(
        lambda x: 0 if str(x).strip().rstrip('.').lower() == 'normal' else 1
    )
    labels = df['labels'].values.astype(np.int64)
    features_df = df.drop(columns=['labels'])

    # One-hot encode any remaining object columns
    cat_cols = features_df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        features_df = pd.get_dummies(features_df, columns=cat_cols, drop_first=False)

    features_df = features_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    if len(features_df) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(features_df), max_samples, replace=False)
        features_df = features_df.iloc[idx]
        labels = labels[idx]

    scaler = StandardScaler()
    features = scaler.fit_transform(features_df.values).astype(np.float32)
    print(f"  KDD Cup 99 loaded: {features.shape[0]} samples, "
          f"{features.shape[1]} features")

    return features, labels


# ---------------------------------------------------------------------------
# Load local CSV datasets (UNSW-NB15, CIC-IDS2017, etc.)
# ---------------------------------------------------------------------------

def load_dataset(data_path: str, dataset_name: str) -> pd.DataFrame:
    """Load a dataset from CSV files in a directory."""
    csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_path}")
    frames = [pd.read_csv(os.path.join(data_path, f), low_memory=False)
              for f in sorted(csv_files)]
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
    label_candidates = [c for c in df.columns
                        if c.lower() == label_col.lower()]
    if not label_candidates:
        label_candidates = [c for c in df.columns
                            if 'label' in c.lower() or 'class' in c.lower()]
    if not label_candidates:
        raise KeyError(f"No label column found matching '{label_col}'")
    actual_label = label_candidates[0]

    # Binary encode
    labels = df[actual_label].apply(
        lambda x: 0 if str(x).strip().upper()
        in ['BENIGN', 'NORMAL', '0', 'NONE'] else 1
    ).values

    # Drop non-numeric columns
    drop_cols = [actual_label]
    for col in df.columns:
        if df[col].dtype == 'object' and col != actual_label:
            drop_cols.append(col)
    features_df = df.drop(columns=drop_cols, errors='ignore')
    features_df = features_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    scaler = StandardScaler()
    features = scaler.fit_transform(features_df.values)

    return features.astype(np.float32), labels.astype(np.int64)


# ---------------------------------------------------------------------------
# Federated partitioning (non-IID via Dirichlet)
# ---------------------------------------------------------------------------

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
            partitions.append((np.empty((0, features.shape[1])),
                               np.empty(0, dtype=np.int64)))

    return partitions


# ---------------------------------------------------------------------------
# High-level entry point: download real data and partition
# ---------------------------------------------------------------------------

def download_and_prepare_data(dataset: str = 'nslkdd',
                              data_dir: str = 'data',
                              n_clients: int = 5,
                              alpha: float = 0.5,
                              max_samples: int = 50000,
                              test_size: float = 0.2) -> tuple:
    """
    Download real network intrusion data and partition for federated learning.

    Args:
        dataset: 'nslkdd' or 'kddcup99'.
        data_dir: Directory for caching downloaded files.
        n_clients: Number of federated clients.
        alpha: Dirichlet concentration parameter (lower = more non-IID).
        max_samples: Maximum total samples to use.
        test_size: Fraction held out for global test set.

    Returns:
        (partitions, test_data, input_dim)
            partitions: list of (features, labels) per client
            test_data: (test_features, test_labels)
            input_dim: number of features
    """
    print(f"\n[Data] Loading real dataset: {dataset.upper()}")

    if dataset == 'nslkdd':
        features, labels = load_nslkdd(
            data_dir=os.path.join(data_dir, 'nslkdd'),
            max_samples=max_samples,
        )
    elif dataset == 'kddcup99':
        features, labels = load_kddcup99(max_samples=max_samples)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. "
                         f"Use 'nslkdd' or 'kddcup99'.")

    input_dim = features.shape[1]

    # Train / test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size,
        stratify=labels, random_state=42,
    )

    print(f"[Data] Total samples : {len(features)}")
    print(f"[Data] Train samples : {len(X_train)}")
    print(f"[Data] Test samples  : {len(X_test)}")
    print(f"[Data] Feature dim   : {input_dim}")
    print(f"[Data] Attack ratio  : {labels.mean():.3f}")

    # Partition training data across clients (non-IID)
    partitions = partition_non_iid(X_train, y_train,
                                   n_clients=n_clients, alpha=alpha)

    return partitions, (X_test, y_test), input_dim


def load_local_dataset(data_path: str, label_col: str = 'label',
                       n_clients: int = 5, alpha: float = 0.5,
                       max_samples: int = 50000,
                       test_size: float = 0.2) -> tuple:
    """
    Load a local CSV dataset (e.g. UNSW-NB15, CIC-IDS2017) and partition
    for federated learning.

    Args:
        data_path: Directory containing CSV files.
        label_col: Name of the label column.
        n_clients: Number of federated clients.
        alpha: Dirichlet concentration parameter.
        max_samples: Maximum samples.
        test_size: Fraction held out for global test set.

    Returns:
        (partitions, test_data, input_dim)
    """
    print(f"\n[Data] Loading local dataset from {data_path}")
    df = load_dataset(data_path, dataset_name='local')
    features, labels = preprocess(df, label_col=label_col,
                                  max_samples=max_samples)
    input_dim = features.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size,
        stratify=labels, random_state=42,
    )

    print(f"[Data] Train samples : {len(X_train)}")
    print(f"[Data] Test samples  : {len(X_test)}")
    print(f"[Data] Feature dim   : {input_dim}")
    print(f"[Data] Attack ratio  : {labels.mean():.3f}")

    partitions = partition_non_iid(X_train, y_train,
                                   n_clients=n_clients, alpha=alpha)

    return partitions, (X_test, y_test), input_dim
