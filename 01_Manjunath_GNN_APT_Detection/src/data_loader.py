"""
Data loading and preprocessing module for CIC-IDS2017 and ToN-IoT datasets.
Constructs graph representations from network flow data for GraphSAGE-based APT detection.

Supports three data sources:
  1. Auto-download: KDD Cup 99 via sklearn (real network intrusion data, works out of the box)
  2. Auto-download: CIC-IDS2017 subset from public mirror (attempted automatically)
  3. Local CSV loading: CIC-IDS2017 or ToN-IoT from a local directory
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# KDD Cup 99 auto-download (always works via sklearn)
# ---------------------------------------------------------------------------

def download_kddcup99(data_dir='data/kddcup99'):
    """
    Download the KDD Cup 99 intrusion detection dataset via sklearn.
    This is a REAL network intrusion dataset with ~494k records.
    Returns a pandas DataFrame with a 'Label' column (BENIGN / ATTACK).
    """
    from sklearn.datasets import fetch_kddcup99

    os.makedirs(data_dir, exist_ok=True)
    cache_path = os.path.join(data_dir, 'kddcup99_processed.csv')

    if os.path.exists(cache_path):
        print(f"  Loading cached KDD Cup 99 data from {cache_path}")
        return pd.read_csv(cache_path, low_memory=False)

    print("  Downloading KDD Cup 99 dataset via sklearn (this may take a moment)...")
    bunch = fetch_kddcup99(subset='SA', random_state=42, percent10=True)

    # Build dataframe from the sklearn Bunch
    feature_names = [
        'duration', 'protocol_type', 'service', 'flag',
        'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
        'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
        'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds',
        'is_host_login', 'is_guest_login', 'count', 'srv_count',
        'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
        'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate',
    ]

    data = bunch.data
    target = bunch.target

    # data is a structured numpy array; convert to DataFrame
    df = pd.DataFrame(data)
    # Use feature names only if column count matches
    if df.shape[1] == len(feature_names):
        df.columns = feature_names
    else:
        df.columns = [f'feature_{i}' for i in range(df.shape[1])]

    # Decode bytes to str for object columns
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    # Create binary label: normal. -> BENIGN, everything else -> ATTACK
    decoded_target = np.array([t.decode('utf-8') if isinstance(t, bytes) else str(t) for t in target])
    df['Label'] = np.where(np.char.startswith(decoded_target, 'normal'), 'BENIGN', 'ATTACK')

    # Create synthetic IP-like columns from count / srv_count for graph structure
    # Use hash of categorical features to create pseudo source/destination IDs
    if 'service' in df.columns and 'flag' in df.columns:
        df['Source IP'] = df['service'].astype(str).apply(hash).apply(lambda x: f"10.0.{abs(x) % 256}.{abs(x) // 256 % 256}")
        df['Destination IP'] = df['flag'].astype(str).apply(hash).apply(lambda x: f"192.168.{abs(x) % 256}.{abs(x) // 256 % 256}")
    else:
        df['Source IP'] = [f"10.0.{i % 256}.{i // 256 % 256}" for i in range(len(df))]
        df['Destination IP'] = [f"192.168.{i % 256}.{i // 256 % 256}" for i in range(len(df))]

    # Drop the original categorical columns (protocol_type, service, flag)
    # and use one-hot or just drop them since the numeric features are sufficient
    cat_cols = [c for c in df.columns if df[c].dtype == object and c not in ['Label', 'Source IP', 'Destination IP']]
    # One-hot encode categorical columns
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    df.to_csv(cache_path, index=False)
    print(f"  Saved processed data to {cache_path} ({len(df)} records)")
    return df


# ---------------------------------------------------------------------------
# CIC-IDS2017 download / local loading
# ---------------------------------------------------------------------------

def download_cicids2017(data_dir='data/cicids2017'):
    """
    Attempt to download CIC-IDS2017 CSV files from public mirrors.
    Falls back to instructions if download fails.
    Returns data_dir path if CSVs are present, None otherwise.
    """
    os.makedirs(data_dir, exist_ok=True)

    # Check if already downloaded
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if csv_files:
        print(f"  Found {len(csv_files)} CIC-IDS2017 CSV file(s) in {data_dir}")
        return data_dir

    # Attempt download from known public mirrors
    import urllib.request

    urls = [
        ("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
         "https://cse-cic-ids2017.s3.us-east-2.amazonaws.com/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"),
        ("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
         "https://cse-cic-ids2017.s3.us-east-2.amazonaws.com/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"),
        ("Friday-WorkingHours-Morning.pcap_ISCX.csv",
         "https://cse-cic-ids2017.s3.us-east-2.amazonaws.com/MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv"),
    ]

    downloaded = False
    for filename, url in urls:
        dest = os.path.join(data_dir, filename)
        if os.path.exists(dest):
            continue
        try:
            print(f"  Downloading {filename}...")
            urllib.request.urlretrieve(url, dest)
            print(f"    -> Saved to {dest}")
            downloaded = True
        except Exception as e:
            print(f"    -> Download failed: {e}")
            # Clean up partial downloads
            if os.path.exists(dest):
                os.remove(dest)

    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if csv_files:
        return data_dir

    # If download failed, print instructions
    print("\n  [INFO] Could not auto-download CIC-IDS2017. To use this dataset:")
    print("    1. Visit https://www.unb.ca/cic/datasets/ids-2017.html")
    print("    2. Download the MachineLearningCSV files")
    print(f"    3. Place CSV files in: {os.path.abspath(data_dir)}")
    print("    Alternatively, download from Kaggle:")
    print("    https://www.kaggle.com/datasets/cicdataset/cicids2017\n")
    return None


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


# ---------------------------------------------------------------------------
# Preprocessing and graph construction (unchanged logic)
# ---------------------------------------------------------------------------

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


def _add_masks(graph, labels):
    """Add train/val/test masks to a graph via stratified splitting."""
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


# ---------------------------------------------------------------------------
# High-level entry points
# ---------------------------------------------------------------------------

def prepare_dataset(data_path: str, dataset_type: str = 'cicids2017',
                    label_col: str = 'Label', max_samples: int = 100000) -> Data:
    """
    End-to-end data preparation pipeline for local CSV datasets.

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
    graph = _add_masks(graph, labels)
    return graph


def download_and_prepare_dataset(max_samples: int = 50000) -> Data:
    """
    Auto-download a real network intrusion dataset and prepare it as a graph.

    Strategy:
      1. Try to download/load CIC-IDS2017 from public mirrors.
      2. If CIC-IDS2017 is available locally, use it.
      3. Otherwise, fall back to KDD Cup 99 via sklearn (guaranteed to work).

    Returns:
        PyTorch Geometric Data object built from REAL intrusion detection data.
    """
    # --- Attempt 1: CIC-IDS2017 ---
    cicids_dir = 'data/cicids2017'
    cicids_path = download_cicids2017(cicids_dir)
    if cicids_path is not None:
        print("  Using CIC-IDS2017 dataset.")
        try:
            return prepare_dataset(cicids_path, 'cicids2017', 'Label', max_samples)
        except Exception as e:
            print(f"  Failed to process CIC-IDS2017: {e}")
            print("  Falling back to KDD Cup 99...")

    # --- Fallback: KDD Cup 99 (always works) ---
    print("  Using KDD Cup 99 dataset (real network intrusion data via sklearn).")
    df = download_kddcup99('data/kddcup99')

    features, labels, src_ips, dst_ips = preprocess_dataframe(
        df, label_col='Label', max_samples=max_samples
    )

    graph = build_graph(features, labels, src_ips, dst_ips)
    graph = _add_masks(graph, labels)
    return graph
