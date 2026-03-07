"""
Data loading and preprocessing for XAI NIDS Ensemble project.
Supports real KDD Cup 99 dataset (auto-download) and local UNSW-NB15 CSV files.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from scipy.stats import zscore


UNSW_FEATURE_NAMES = [
    'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl',
    'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit',
    'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat',
    'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src',
    'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
    'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd',
    'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports'
]


def load_unswnb15(data_path):
    """Load UNSW-NB15 dataset from CSV files."""
    train_file = f"{data_path}/UNSW_NB15_training-set.csv"
    test_file = f"{data_path}/UNSW_NB15_testing-set.csv"

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    df = pd.concat([train_df, test_df], ignore_index=True)

    if 'label' in df.columns:
        labels = df['label'].values
    elif 'attack_cat' in df.columns:
        labels = (df['attack_cat'] != 'Normal').astype(int).values

    drop_cols = ['id', 'label', 'attack_cat']
    feature_cols = [c for c in df.columns if c not in drop_cols]

    for col in feature_cols:
        if df[col].dtype == 'object':
            df[col] = pd.Categorical(df[col]).codes

    features = df[feature_cols].values.astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features, labels.astype(np.int64), feature_cols


def download_and_load_data(data_dir='data'):
    """
    Download and load real network intrusion detection data.

    Uses sklearn's built-in KDD Cup 99 dataset, a widely-used benchmark
    for network intrusion detection research. The 10% subset is used
    for manageable size while retaining all attack categories.

    Returns:
        features: np.ndarray of shape (n_samples, n_features)
        labels: np.ndarray of shape (n_samples,) with binary labels (0=normal, 1=attack)
        feature_names: list of feature name strings
    """
    from sklearn.datasets import fetch_kddcup99

    os.makedirs(data_dir, exist_ok=True)

    print("  Downloading KDD Cup 99 dataset (real network intrusion data)...")
    print("  Using 10% subset for manageable size...")
    kdd = fetch_kddcup99(as_frame=True, percent10=True, data_home=data_dir)
    df = kdd.frame

    print(f"  Raw dataset shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # Separate target
    target_col = df.columns[-1]  # 'labels' column
    labels_raw = df[target_col].copy()
    feature_df = df.drop(columns=[target_col])

    # Create binary labels: normal (0) vs attack (1)
    # KDD labels are bytes like b'normal.' or b'smurf.' etc.
    labels_str = labels_raw.astype(str).str.strip().str.rstrip('.')
    binary_labels = (labels_str != "b'normal'").astype(np.int64).values

    # Also try decoding if they are actual bytes
    try:
        decoded = labels_raw.apply(
            lambda x: x.decode('utf-8').strip().rstrip('.')
            if isinstance(x, bytes) else str(x).strip().rstrip('.')
        )
        binary_labels = (decoded != 'normal').astype(np.int64).values
    except Exception:
        pass

    print(f"  Label distribution: Normal={np.sum(binary_labels == 0)}, "
          f"Attack={np.sum(binary_labels == 1)}")

    # Encode categorical features (protocol_type, service, flag)
    le_dict = {}
    feature_names = list(feature_df.columns)
    for col in feature_df.columns:
        if feature_df[col].dtype == 'object' or feature_df[col].dtype.name == 'category':
            le = LabelEncoder()
            # Handle bytes
            feature_df[col] = feature_df[col].apply(
                lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x)
            )
            feature_df[col] = le.fit_transform(feature_df[col])
            le_dict[col] = le

    features = feature_df.values.astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  Final feature matrix: {features.shape}")
    print(f"  Number of features: {len(feature_names)}")
    print(f"  Attack ratio: {binary_labels.mean():.3f}")

    return features, binary_labels, feature_names


def preprocess_data(features, labels, feature_names=None, test_size=0.2,
                    val_size=0.2):
    """
    Preprocess pipeline: robust scaling, train/val/test split.
    Following UNSW-NB15 best practices with 60/20/20 split.
    """
    # Robust scaling (handles outliers better than StandardScaler)
    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(features)

    # Winsorisation: clip extreme values at 1st/99th percentile
    for i in range(features_scaled.shape[1]):
        p1, p99 = np.percentile(features_scaled[:, i], [1, 99])
        features_scaled[:, i] = np.clip(features_scaled[:, i], p1, p99)

    # Split: 60% train, 20% val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        features_scaled, labels, test_size=test_size,
        random_state=42, stratify=labels
    )

    val_frac = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_frac,
        random_state=42, stratify=y_temp
    )

    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"  Attack ratio - Train: {y_train.mean():.3f} | "
          f"Val: {y_val.mean():.3f} | Test: {y_test.mean():.3f}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
