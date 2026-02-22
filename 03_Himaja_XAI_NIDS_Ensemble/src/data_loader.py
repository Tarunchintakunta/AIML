"""
Data loading and preprocessing for XAI NIDS Ensemble project.
Supports UNSW-NB15 dataset and synthetic data generation.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
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


def generate_synthetic_data(n_samples=10000, n_features=39):
    """Generate synthetic UNSW-NB15-like data for demonstration."""
    np.random.seed(42)

    n_benign = int(n_samples * 0.7)
    n_attack = n_samples - n_benign

    # Benign traffic patterns
    benign = np.random.randn(n_benign, n_features).astype(np.float32)
    benign[:, 0] = np.abs(np.random.exponential(0.5, n_benign))  # dur
    benign[:, 3] = np.abs(np.random.exponential(500, n_benign))  # sbytes
    benign[:, 4] = np.abs(np.random.exponential(800, n_benign))  # dbytes
    benign[:, 6] = np.random.choice([62, 64, 128, 254], n_benign)  # sttl

    # Attack traffic patterns with shifted distributions
    attack = np.random.randn(n_attack, n_features).astype(np.float32) + 0.8
    attack[:, 0] = np.abs(np.random.exponential(2.0, n_attack))  # longer dur
    attack[:, 3] = np.abs(np.random.exponential(2000, n_attack))  # more sbytes
    attack[:, 4] = np.abs(np.random.exponential(100, n_attack))  # less dbytes
    attack[:, 6] = np.random.choice([32, 128, 252, 255], n_attack)  # sttl
    attack[:, 5] = np.abs(np.random.exponential(5000, n_attack))  # rate
    attack[:, 8] = np.abs(np.random.exponential(3000, n_attack))  # sload

    features = np.vstack([benign, attack])
    labels = np.array([0]*n_benign + [1]*n_attack, dtype=np.int64)

    shuffle_idx = np.random.permutation(n_samples)
    features = features[shuffle_idx]
    labels = labels[shuffle_idx]

    feature_names = UNSW_FEATURE_NAMES if n_features == 39 else \
        [f'feature_{i}' for i in range(n_features)]

    return features, labels, feature_names


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
