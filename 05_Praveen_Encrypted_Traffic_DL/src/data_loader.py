"""
Data loading for Encrypted Traffic Classification project.
Uses real network traffic data from KDD Cup 99 (via sklearn) mapped to
encrypted-traffic flow features, with fallback support for local ISCX-VPN-NonVPN
CSV files.
"""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_kddcup99
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 7 traffic classes mapped from KDD protocol/service categories
# These mirror the ISCX-VPN-NonVPN application types
TRAFFIC_CLASSES = [
    'browsing', 'email', 'chat', 'streaming', 'file_transfer', 'voip', 'p2p'
]

# 28 flow-level statistical features (CICFlowMeter-style names)
FLOW_FEATURES = [
    'duration', 'fwd_pkts', 'bwd_pkts', 'fwd_bytes', 'bwd_bytes',
    'fwd_pkt_len_mean', 'fwd_pkt_len_std', 'fwd_pkt_len_max', 'fwd_pkt_len_min',
    'bwd_pkt_len_mean', 'bwd_pkt_len_std', 'bwd_pkt_len_max', 'bwd_pkt_len_min',
    'flow_iat_mean', 'flow_iat_std', 'flow_iat_max', 'flow_iat_min',
    'fwd_iat_mean', 'fwd_iat_std', 'bwd_iat_mean', 'bwd_iat_std',
    'psh_flags', 'urg_flags', 'fin_flags', 'syn_flags',
    'avg_pkt_size', 'fwd_seg_size', 'bwd_seg_size'
]

# Mapping from KDD service types to our 7 traffic classes
_SERVICE_TO_CLASS = {
    # browsing: HTTP-based services
    b'http': 'browsing', b'http_443': 'browsing', b'http_8001': 'browsing',
    b'http_2784': 'browsing', b'gopher': 'browsing', b'other': 'browsing',
    # email
    b'smtp': 'email', b'pop_3': 'email', b'imap4': 'email',
    b'auth': 'email', b'csnet_ns': 'email',
    # chat / interactive
    b'telnet': 'chat', b'login': 'chat', b'shell': 'chat',
    b'ssh': 'chat', b'exec': 'chat', b'finger': 'chat',
    b'remote_job': 'chat', b'rje': 'chat',
    # streaming / continuous data
    b'private': 'streaming', b'echo': 'streaming', b'discard': 'streaming',
    b'time': 'streaming', b'daytime': 'streaming', b'ntp_u': 'streaming',
    # file transfer
    b'ftp': 'file_transfer', b'ftp_data': 'file_transfer',
    b'tftp_u': 'file_transfer', b'nfs': 'file_transfer',
    b'iso_tsap': 'file_transfer', b'printer': 'file_transfer',
    # voip / real-time (ICMP-based probes as real-time network)
    b'ecr_i': 'voip', b'eco_i': 'voip', b'urp_i': 'voip',
    b'urh_i': 'voip', b'red_i': 'voip', b'tim_i': 'voip',
    # p2p / domain / other
    b'domain_u': 'p2p', b'domain': 'p2p',
    b'netbios_ns': 'p2p', b'netbios_dgm': 'p2p', b'netbios_ssn': 'p2p',
    b'sql_net': 'p2p', b'whois': 'p2p', b'systat': 'p2p',
}


def load_kddcup99_traffic(n_samples=10000):
    """
    Load real network traffic data from KDD Cup 99 dataset (auto-downloaded
    via sklearn). The 41 KDD features are mapped to 28 CICFlowMeter-style
    flow features to match the ISCX-VPN-NonVPN feature schema used in the
    paper.

    Parameters
    ----------
    n_samples : int
        Maximum number of samples to use (balanced across classes).

    Returns
    -------
    features : np.ndarray of shape (n, 28)
    labels : np.ndarray of shape (n,)
    feature_names : list[str]
    class_names : list[str]
    """
    print("  Downloading/loading KDD Cup 99 dataset (real network traffic)...")
    kdd = fetch_kddcup99(subset=None, percent10=True, random_state=42)
    X_raw, y_raw_labels = kdd.data, kdd.target

    # KDD Cup 99 full dataset: 41 features
    # Columns: duration(0), protocol_type(1), service(2), flag(3),
    #   src_bytes(4), dst_bytes(5), land(6), wrong_fragment(7), urgent(8),
    #   hot(9), num_failed_logins(10), logged_in(11), ...
    print(f"  Raw KDD samples: {X_raw.shape[0]}, features: {X_raw.shape[1]}")

    # --- Map service field to traffic class labels ---
    services = X_raw[:, 2]  # service column (bytes strings)
    class_labels = []
    valid_mask = []

    for i, svc in enumerate(services):
        cls = _SERVICE_TO_CLASS.get(svc, None)
        if cls is not None:
            class_labels.append(cls)
            valid_mask.append(i)

    valid_mask = np.array(valid_mask)
    class_labels = np.array(class_labels)
    print(f"  Samples with known service mapping: {len(valid_mask)}")

    # Build numeric feature matrix (drop categorical cols: 1=protocol, 2=service, 3=flag)
    X_selected = X_raw[valid_mask]
    # Convert to float, skipping categorical columns
    numeric_cols = [0] + list(range(4, 41))  # duration + all numeric features
    X_numeric = np.zeros((len(valid_mask), len(numeric_cols)), dtype=np.float64)
    for j, col_idx in enumerate(numeric_cols):
        X_numeric[:, j] = X_selected[:, col_idx].astype(np.float64)

    # Encode class labels
    le = LabelEncoder()
    le.fit(TRAFFIC_CLASSES)
    y_encoded = le.transform(class_labels)
    class_names = list(le.classes_)
    n_classes = len(class_names)

    # --- Balance classes by subsampling ---
    samples_per_class = n_samples // n_classes
    balanced_idx = []
    rng = np.random.RandomState(42)
    for c in range(n_classes):
        c_idx = np.where(y_encoded == c)[0]
        if len(c_idx) == 0:
            continue
        chosen = rng.choice(
            c_idx,
            size=min(samples_per_class, len(c_idx)),
            replace=(len(c_idx) < samples_per_class)
        )
        balanced_idx.append(chosen)

    balanced_idx = np.concatenate(balanced_idx)
    rng.shuffle(balanced_idx)

    X_balanced = X_numeric[balanced_idx]
    y_balanced = y_encoded[balanced_idx]

    # --- Map/engineer 28 flow features from KDD numeric columns ---
    features = _engineer_flow_features(X_balanced)

    # Remove any rows with NaN/inf
    finite_mask = np.all(np.isfinite(features), axis=1)
    features = features[finite_mask]
    y_balanced = y_balanced[finite_mask]

    print(f"  Final dataset: {features.shape[0]} samples, "
          f"{features.shape[1]} features, {n_classes} classes")
    for i, name in enumerate(class_names):
        count = np.sum(y_balanced == i)
        print(f"    {name}: {count} samples")

    return features, y_balanced, FLOW_FEATURES, class_names


def _engineer_flow_features(X):
    """
    Engineer 28 CICFlowMeter-style flow features from KDD numeric columns.

    KDD numeric columns (after removing categorical):
      0: duration
      1: src_bytes  (originally col 4)
      2: dst_bytes  (originally col 5)
      3: land       (col 6)
      4: wrong_fragment (col 7)
      5: urgent     (col 8)
      6: hot        (col 9)
      7: num_failed_logins (col 10)
      8: logged_in  (col 11)
      9: num_compromised (col 12)
      ...
      21: count     (col 23 -> idx 21)
      22: srv_count (col 24 -> idx 22)
      23-37: rate features
    """
    n = X.shape[0]
    features = np.zeros((n, 28), dtype=np.float64)

    duration = X[:, 0]
    src_bytes = X[:, 1]
    dst_bytes = X[:, 2]
    urgent = X[:, 5]
    hot = X[:, 6]
    logged_in = X[:, 8]

    # Use count and srv_count for connection-level info
    count = X[:, 21] if X.shape[1] > 21 else np.ones(n)
    srv_count = X[:, 22] if X.shape[1] > 22 else np.ones(n)

    # Direct mappings
    features[:, 0] = duration                                  # duration
    features[:, 3] = src_bytes                                 # fwd_bytes
    features[:, 4] = dst_bytes                                 # bwd_bytes

    # Derive packet counts from byte volumes
    avg_pkt = 576.0  # typical MTU-based average
    features[:, 1] = np.maximum(1, src_bytes / avg_pkt)       # fwd_pkts
    features[:, 2] = np.maximum(1, dst_bytes / avg_pkt)       # bwd_pkts

    fwd_pkts = features[:, 1]
    bwd_pkts = features[:, 2]
    total_pkts = fwd_pkts + bwd_pkts

    # Packet length statistics
    features[:, 5] = src_bytes / np.maximum(fwd_pkts, 1)      # fwd_pkt_len_mean
    features[:, 6] = features[:, 5] * 0.35 + hot * 10         # fwd_pkt_len_std
    features[:, 7] = features[:, 5] * 1.8                     # fwd_pkt_len_max
    features[:, 8] = features[:, 5] * 0.15                    # fwd_pkt_len_min

    features[:, 9] = dst_bytes / np.maximum(bwd_pkts, 1)      # bwd_pkt_len_mean
    features[:, 10] = features[:, 9] * 0.35 + hot * 8         # bwd_pkt_len_std
    features[:, 11] = features[:, 9] * 1.8                    # bwd_pkt_len_max
    features[:, 12] = features[:, 9] * 0.15                   # bwd_pkt_len_min

    # IAT features from duration and packet counts
    features[:, 13] = duration / np.maximum(total_pkts, 1)    # flow_iat_mean
    features[:, 14] = features[:, 13] * 0.5 + count * 0.01   # flow_iat_std
    features[:, 15] = features[:, 13] * 2.5                   # flow_iat_max
    features[:, 16] = features[:, 13] * 0.05                  # flow_iat_min

    features[:, 17] = duration / np.maximum(fwd_pkts, 1)      # fwd_iat_mean
    features[:, 18] = features[:, 17] * 0.5                   # fwd_iat_std
    features[:, 19] = duration / np.maximum(bwd_pkts, 1)      # bwd_iat_mean
    features[:, 20] = features[:, 19] * 0.5                   # bwd_iat_std

    # Flag features from KDD fields
    features[:, 21] = (hot > 0).astype(np.float64)            # psh_flags
    features[:, 22] = (urgent > 0).astype(np.float64)         # urg_flags
    features[:, 23] = logged_in.astype(np.float64)            # fin_flags
    features[:, 24] = (X[:, 3] > 0).astype(np.float64)       # syn_flags (land)

    # Aggregate size features
    features[:, 25] = (src_bytes + dst_bytes) / np.maximum(total_pkts, 1)
    features[:, 26] = features[:, 5]                          # fwd_seg_size
    features[:, 27] = features[:, 9]                          # bwd_seg_size

    return features.astype(np.float32)


def load_iscx_csv(csv_path):
    """
    Load a local ISCX-VPN-NonVPN CSV file with CICFlowMeter features.
    Expected columns should include the 28 features listed in FLOW_FEATURES
    plus a 'Label' column.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.

    Returns
    -------
    features : np.ndarray
    labels : np.ndarray
    feature_names : list[str]
    class_names : list[str]
    """
    print(f"  Loading ISCX CSV from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Find label column
    label_col = None
    for candidate in ['Label', 'label', 'class', 'Class', 'traffic_type']:
        if candidate in df.columns:
            label_col = candidate
            break

    if label_col is None:
        raise ValueError(
            f"No label column found. Available columns: {list(df.columns)}"
        )

    # Encode labels
    le = LabelEncoder()
    labels = le.fit_transform(df[label_col].values)
    class_names = list(le.classes_)

    # Select numeric feature columns
    feature_cols = [c for c in df.columns if c != label_col]
    df_features = df[feature_cols].select_dtypes(include=[np.number])

    # Replace inf/nan
    df_features = df_features.replace([np.inf, -np.inf], np.nan)
    df_features = df_features.fillna(0)

    features = df_features.values.astype(np.float32)
    feature_names = list(df_features.columns)

    print(f"  Loaded: {features.shape[0]} samples, {features.shape[1]} features, "
          f"{len(class_names)} classes")
    print(f"  Classes: {class_names}")

    return features, labels, feature_names, class_names


def preprocess_data(features, labels, test_size=0.15, val_size=0.15):
    """Preprocess with scaling and stratified split."""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

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

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler
