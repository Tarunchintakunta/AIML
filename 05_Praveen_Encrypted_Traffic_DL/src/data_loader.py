"""
Data loading for Encrypted Traffic Classification project.
Supports ISCX-VPN-NonVPN dataset and synthetic data.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 7 traffic classes: VPN and non-VPN variants
TRAFFIC_CLASSES = [
    'browsing', 'email', 'chat', 'streaming', 'file_transfer', 'voip', 'p2p'
]

# 28 flow-level statistical features from ISCX-VPN-NonVPN
FLOW_FEATURES = [
    'duration', 'fwd_pkts', 'bwd_pkts', 'fwd_bytes', 'bwd_bytes',
    'fwd_pkt_len_mean', 'fwd_pkt_len_std', 'fwd_pkt_len_max', 'fwd_pkt_len_min',
    'bwd_pkt_len_mean', 'bwd_pkt_len_std', 'bwd_pkt_len_max', 'bwd_pkt_len_min',
    'flow_iat_mean', 'flow_iat_std', 'flow_iat_max', 'flow_iat_min',
    'fwd_iat_mean', 'fwd_iat_std', 'bwd_iat_mean', 'bwd_iat_std',
    'psh_flags', 'urg_flags', 'fin_flags', 'syn_flags',
    'avg_pkt_size', 'fwd_seg_size', 'bwd_seg_size'
]


def generate_synthetic_data(n_samples=10000, n_classes=7, n_features=28):
    """Generate synthetic ISCX-VPN-NonVPN-like traffic flow data."""
    np.random.seed(42)

    samples_per_class = n_samples // n_classes
    features_list = []
    labels_list = []

    for cls_id in range(n_classes):
        # Each class has distinct statistical patterns
        base_shift = cls_id * 0.5
        cls_features = np.random.randn(samples_per_class, n_features).astype(np.float32)

        # Duration varies by application type
        cls_features[:, 0] = np.abs(np.random.exponential(
            (cls_id + 1) * 2, samples_per_class))
        # Packet counts
        cls_features[:, 1] = np.abs(np.random.poisson(
            (cls_id + 1) * 10, samples_per_class))
        cls_features[:, 2] = np.abs(np.random.poisson(
            (cls_id + 1) * 8, samples_per_class))
        # Byte volumes
        cls_features[:, 3] = np.abs(np.random.exponential(
            (cls_id + 1) * 500, samples_per_class))
        cls_features[:, 4] = np.abs(np.random.exponential(
            (cls_id + 1) * 400, samples_per_class))
        # Packet length stats with class-specific patterns
        cls_features[:, 5:9] += base_shift
        cls_features[:, 9:13] += base_shift * 0.8
        # IAT features
        cls_features[:, 13:17] = np.abs(cls_features[:, 13:17]) + cls_id * 0.3
        # Flags
        cls_features[:, 21:25] = np.random.binomial(
            1, 0.1 * (cls_id + 1) / n_classes,
            (samples_per_class, 4)).astype(np.float32)

        features_list.append(cls_features)
        labels_list.append(np.full(samples_per_class, cls_id, dtype=np.int64))

    features = np.vstack(features_list)
    labels = np.concatenate(labels_list)

    shuffle_idx = np.random.permutation(len(labels))
    features = features[shuffle_idx]
    labels = labels[shuffle_idx]

    return features, labels, FLOW_FEATURES[:n_features], TRAFFIC_CLASSES[:n_classes]


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
