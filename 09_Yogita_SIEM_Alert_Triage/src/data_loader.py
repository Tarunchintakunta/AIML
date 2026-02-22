"""
data_loader.py - Synthetic SIEM Alert Data Generator
=====================================================
Generates synthetic SIEM alert datasets modeled after CIC-IDS2017,
NSL-KDD, and UNSW-NB15 with realistic class imbalance.

3-Class Labels:
    0 = False Positive  (85%)
    1 = Indeterminate   (10%)
    2 = True Positive   ( 5%)

20 features total capturing network traffic and alert metadata.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "alert_severity",          # 1-5 ordinal
    "src_port",                # source port number
    "dst_port",               # destination port number
    "protocol_encoded",        # encoded protocol (TCP/UDP/ICMP/Other)
    "packet_count",            # packets in session
    "byte_count",              # total bytes transferred
    "duration",                # session duration (seconds)
    "alert_type_encoded",      # encoded alert category
    "time_of_day",             # hour of alert (0-23)
    "day_of_week",             # day (0=Mon, 6=Sun)
    "src_ip_entropy",          # entropy of source IP distribution
    "dst_ip_entropy",          # entropy of destination IP distribution
    "flag_count",              # TCP flag anomaly count
    "payload_size_mean",       # mean payload size
    "payload_size_std",        # std-dev of payload size
    "connection_rate",         # connections per minute from source
    "failed_login_count",      # failed logins in window
    "dns_query_count",         # DNS queries in window
    "is_internal_src",         # 1 if source is internal
    "reputation_score",        # threat-intel reputation (0-100)
]

CLASS_NAMES = ["False Positive", "Indeterminate", "True Positive"]
CLASS_DISTRIBUTION = [0.85, 0.10, 0.05]  # FP, Indet, TP


def _generate_class_features(n_samples: int, label: int,
                             rng: np.random.Generator) -> np.ndarray:
    """Generate feature matrix for a single class with class-specific
    distributional properties so that the problem is learnable but not
    trivially separable."""

    X = np.zeros((n_samples, 20))

    if label == 0:  # False Positive ------------------------------------------
        X[:, 0] = rng.choice([1, 2, 3], size=n_samples, p=[0.5, 0.35, 0.15])
        X[:, 1] = rng.integers(1024, 65535, size=n_samples)
        X[:, 2] = rng.choice([80, 443, 8080, 8443, 53],
                              size=n_samples, p=[0.3, 0.3, 0.15, 0.15, 0.1])
        X[:, 3] = rng.choice([0, 1, 2, 3], size=n_samples,
                              p=[0.6, 0.25, 0.1, 0.05])
        X[:, 4] = rng.exponential(50, size=n_samples) + 1
        X[:, 5] = rng.exponential(5000, size=n_samples) + 100
        X[:, 6] = rng.exponential(10, size=n_samples) + 0.1
        X[:, 7] = rng.choice(range(10), size=n_samples)
        X[:, 8] = rng.integers(0, 24, size=n_samples)
        X[:, 9] = rng.integers(0, 7, size=n_samples)
        X[:, 10] = rng.normal(3.5, 0.8, size=n_samples)
        X[:, 11] = rng.normal(3.0, 0.7, size=n_samples)
        X[:, 12] = rng.poisson(1, size=n_samples)
        X[:, 13] = rng.normal(500, 200, size=n_samples)
        X[:, 14] = rng.exponential(100, size=n_samples)
        X[:, 15] = rng.exponential(5, size=n_samples) + 0.5
        X[:, 16] = rng.poisson(0.3, size=n_samples)
        X[:, 17] = rng.poisson(5, size=n_samples)
        X[:, 18] = rng.choice([0, 1], size=n_samples, p=[0.3, 0.7])
        X[:, 19] = rng.normal(15, 10, size=n_samples)

    elif label == 1:  # Indeterminate -----------------------------------------
        X[:, 0] = rng.choice([2, 3, 4], size=n_samples, p=[0.3, 0.45, 0.25])
        X[:, 1] = rng.integers(1024, 65535, size=n_samples)
        X[:, 2] = rng.choice([80, 443, 22, 3389, 445],
                              size=n_samples, p=[0.2, 0.2, 0.25, 0.2, 0.15])
        X[:, 3] = rng.choice([0, 1, 2, 3], size=n_samples,
                              p=[0.4, 0.3, 0.2, 0.1])
        X[:, 4] = rng.exponential(120, size=n_samples) + 5
        X[:, 5] = rng.exponential(15000, size=n_samples) + 500
        X[:, 6] = rng.exponential(30, size=n_samples) + 1.0
        X[:, 7] = rng.choice(range(10), size=n_samples)
        # Indeterminate alerts slightly more common in evening/night
        p_indet_hour = np.array([0.025]*8 + [0.05]*4 + [0.025]*4 + [0.075]*4 + [0.05]*4)
        p_indet_hour /= p_indet_hour.sum()
        X[:, 8] = rng.choice(range(24), size=n_samples, p=p_indet_hour)
        X[:, 9] = rng.integers(0, 7, size=n_samples)
        X[:, 10] = rng.normal(4.5, 1.0, size=n_samples)
        X[:, 11] = rng.normal(4.0, 1.0, size=n_samples)
        X[:, 12] = rng.poisson(3, size=n_samples)
        X[:, 13] = rng.normal(800, 300, size=n_samples)
        X[:, 14] = rng.exponential(200, size=n_samples)
        X[:, 15] = rng.exponential(15, size=n_samples) + 2
        X[:, 16] = rng.poisson(2, size=n_samples)
        X[:, 17] = rng.poisson(12, size=n_samples)
        X[:, 18] = rng.choice([0, 1], size=n_samples, p=[0.5, 0.5])
        X[:, 19] = rng.normal(45, 15, size=n_samples)

    else:  # True Positive (label == 2) ---------------------------------------
        X[:, 0] = rng.choice([3, 4, 5], size=n_samples, p=[0.15, 0.35, 0.5])
        X[:, 1] = rng.integers(1024, 65535, size=n_samples)
        X[:, 2] = rng.choice([22, 3389, 445, 4444, 8888],
                              size=n_samples, p=[0.25, 0.2, 0.2, 0.2, 0.15])
        X[:, 3] = rng.choice([0, 1, 2, 3], size=n_samples,
                              p=[0.3, 0.2, 0.3, 0.2])
        X[:, 4] = rng.exponential(300, size=n_samples) + 20
        X[:, 5] = rng.exponential(50000, size=n_samples) + 2000
        X[:, 6] = rng.exponential(60, size=n_samples) + 5.0
        X[:, 7] = rng.choice(range(10), size=n_samples)
        # TP attacks more common at night (0-5) and late evening (18-23)
        p_tp_hour = np.array([0.0625]*6 + [0.02083]*6 + [0.02084]*6 + [0.0625]*6)
        p_tp_hour /= p_tp_hour.sum()
        X[:, 8] = rng.choice(range(24), size=n_samples, p=p_tp_hour)
        X[:, 9] = rng.integers(0, 7, size=n_samples)
        X[:, 10] = rng.normal(6.0, 1.2, size=n_samples)
        X[:, 11] = rng.normal(5.5, 1.1, size=n_samples)
        X[:, 12] = rng.poisson(6, size=n_samples)
        X[:, 13] = rng.normal(1200, 400, size=n_samples)
        X[:, 14] = rng.exponential(350, size=n_samples)
        X[:, 15] = rng.exponential(40, size=n_samples) + 5
        X[:, 16] = rng.poisson(5, size=n_samples)
        X[:, 17] = rng.poisson(25, size=n_samples)
        X[:, 18] = rng.choice([0, 1], size=n_samples, p=[0.7, 0.3])
        X[:, 19] = rng.normal(75, 15, size=n_samples)

    return X


def generate_synthetic_siem_data(
    n_samples: int = 20000,
    random_state: int = 42,
    dataset_name: str = "combined",
) -> pd.DataFrame:
    """
    Generate a synthetic SIEM alert dataset.

    Parameters
    ----------
    n_samples : int
        Total number of alert samples.
    random_state : int
        Seed for reproducibility.
    dataset_name : str
        One of 'cic-ids2017', 'nsl-kdd', 'unsw-nb15', or 'combined'.
        This tag is stored in a 'source_dataset' column but does not
        change the generation logic (all three are synthetic).

    Returns
    -------
    pd.DataFrame
        DataFrame with 20 feature columns, a 'label' column (0/1/2),
        and a 'source_dataset' column.
    """
    rng = np.random.default_rng(random_state)

    class_counts = [
        int(n_samples * CLASS_DISTRIBUTION[0]),
        int(n_samples * CLASS_DISTRIBUTION[1]),
        0,  # placeholder
    ]
    class_counts[2] = n_samples - class_counts[0] - class_counts[1]

    X_parts, y_parts = [], []
    for label, count in enumerate(class_counts):
        X_cls = _generate_class_features(count, label, rng)
        X_parts.append(X_cls)
        y_parts.append(np.full(count, label))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    # Clip non-negative features
    for col in [0, 4, 5, 6, 12, 13, 14, 15, 16, 17]:
        X[:, col] = np.clip(X[:, col], 0, None)
    X[:, 19] = np.clip(X[:, 19], 0, 100)

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["label"] = y.astype(int)

    # Assign source dataset tags
    if dataset_name == "combined":
        tags = rng.choice(
            ["CIC-IDS2017", "NSL-KDD", "UNSW-NB15"],
            size=n_samples, p=[0.4, 0.3, 0.3],
        )
        df["source_dataset"] = tags
    else:
        df["source_dataset"] = dataset_name

    # Shuffle
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
):
    """
    Split and scale the data for modelling.

    Returns
    -------
    dict with keys: X_train, X_val, X_test, y_train, y_val, y_test,
                    scaler, feature_names
    """
    feature_cols = [c for c in df.columns if c not in ("label", "source_dataset")]
    X = df[feature_cols].values
    y = df["label"].values

    # Train / temp split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + val_size,
        random_state=random_state, stratify=y,
    )
    # Val / test split
    relative_val = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - relative_val,
        random_state=random_state, stratify=y_temp,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": feature_cols,
    }


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = generate_synthetic_siem_data(n_samples=5000)
    print(f"Shape: {df.shape}")
    print(f"\nClass distribution:\n{df['label'].value_counts().sort_index()}")
    print(f"\nDataset sources:\n{df['source_dataset'].value_counts()}")
    print(f"\nFeature summary:\n{df.describe().T[['mean', 'std', 'min', 'max']]}")
