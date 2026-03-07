"""
data_loader.py - Real Dataset Loader for SIEM Alert Triage
===========================================================
Downloads and preprocesses the NSL-KDD dataset, mapping its intrusion
labels to 3-class SIEM triage categories:

    0 = False Positive   (normal traffic)
    1 = Indeterminate    (probe / low-severity attacks)
    2 = True Positive    (DoS, R2L, U2R / high-severity attacks)

Primary source: NSL-KDD from GitHub (KDDTrain+.txt, KDDTest+.txt)
Fallback: sklearn.datasets.fetch_kddcup99
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ---------------------------------------------------------------------------
# NSL-KDD column definitions
# ---------------------------------------------------------------------------
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

CLASS_NAMES = ["False Positive", "Indeterminate", "True Positive"]

# ---------------------------------------------------------------------------
# NSL-KDD label -> SIEM triage mapping
# ---------------------------------------------------------------------------
# Probe attacks (low severity) -> Indeterminate (class 1)
PROBE_ATTACKS = {
    "nmap", "ipsweep", "portsweep", "satan", "saint", "mscan",
}

# High-severity attacks -> True Positive (class 2)
# Includes DoS, R2L, U2R
HIGH_SEVERITY_ATTACKS = {
    # DoS
    "neptune", "smurf", "back", "teardrop", "pod", "land",
    "apache2", "udpstorm", "processtable", "mailbomb",
    # R2L
    "guess_passwd", "ftp_write", "imap", "phf", "multihop",
    "warezmaster", "warezclient", "spy", "xlock", "xsnoop",
    "snmpguess", "snmpgetattack", "httptunnel", "sendmail",
    "named", "worm",
    # U2R
    "buffer_overflow", "rootkit", "loadmodule", "perl",
    "sqlattack", "xterm", "ps", "httptunnel",
}

NSL_KDD_TRAIN_URL = (
    "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
)
NSL_KDD_TEST_URL = (
    "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"
)


def _map_label_to_triage(label_str: str) -> int:
    """Map an NSL-KDD attack label to a 3-class SIEM triage category."""
    label_str = label_str.strip().lower()
    if label_str == "normal":
        return 0  # False Positive
    elif label_str in PROBE_ATTACKS:
        return 1  # Indeterminate
    else:
        # Everything else (DoS, R2L, U2R, and any unknown attack) -> True Positive
        return 2


def load_nsl_kdd(data_dir: str = None) -> pd.DataFrame:
    """
    Download and load the NSL-KDD dataset.

    Tries to download from GitHub first. Falls back to sklearn's
    fetch_kddcup99 if the download fails.

    Parameters
    ----------
    data_dir : str, optional
        Directory to cache downloaded files. Defaults to project data/ dir.

    Returns
    -------
    pd.DataFrame
        Combined train+test DataFrame with triage labels and encoded features.
    """
    if data_dir is None:
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
        )
    os.makedirs(data_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "KDDTrain+.txt")
    test_path = os.path.join(data_dir, "KDDTest+.txt")

    try:
        df = _load_from_github(train_path, test_path)
        print(f"  Loaded NSL-KDD from GitHub ({len(df):,} samples)")
    except Exception as e:
        print(f"  GitHub download failed: {e}")
        print("  Falling back to sklearn fetch_kddcup99 ...")
        df = _load_from_sklearn()
        print(f"  Loaded KDD Cup 99 via sklearn ({len(df):,} samples)")

    return df


def _load_from_github(train_path: str, test_path: str) -> pd.DataFrame:
    """Download NSL-KDD from GitHub and return combined DataFrame."""
    import urllib.request

    # Download if not cached
    for url, path in [(NSL_KDD_TRAIN_URL, train_path),
                      (NSL_KDD_TEST_URL, test_path)]:
        if not os.path.exists(path):
            print(f"  Downloading {os.path.basename(path)} ...")
            urllib.request.urlretrieve(url, path)

    # Load CSVs
    df_train = pd.read_csv(train_path, header=None, names=NSL_KDD_COLUMNS)
    df_test = pd.read_csv(test_path, header=None, names=NSL_KDD_COLUMNS)

    df_train["split"] = "train"
    df_test["split"] = "test"

    df = pd.concat([df_train, df_test], ignore_index=True)

    # Map labels to triage classes
    df["triage_label"] = df["label"].apply(_map_label_to_triage)

    # Encode categorical features
    label_encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Drop original label and difficulty columns, rename triage_label
    df = df.drop(columns=["label", "difficulty"])
    df = df.rename(columns={"triage_label": "label"})

    # Add source dataset tag
    df["source_dataset"] = "NSL-KDD"

    return df


def _load_from_sklearn() -> pd.DataFrame:
    """Fallback: load KDD Cup 99 (10%) via sklearn and map to triage labels."""
    from sklearn.datasets import fetch_kddcup99

    bunch = fetch_kddcup99(subset=None, percent10=True, as_frame=True)
    df = bunch.frame.copy()

    # Rename target column
    df = df.rename(columns={"labels": "label_raw"})

    # Clean label strings (sklearn returns bytes or strings with trailing '.')
    df["label_raw"] = df["label_raw"].apply(
        lambda x: x.decode("utf-8").rstrip(".") if isinstance(x, bytes)
        else str(x).rstrip(".")
    )

    # Map to triage
    df["label"] = df["label_raw"].apply(_map_label_to_triage)

    # Encode categoricals
    categorical_sklearn = ["protocol_type", "service", "flag"]
    for col in categorical_sklearn:
        if col in df.columns:
            # sklearn may return bytes
            df[col] = df[col].apply(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else str(x)
            )
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    # Drop raw label, keep only numeric + label
    df = df.drop(columns=["label_raw"], errors="ignore")

    # Add source tag and split
    df["source_dataset"] = "KDD-Cup-99"
    df["split"] = "combined"

    return df


def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    use_original_split: bool = True,
):
    """
    Split and scale the data for modelling.

    If the DataFrame has a 'split' column with 'train'/'test' values and
    use_original_split is True, uses the original NSL-KDD train/test split.
    Otherwise, performs a random stratified split.

    Returns
    -------
    dict with keys: X_train, X_val, X_test, y_train, y_val, y_test,
                    scaler, feature_names
    """
    exclude_cols = {"label", "source_dataset", "split"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].values.astype(np.float64)
    y = df["label"].values.astype(int)

    has_split = "split" in df.columns
    has_train_test = (
        has_split
        and set(df["split"].unique()) >= {"train", "test"}
    )

    if use_original_split and has_train_test:
        # Use NSL-KDD original train/test split
        train_mask = df["split"].values == "train"
        test_mask = df["split"].values == "test"

        X_train_full = X[train_mask]
        y_train_full = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]

        # Carve out validation set from training data
        val_frac = val_size / (1.0 - test_size)  # approximate
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=val_frac,
            random_state=random_state,
            stratify=y_train_full,
        )
    else:
        # Random stratified split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_size + val_size,
            random_state=random_state, stratify=y,
        )
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
    df = load_nsl_kdd()
    print(f"Shape: {df.shape}")
    print(f"\nClass distribution:\n{df['label'].value_counts().sort_index()}")
    for label, name in enumerate(CLASS_NAMES):
        count = (df["label"] == label).sum()
        print(f"  {name:20s}: {count:6,} ({count/len(df):6.1%})")
    print(f"\nDataset sources:\n{df['source_dataset'].value_counts()}")
    print(f"\nFeature summary:\n{df.describe().T[['mean', 'std', 'min', 'max']]}")
