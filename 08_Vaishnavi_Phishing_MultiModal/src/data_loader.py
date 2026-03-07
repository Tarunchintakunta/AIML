"""
data_loader.py - UCI Phishing Dataset Loader (Multi-Modal)
==========================================================
Project 8: Explainable Multi-Modal Phishing Detection
Student: Vaishnavi Purohit (24260339)

Downloads and loads the REAL UCI Phishing Websites dataset (11,055 samples,
30 features) and splits features into three modality groups for the
multi-modal architecture:
  (a) URL-based features  - 10 features capturing URL characteristics
  (b) Content-based features - 10 features capturing page content signals
  (c) External features - 10 features capturing domain/traffic metadata

Dataset source:
  Mohammad, R., McCluskey, T.L. & Thabtah, F. (2014).
  UCI Machine Learning Repository: Phishing Websites Data Set.
  https://archive.ics.uci.edu/ml/datasets/Phishing+Websites

References:
  Al-Subaiey et al. 2024; Alhuzali et al. 2025; Patra et al. 2025
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# UCI Phishing dataset attribute names (30 features from the ARFF file)
# Ordered as they appear in the original ARFF:
#   1. having_IP_Address
#   2. URL_Length
#   3. Shortining_Service
#   4. having_At_Symbol
#   5. double_slash_redirecting
#   6. Prefix_Suffix
#   7. having_Sub_Domain
#   8. SSLfinal_State
#   9. Domain_registeration_length
#  10. Favicon
#  11. port
#  12. HTTPS_token
#  13. Request_URL
#  14. URL_of_Anchor
#  15. Links_in_tags
#  16. SFH
#  17. Submitting_to_email
#  18. Abnormal_URL
#  19. Redirect
#  20. on_mouseover
#  21. RightClick
#  22. popUpWidnow
#  23. Iframe
#  24. age_of_domain
#  25. DNSRecord
#  26. web_traffic
#  27. Page_Rank
#  28. Google_Index
#  29. Links_pointing_to_page
#  30. Statistical_report
# ---------------------------------------------------------------------------

# Modality 1: URL-based features (features about the URL structure)
URL_FEATURE_NAMES = [
    "having_IP_Address",      # 1
    "URL_Length",              # 2
    "Shortining_Service",     # 3
    "having_At_Symbol",       # 4
    "double_slash_redirecting",  # 5
    "Prefix_Suffix",          # 6
    "having_Sub_Domain",      # 7
    "SSLfinal_State",         # 8
    "Domain_registeration_length",  # 9
    "Favicon",                # 10
]

# Modality 2: Content-based features (page content and behavior)
CONTENT_FEATURE_NAMES = [
    "port",                   # 11
    "HTTPS_token",            # 12
    "Request_URL",            # 13
    "URL_of_Anchor",          # 14
    "Links_in_tags",          # 15
    "SFH",                    # 16
    "Submitting_to_email",    # 17
    "Abnormal_URL",           # 18
    "Redirect",               # 19
    "on_mouseover",           # 20
]

# Modality 3: External / third-party features (domain reputation, traffic)
EXTERNAL_FEATURE_NAMES = [
    "RightClick",             # 21
    "popUpWidnow",            # 22
    "Iframe",                 # 23
    "age_of_domain",          # 24
    "DNSRecord",              # 25
    "web_traffic",            # 26
    "Page_Rank",              # 27
    "Google_Index",           # 28
    "Links_pointing_to_page", # 29
    "Statistical_report",     # 30
]

ALL_FEATURE_NAMES = URL_FEATURE_NAMES + CONTENT_FEATURE_NAMES + EXTERNAL_FEATURE_NAMES

# Modality column ranges (used by explainers.py)
MODALITY_RANGES = {
    "url": (0, 10),
    "content": (10, 20),
    "external": (20, 30),
}


def download_uci_phishing(data_dir: str = None) -> str:
    """Download the UCI Phishing Websites ARFF file if not already cached.

    Parameters
    ----------
    data_dir : str or None
        Directory to store the downloaded file. Defaults to
        <project_root>/data/phishing.

    Returns
    -------
    str : path to the downloaded ARFF file
    """
    import urllib.request

    if data_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(project_root, "data", "phishing")

    os.makedirs(data_dir, exist_ok=True)
    arff_path = os.path.join(data_dir, "phishing.arff")

    if not os.path.exists(arff_path):
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
               "00327/Training%20Dataset.arff")
        print(f"[DataLoader] Downloading UCI Phishing dataset from:\n  {url}")
        try:
            urllib.request.urlretrieve(url, arff_path)
            print(f"[DataLoader] Saved to {arff_path}")
        except Exception as e:
            print(f"[DataLoader] Download failed: {e}")
            print("[DataLoader] Attempting fallback: parsing as CSV...")
            raise
    else:
        print(f"[DataLoader] Using cached dataset: {arff_path}")

    return arff_path


def _parse_arff(arff_path: str) -> pd.DataFrame:
    """Parse the ARFF file into a pandas DataFrame.

    Uses scipy.io.arff if available, otherwise falls back to manual parsing.
    """
    try:
        from scipy.io import arff
        data, meta = arff.loadarff(arff_path)
        df = pd.DataFrame(data)
        # scipy.io.arff may return bytes for nominal attributes; decode them
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].apply(
                    lambda x: x.decode('utf-8') if isinstance(x, bytes) else x
                )
        df = df.apply(pd.to_numeric, errors='coerce')
        return df
    except ImportError:
        pass

    # Fallback: manual ARFF parsing
    print("[DataLoader] scipy not available, parsing ARFF manually...")
    columns = []
    data_lines = []
    in_data = False

    with open(arff_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            if line.upper().startswith('@ATTRIBUTE'):
                parts = line.split()
                columns.append(parts[1])
            elif line.upper().startswith('@DATA'):
                in_data = True
                continue
            elif in_data:
                values = line.split(',')
                data_lines.append([float(v) for v in values])

    df = pd.DataFrame(data_lines, columns=columns)
    return df


def load_uci_phishing(data_dir: str = None) -> dict:
    """Download (if needed) and load the UCI Phishing dataset.

    The dataset has 30 features and 1 label column ('Result').
    Labels in the original dataset: -1 (phishing), 1 (legitimate).
    We remap to: 1 (phishing), 0 (legitimate).

    The 30 features are split into 3 modality groups of 10 each:
      - URL features (first 10)
      - Content features (next 10)
      - External features (last 10)

    Returns
    -------
    dict with keys:
        'X_url', 'X_content', 'X_external', 'X_combined', 'y',
        'feature_names_url', 'feature_names_content',
        'feature_names_external', 'feature_names_all', 'df'
    """
    arff_path = download_uci_phishing(data_dir)
    df = _parse_arff(arff_path)

    # The last column is the label ('Result')
    label_col = df.columns[-1]
    feature_cols = df.columns[:-1].tolist()

    # Rename feature columns to our canonical names
    rename_map = {old: new for old, new in zip(feature_cols, ALL_FEATURE_NAMES)}
    df = df.rename(columns=rename_map)
    df = df.rename(columns={label_col: "label"})

    # Remap labels: original -1 -> 1 (phishing), original 1 -> 0 (legitimate)
    # Some versions use {-1, 0, 1} -- handle both
    df["label"] = df["label"].apply(lambda x: 1 if x == -1 else 0)

    # Drop any rows with NaN
    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    if len(df) < n_before:
        print(f"[DataLoader] Dropped {n_before - len(df)} rows with NaN values")

    # Extract feature arrays
    X_url = df[URL_FEATURE_NAMES].values.astype(np.float64)
    X_content = df[CONTENT_FEATURE_NAMES].values.astype(np.float64)
    X_external = df[EXTERNAL_FEATURE_NAMES].values.astype(np.float64)
    X_combined = df[ALL_FEATURE_NAMES].values.astype(np.float64)
    y = df["label"].values.astype(int)

    n_phishing = (y == 1).sum()
    n_legit = (y == 0).sum()

    print(f"[DataLoader] Loaded UCI Phishing dataset: {len(df)} samples "
          f"({n_phishing} phishing, {n_legit} legitimate)")
    print(f"  URL features      : {X_url.shape[1]}")
    print(f"  Content features  : {X_content.shape[1]}")
    print(f"  External features : {X_external.shape[1]}")
    print(f"  Combined          : {X_combined.shape[1]}")

    return {
        "X_url": X_url,
        "X_content": X_content,
        "X_external": X_external,
        "X_combined": X_combined,
        "y": y,
        "feature_names_url": URL_FEATURE_NAMES,
        "feature_names_content": CONTENT_FEATURE_NAMES,
        "feature_names_external": EXTERNAL_FEATURE_NAMES,
        "feature_names_all": ALL_FEATURE_NAMES,
        "df": df,
    }


def prepare_splits(data: dict, test_size: float = 0.2,
                   random_state: int = 42, scale: bool = True) -> dict:
    """Split data into train/test and optionally scale.

    Returns
    -------
    dict with train/test splits for each modality and combined.
    """
    y = data["y"]
    splits = {}

    modalities = [
        ("url", data["X_url"], data["feature_names_url"]),
        ("content", data["X_content"], data["feature_names_content"]),
        ("external", data["X_external"], data["feature_names_external"]),
        ("combined", data["X_combined"], data["feature_names_all"]),
    ]

    for key, X, names in modalities:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y
        )

        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        else:
            scaler = None

        splits[key] = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "feature_names": names,
            "scaler": scaler,
        }

    print(f"[DataLoader] Train/test split: "
          f"{splits['combined']['X_train'].shape[0]} / "
          f"{splits['combined']['X_test'].shape[0]}")

    return splits


if __name__ == "__main__":
    data = load_uci_phishing()
    splits = prepare_splits(data)
    print("\nModality shapes (train):")
    for mod in ["url", "content", "external", "combined"]:
        print(f"  {mod:10s}: {splits[mod]['X_train'].shape}")
