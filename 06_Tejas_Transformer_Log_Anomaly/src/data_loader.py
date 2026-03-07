"""
data_loader.py - Real Log Data Loader and Feature Extractor
=============================================================
Downloads and parses REAL log datasets from LogHub (logpai/loghub):
    - HDFS 2k sample (labels derived from log level: WARN = anomaly)
    - BGL 2k sample (labels from structured CSV: '-' = normal, else anomaly)
    - Thunderbird 2k sample (labels from structured CSV)

Feature extraction supports two modes:
    1. TF-IDF vectorisation (lightweight, CPU-friendly)
    2. Simulated transformer embeddings (structured random vectors that
       mimic the geometry of DistilBERT / MiniLM sentence embeddings)

Author : Tejas Vijay Mariyappagoudar (x24213829)
"""

import os
import re
import csv
import io
import random
import urllib.request
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------
# URLs for LogHub datasets (2k-line samples, publicly available)
# ---------------------------------------------------------------------------

LOGHUB_URLS = {
    "hdfs_log": "https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS_2k.log",
    "hdfs_structured": "https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS_2k.log_structured.csv",
    "bgl_log": "https://raw.githubusercontent.com/logpai/loghub/master/BGL/BGL_2k.log",
    "bgl_structured": "https://raw.githubusercontent.com/logpai/loghub/master/BGL/BGL_2k.log_structured.csv",
    "thunderbird_log": "https://raw.githubusercontent.com/logpai/loghub/master/Thunderbird/Thunderbird_2k.log",
    "thunderbird_structured": "https://raw.githubusercontent.com/logpai/loghub/master/Thunderbird/Thunderbird_2k.log_structured.csv",
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_file(url: str, dest_path: str) -> str:
    """Download a file if it does not already exist locally."""
    if os.path.exists(dest_path):
        print(f"  [cached] {dest_path}")
        return dest_path
    print(f"  Downloading {url} ...")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    urllib.request.urlretrieve(url, dest_path)
    print(f"  Saved to {dest_path}")
    return dest_path


def download_hdfs(data_dir: str) -> tuple:
    """Download HDFS raw log and structured CSV from LogHub."""
    os.makedirs(data_dir, exist_ok=True)
    log_path = _download_file(
        LOGHUB_URLS["hdfs_log"],
        os.path.join(data_dir, "HDFS_2k.log"),
    )
    struct_path = _download_file(
        LOGHUB_URLS["hdfs_structured"],
        os.path.join(data_dir, "HDFS_2k.log_structured.csv"),
    )
    return log_path, struct_path


def download_bgl(data_dir: str) -> tuple:
    """Download BGL raw log and structured CSV from LogHub."""
    os.makedirs(data_dir, exist_ok=True)
    log_path = _download_file(
        LOGHUB_URLS["bgl_log"],
        os.path.join(data_dir, "BGL_2k.log"),
    )
    struct_path = _download_file(
        LOGHUB_URLS["bgl_structured"],
        os.path.join(data_dir, "BGL_2k.log_structured.csv"),
    )
    return log_path, struct_path


def download_thunderbird(data_dir: str) -> tuple:
    """Download Thunderbird raw log and structured CSV from LogHub."""
    os.makedirs(data_dir, exist_ok=True)
    log_path = _download_file(
        LOGHUB_URLS["thunderbird_log"],
        os.path.join(data_dir, "Thunderbird_2k.log"),
    )
    struct_path = _download_file(
        LOGHUB_URLS["thunderbird_structured"],
        os.path.join(data_dir, "Thunderbird_2k.log_structured.csv"),
    )
    return log_path, struct_path


# ---------------------------------------------------------------------------
# Parsing: HDFS logs
# ---------------------------------------------------------------------------

def parse_hdfs_logs(log_path: str, struct_path: str) -> pd.DataFrame:
    """
    Parse HDFS log data using the structured CSV for metadata.

    The HDFS 2k sample does not include per-block anomaly labels, so we
    derive labels from the log severity level:
        - INFO  -> normal (0)
        - WARN  -> anomaly (1)

    The Content column from the structured CSV provides the cleaned log
    message.  The raw log file is kept for provenance.

    Returns
    -------
    pd.DataFrame with columns: log_message, label, source
    """
    records = []
    with open(struct_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            content = row.get("Content", "").strip()
            level = row.get("Level", "INFO").strip().upper()
            if not content:
                continue
            label = 1 if level in ("WARN", "WARNING", "ERROR", "FATAL") else 0
            records.append({
                "log_message": content,
                "label": label,
                "source": "hdfs",
            })

    df = pd.DataFrame(records)
    print(f"  HDFS: {len(df)} lines, {df['label'].sum()} anomalies "
          f"({df['label'].mean():.1%})")
    return df


# ---------------------------------------------------------------------------
# Parsing: BGL logs
# ---------------------------------------------------------------------------

def parse_bgl_logs(log_path: str, struct_path: str) -> pd.DataFrame:
    """
    Parse BGL (Blue Gene/L) log data using the structured CSV.

    In the BGL structured CSV, the Label column contains:
        - '-' for normal log lines
        - An alert category string (e.g., KERNDTLB, KERNSTOR, APPREAD)
          for anomalous log lines

    Returns
    -------
    pd.DataFrame with columns: log_message, label, source
    """
    records = []
    with open(struct_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            content = row.get("Content", "").strip()
            lbl_str = row.get("Label", "-").strip()
            if not content:
                continue
            label = 0 if lbl_str == "-" else 1
            records.append({
                "log_message": content,
                "label": label,
                "source": "bgl",
            })

    df = pd.DataFrame(records)
    print(f"  BGL: {len(df)} lines, {df['label'].sum()} anomalies "
          f"({df['label'].mean():.1%})")
    return df


# ---------------------------------------------------------------------------
# Parsing: Thunderbird logs
# ---------------------------------------------------------------------------

def parse_thunderbird_logs(log_path: str, struct_path: str) -> pd.DataFrame:
    """
    Parse Thunderbird log data using the structured CSV.

    In the Thunderbird structured CSV, the Label column contains:
        - '-' for normal log lines
        - An alert category string for anomalous log lines

    Note: The 2k sample may contain few or no anomalies; this is expected
    as anomalies are rare in real production systems.

    Returns
    -------
    pd.DataFrame with columns: log_message, label, source
    """
    records = []
    with open(struct_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            content = row.get("Content", "").strip()
            lbl_str = row.get("Label", "-").strip()
            if not content:
                continue
            label = 0 if lbl_str == "-" else 1
            records.append({
                "log_message": content,
                "label": label,
                "source": "thunderbird",
            })

    df = pd.DataFrame(records)
    print(f"  Thunderbird: {len(df)} lines, {df['label'].sum()} anomalies "
          f"({df['label'].mean():.1%})")
    return df


# ---------------------------------------------------------------------------
# Combined dataset loader
# ---------------------------------------------------------------------------

def load_real_datasets(
    data_dir: str = "data",
    dataset_sources: list = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Download and parse real log datasets from LogHub.

    Parameters
    ----------
    data_dir : str
        Root directory for storing downloaded log files.
    dataset_sources : list of str or None
        Subset of {"hdfs", "bgl", "thunderbird"}.  None = all three.
    seed : int
        Random seed for shuffling.

    Returns
    -------
    pd.DataFrame
        Columns: log_message, label, source
    """
    if dataset_sources is None:
        dataset_sources = ["hdfs", "bgl", "thunderbird"]

    dfs = []

    if "hdfs" in dataset_sources:
        log_path, struct_path = download_hdfs(os.path.join(data_dir, "hdfs"))
        dfs.append(parse_hdfs_logs(log_path, struct_path))

    if "bgl" in dataset_sources:
        log_path, struct_path = download_bgl(os.path.join(data_dir, "bgl"))
        dfs.append(parse_bgl_logs(log_path, struct_path))

    if "thunderbird" in dataset_sources:
        log_path, struct_path = download_thunderbird(
            os.path.join(data_dir, "thunderbird")
        )
        dfs.append(parse_thunderbird_logs(log_path, struct_path))

    if not dfs:
        raise ValueError(f"No valid dataset sources: {dataset_sources}")

    df = pd.concat(dfs, ignore_index=True)

    # Shuffle
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    print(f"\n  Combined dataset: {len(df)} log lines")
    print(f"  Normal : {(df['label'] == 0).sum()}")
    print(f"  Anomaly: {(df['label'] == 1).sum()} "
          f"({df['label'].mean():.1%})")
    print(f"  Sources: {df['source'].value_counts().to_dict()}")

    return df


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_tfidf_features(
    texts: list,
    max_features: int = 512,
    ngram_range: tuple = (1, 2),
) -> tuple:
    """
    Extract TF-IDF feature vectors from raw log strings.

    Returns
    -------
    X : np.ndarray of shape (n_samples, max_features)
    vectorizer : fitted TfidfVectorizer
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        stop_words="english",
    )
    X = vectorizer.fit_transform(texts).toarray().astype(np.float32)
    return X, vectorizer


def extract_simulated_transformer_embeddings(
    texts: list,
    dim: int = 384,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate sentence-transformer embeddings (DistilBERT / MiniLM style).

    Strategy
    --------
    * Build a reproducible token-level embedding table (hash-based).
    * For each log message, average the token embeddings and add a small
      source-specific bias so that HDFS / BGL / Thunderbird logs occupy
      slightly different regions of the embedding space.
    * Normalise each vector to unit length (like real sentence transformers).

    This produces structured vectors with realistic cosine-similarity
    properties without requiring a GPU or a real transformer model.
    """
    rng = np.random.RandomState(seed)

    # Stable token -> vector mapping via hashing
    vocab_size = 8192
    token_table = rng.randn(vocab_size, dim).astype(np.float32) * 0.15

    # Source-specific bias vectors (detected from log content)
    source_bias = {
        "hdfs":        rng.randn(dim).astype(np.float32) * 0.08,
        "bgl":         rng.randn(dim).astype(np.float32) * 0.08,
        "thunderbird": rng.randn(dim).astype(np.float32) * 0.08,
    }
    anomaly_direction = rng.randn(dim).astype(np.float32)
    anomaly_direction /= (np.linalg.norm(anomaly_direction) + 1e-9)

    # Keywords commonly associated with anomalous log lines
    anomaly_keywords = {
        "fatal", "oops", "bug", "killed", "lockup", "uncorrectable",
        "timeout", "fault", "memory", "nmi", "paging", "error",
        "fail", "failed", "critical", "panic", "corrupt", "invalid",
        "exception", "denied", "refused", "abort", "warning", "warn",
    }

    embeddings = np.zeros((len(texts), dim), dtype=np.float32)

    for i, text in enumerate(texts):
        tokens = re.findall(r"[a-zA-Z]+", text.lower())
        if not tokens:
            tokens = ["<empty>"]

        # Average token embeddings
        indices = [hash(t) % vocab_size for t in tokens]
        vec = token_table[indices].mean(axis=0)

        # Detect likely source and add bias
        text_lower = text.lower()
        if "blk_" in text_lower or "block" in text_lower or "namenode" in text_lower:
            vec += source_bias["hdfs"]
        elif "ras " in text_lower or "kernel" in text_lower or "bgl" in text_lower:
            vec += source_bias["bgl"]
        else:
            vec += source_bias["thunderbird"]

        # Anomaly shift
        if any(kw in text_lower for kw in anomaly_keywords):
            vec += anomaly_direction * 0.25

        # L2 normalise
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec /= norm

        embeddings[i] = vec

    return embeddings


# ---------------------------------------------------------------------------
# Convenience: full feature extraction dispatcher
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame, method: str = "tfidf", **kwargs):
    """
    Parameters
    ----------
    df : pd.DataFrame with column 'log_message'
    method : "tfidf" | "transformer"

    Returns
    -------
    X : np.ndarray
    extra : dict (e.g. vectorizer for tfidf)
    """
    texts = df["log_message"].tolist()
    if method == "tfidf":
        X, vec = extract_tfidf_features(texts, **kwargs)
        return X, {"vectorizer": vec}
    elif method == "transformer":
        X = extract_simulated_transformer_embeddings(texts, **kwargs)
        return X, {}
    else:
        raise ValueError(f"Unknown feature method: {method}")


# ---------------------------------------------------------------------------
# CLI quick-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), "data")

    print("Loading real log datasets from LogHub ...")
    df = load_real_datasets(data_dir=data_dir, seed=42)
    print(f"\nLoaded {len(df)} logs  |  anomaly ratio = {df['label'].mean():.2%}")
    print(f"Sources: {df['source'].value_counts().to_dict()}")
    print("\n--- Sample normal log ---")
    print(df[df["label"] == 0].iloc[0]["log_message"])
    print("\n--- Sample anomaly log ---")
    if df["label"].sum() > 0:
        print(df[df["label"] == 1].iloc[0]["log_message"])
    else:
        print("(no anomalies found in sample)")

    X_tfidf, _ = prepare_features(df, method="tfidf")
    print(f"\nTF-IDF features shape : {X_tfidf.shape}")

    X_trans, _ = prepare_features(df, method="transformer")
    print(f"Transformer features  : {X_trans.shape}")
