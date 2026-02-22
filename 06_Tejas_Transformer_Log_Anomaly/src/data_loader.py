"""
data_loader.py - Synthetic Log Data Generator and Feature Extractor
====================================================================
Generates synthetic cloud security log data modelled on HDFS, BGL, and
Thunderbird datasets.  Each log line is produced from a template bank and
labelled as normal (0) or anomaly (1).  Feature extraction supports two
modes:
    1. TF-IDF vectorisation (lightweight, CPU-friendly)
    2. Simulated transformer embeddings (structured random vectors that
       mimic the geometry of DistilBERT / MiniLM sentence embeddings)

Author : Tejas Vijay Mariyappagoudar (x24213829)
"""

import re
import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------
# Template banks
# ---------------------------------------------------------------------------

HDFS_NORMAL_TEMPLATES = [
    "PacketResponder {blk_id} for block blk_{block} terminating",
    "Receiving block blk_{block} src: /{src_ip} dest: /{dst_ip}",
    "BLOCK* NameSystem.addStoredBlock: blockMap updated: {ip} is added to blk_{block} size {size}",
    "BLOCK* NameSystem.allocateBlock: /user/root/rand/_temporary/_task_{task}_r_{reduce}/part-{part}. blk_{block}",
    "Verification succeeded for blk_{block}",
    "Received block blk_{block} of size {size} from /{src_ip}",
    "writeBlock blk_{block} received exception java.io.IOException: Connection reset by peer",
    "PacketResponder {blk_id} for block blk_{block} Interrupted",
    "Deleting block blk_{block} file /mnt/hadoop/dfs/data/current/blk_{block}",
    "Starting thread to transfer block blk_{block} to /{dst_ip}",
]

HDFS_ANOMALY_TEMPLATES = [
    "BLOCK* ask {ip} to delete blk_{block}",
    "BLOCK* NameSystem.addStoredBlock: redundant addStoredBlock request received for blk_{block} on /{ip} size {size}",
    "BLOCK* NameSystem.delete: blk_{block} is added to invalidSet of /{ip}",
    "PendingReplicationMonitor timed out block blk_{block}",
    "BLOCK* Removing block blk_{block} from neededReplications as it has enough replicas",
    "BLOCK* ask {ip} to replicate blk_{block} to datanode(s) /{dst_ip}",
]

BGL_NORMAL_TEMPLATES = [
    "RAS KERNEL INFO generating core.{core_id}",
    "RAS KERNEL INFO instruction cache parity error corrected",
    "RAS KERNEL INFO CE sym {sym}, at {addr}, mask {mask}",
    "RAS APP FATAL ciod: Login {user} on node {node}: No such file or directory",
    "RAS KERNEL INFO L3 cache correctable error detected and corrected",
    "RAS KERNEL INFO TLB error corrected on reload",
    "RAS KERNEL INFO DDR controller chip-kill applied successfully",
    "RAS KERNEL INFO floating point alignment exception, corrected",
    "RAS KERNEL INFO network link recovery complete for torus {dim}+",
    "RAS KERNEL INFO total interrupts = {int_count}",
]

BGL_ANOMALY_TEMPLATES = [
    "RAS KERNEL FATAL machine check interrupt: uncorrectable error",
    "RAS KERNEL FATAL double-hummer alignment exception, data not available",
    "RAS KERNEL FATAL data storage interrupt: store failed, kernel memory failure",
    "RAS KERNEL FATAL network link FATAL error on torus {dim}+",
    "RAS KERNEL FATAL ciod: failed to read message prefix on control stream",
    "RAS KERNEL FATAL L3 uncorrectable DCR read timeout",
]

THUNDERBIRD_NORMAL_TEMPLATES = [
    "sshd(pam_unix)[{pid}]: session opened for user {user} by (uid=0)",
    "sshd(pam_unix)[{pid}]: session closed for user {user}",
    "kernel: end_request: I/O error, dev fd0, sector 0",
    "automount[{pid}]: lookup_mount: lookup(ldap): no more results for {mount}",
    "crond(pam_unix)[{pid}]: session opened for user root by (uid=0)",
    "syslogd {version}: restart",
    "kernel: NET: Registered protocol family {family}",
    "ntpd[{pid}]: synchronized to {ntp_server}, stratum {stratum}",
    "named[{pid}]: client {ip}#{port}: query: {domain} IN A",
    "sendmail[{pid}]: {msg_id}: from=<{user}@{domain}>, size={size}, class=0",
]

THUNDERBIRD_ANOMALY_TEMPLATES = [
    "kernel: Oops: {oops_code}, Not tainted",
    "kernel: general protection fault: {fault_code} [#1]",
    "kernel: BUG: unable to handle kernel paging request at virtual address {vaddr}",
    "kernel: EDAC MC{mc}: {num} CE error on CPU#{cpu}",
    "kernel: Out of Memory: Killed process {pid} ({proc})",
    "kernel: NMI Watchdog detected LOCKUP on CPU {cpu}",
]


def _random_ip():
    return f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"


def _fill_template(template: str) -> str:
    """Replace placeholders in a template with random realistic values."""
    replacements = {
        "{blk_id}": str(random.randint(0, 5)),
        "{block}": str(random.randint(1000000000, 9999999999)),
        "{src_ip}": _random_ip(),
        "{dst_ip}": _random_ip(),
        "{ip}": _random_ip(),
        "{size}": str(random.choice([67108864, 33554432, 134217728, 16384, 65536])),
        "{task}": f"{random.randint(200700,200800)}_{random.randint(1,50)}_{random.randint(0,9)}",
        "{reduce}": f"{random.randint(0,9)}",
        "{part}": f"{random.randint(0,999):05d}",
        "{core_id}": str(random.randint(1000, 9999)),
        "{sym}": str(random.randint(0, 15)),
        "{addr}": f"0x{random.randint(0, 0xFFFFFFFF):08x}",
        "{mask}": f"0x{random.randint(0, 0xFF):02x}",
        "{user}": random.choice(["root", "admin", "hadoop", "spark", "tejas", "deploy"]),
        "{node}": f"R{random.randint(0,7)}{random.randint(0,7)}-M{random.randint(0,1)}-N{random.randint(0,9)}",
        "{dim}": random.choice(["X", "Y", "Z"]),
        "{int_count}": str(random.randint(100000, 999999)),
        "{pid}": str(random.randint(1000, 65535)),
        "{mount}": f"/home/{random.choice(['user1','user2','admin','data'])}",
        "{version}": "1.4.1",
        "{family}": str(random.choice([2, 10, 17, 29])),
        "{ntp_server}": f"{_random_ip()}",
        "{stratum}": str(random.randint(1, 5)),
        "{port}": str(random.randint(1024, 65535)),
        "{domain}": random.choice(["example.com", "cluster.local", "node.internal"]),
        "{msg_id}": f"k{random.randint(1,12):02d}{random.randint(1,28):02d}{random.randint(0,23):02d}{random.randint(0,59):02d}{random.randint(0,59):02d}",
        "{oops_code}": f"0000 [#{random.randint(1,3)}]",
        "{fault_code}": f"0000 [{random.randint(1,3)}]",
        "{vaddr}": f"0x{random.randint(0, 0xFFFFFFFF):08x}",
        "{mc}": str(random.randint(0, 3)),
        "{num}": str(random.randint(1, 128)),
        "{cpu}": str(random.randint(0, 31)),
        "{proc}": random.choice(["java", "python", "httpd", "mysqld", "sshd"]),
    }
    result = template
    for placeholder, value in replacements.items():
        result = result.replace(placeholder, value)
    return result


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    n_samples: int = 5000,
    anomaly_ratio: float = 0.15,
    dataset_sources: list = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic log dataset.

    Parameters
    ----------
    n_samples : int
        Total number of log lines to generate.
    anomaly_ratio : float
        Fraction of logs that are anomalies (0.0 - 1.0).
    dataset_sources : list of str or None
        Subset of {"hdfs", "bgl", "thunderbird"}.  None = all three.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: log_message, label, source
    """
    random.seed(seed)
    np.random.seed(seed)

    if dataset_sources is None:
        dataset_sources = ["hdfs", "bgl", "thunderbird"]

    normal_banks = {
        "hdfs": HDFS_NORMAL_TEMPLATES,
        "bgl": BGL_NORMAL_TEMPLATES,
        "thunderbird": THUNDERBIRD_NORMAL_TEMPLATES,
    }
    anomaly_banks = {
        "hdfs": HDFS_ANOMALY_TEMPLATES,
        "bgl": BGL_ANOMALY_TEMPLATES,
        "thunderbird": THUNDERBIRD_ANOMALY_TEMPLATES,
    }

    n_anomaly = int(n_samples * anomaly_ratio)
    n_normal = n_samples - n_anomaly

    logs = []
    labels = []
    sources = []

    for _ in range(n_normal):
        src = random.choice(dataset_sources)
        template = random.choice(normal_banks[src])
        logs.append(_fill_template(template))
        labels.append(0)
        sources.append(src)

    for _ in range(n_anomaly):
        src = random.choice(dataset_sources)
        template = random.choice(anomaly_banks[src])
        logs.append(_fill_template(template))
        labels.append(1)
        sources.append(src)

    df = pd.DataFrame({"log_message": logs, "label": labels, "source": sources})
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
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
    * Anomaly logs receive a subtle directional shift, making them
      statistically separable but not trivially so.

    This produces structured vectors with realistic cosine-similarity
    properties without requiring a GPU or a real transformer model.
    """
    rng = np.random.RandomState(seed)

    # Stable token -> vector mapping via hashing
    vocab_size = 8192
    token_table = rng.randn(vocab_size, dim).astype(np.float32) * 0.15

    # Source-specific bias vectors
    source_bias = {
        "hdfs":        rng.randn(dim).astype(np.float32) * 0.08,
        "bgl":         rng.randn(dim).astype(np.float32) * 0.08,
        "thunderbird": rng.randn(dim).astype(np.float32) * 0.08,
    }
    anomaly_direction = rng.randn(dim).astype(np.float32)
    anomaly_direction /= (np.linalg.norm(anomaly_direction) + 1e-9)

    # Simple heuristic: messages containing FATAL / Oops / BUG / killed /
    # uncorrectable etc. are treated as anomaly-like for the bias shift
    anomaly_keywords = {
        "fatal", "oops", "bug", "killed", "lockup", "uncorrectable",
        "timeout", "fault", "memory", "nmi", "paging",
        "invalidset", "replicate", "redundant", "timed",
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
        if "blk_" in text_lower or "block" in text_lower:
            vec += source_bias["hdfs"]
        elif "ras " in text_lower or "kernel info" in text_lower or "kernel fatal" in text_lower:
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
    df = generate_dataset(n_samples=500, anomaly_ratio=0.15, seed=42)
    print(f"Generated {len(df)} logs  |  anomaly ratio = {df['label'].mean():.2%}")
    print(f"Sources: {df['source'].value_counts().to_dict()}")
    print("\n--- Sample normal log ---")
    print(df[df["label"] == 0].iloc[0]["log_message"])
    print("\n--- Sample anomaly log ---")
    print(df[df["label"] == 1].iloc[0]["log_message"])

    X_tfidf, _ = prepare_features(df, method="tfidf")
    print(f"\nTF-IDF features shape : {X_tfidf.shape}")

    X_trans, _ = prepare_features(df, method="transformer")
    print(f"Transformer features  : {X_trans.shape}")
