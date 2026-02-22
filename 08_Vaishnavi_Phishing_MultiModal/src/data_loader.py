"""
data_loader.py - Synthetic Multi-Modal Phishing Dataset Generator
=================================================================
Project 8: Explainable Multi-Modal Phishing Detection
Student: Vaishnavi Purohit (24260339)

Generates synthetic phishing data with three modalities:
  (a) Text features  - 20 TF-IDF-like features simulating email content
  (b) URL features   - 10 features capturing URL metadata
  (c) Temporal features - 5 features capturing time-based signals

References:
  Al-Subaiey et al. 2024; Alhuzali et al. 2025; Patra et al. 2025
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Feature name constants
# ---------------------------------------------------------------------------
TEXT_FEATURE_NAMES = [
    "tfidf_urgent", "tfidf_click", "tfidf_verify", "tfidf_account",
    "tfidf_password", "tfidf_suspend", "tfidf_confirm", "tfidf_update",
    "tfidf_login", "tfidf_secure", "tfidf_bank", "tfidf_paypal",
    "tfidf_offer", "tfidf_free", "tfidf_winner", "tfidf_dear",
    "tfidf_customer", "tfidf_immediately", "tfidf_expire", "tfidf_limited"
]

URL_FEATURE_NAMES = [
    "url_length", "num_dots", "num_hyphens", "num_slashes",
    "has_https", "has_ip_address", "suspicious_tld",
    "num_subdomains", "path_length", "has_at_symbol"
]

TEMPORAL_FEATURE_NAMES = [
    "hour_sent", "day_of_week", "domain_age_days",
    "time_since_last_email_hrs", "email_frequency_per_day"
]

ALL_FEATURE_NAMES = TEXT_FEATURE_NAMES + URL_FEATURE_NAMES + TEMPORAL_FEATURE_NAMES


def _generate_text_features(n_samples: int, labels: np.ndarray,
                            rng: np.random.Generator) -> np.ndarray:
    """Generate 20 TF-IDF-like text features.

    Phishing emails tend to have higher weights on urgency / action words
    (columns 0-9) while legitimate emails have more uniform distributions.
    """
    X = np.zeros((n_samples, 20), dtype=np.float64)
    phishing_mask = labels == 1

    # Phishing: higher urgency words (first 10 features)
    n_phish = phishing_mask.sum()
    n_legit = n_samples - n_phish

    # Phishing text features
    X[phishing_mask, :10] = rng.exponential(0.25, size=(n_phish, 10))
    X[phishing_mask, 10:] = rng.exponential(0.08, size=(n_phish, 10))

    # Legitimate text features
    X[~phishing_mask, :10] = rng.exponential(0.06, size=(n_legit, 10))
    X[~phishing_mask, 10:] = rng.exponential(0.12, size=(n_legit, 10))

    # Add noise
    X += rng.normal(0, 0.02, size=X.shape)
    X = np.clip(X, 0, None)

    return X


def _generate_url_features(n_samples: int, labels: np.ndarray,
                           rng: np.random.Generator) -> np.ndarray:
    """Generate 10 URL metadata features.

    Phishing URLs tend to be longer, have more dots/hyphens, use IP
    addresses, suspicious TLDs, and rarely use HTTPS.
    """
    X = np.zeros((n_samples, 10), dtype=np.float64)
    phishing_mask = labels == 1
    n_phish = phishing_mask.sum()
    n_legit = n_samples - n_phish

    # url_length
    X[phishing_mask, 0] = rng.normal(85, 20, n_phish)
    X[~phishing_mask, 0] = rng.normal(40, 12, n_legit)

    # num_dots
    X[phishing_mask, 1] = rng.poisson(4, n_phish).astype(float)
    X[~phishing_mask, 1] = rng.poisson(2, n_legit).astype(float)

    # num_hyphens
    X[phishing_mask, 2] = rng.poisson(3, n_phish).astype(float)
    X[~phishing_mask, 2] = rng.poisson(0.5, n_legit).astype(float)

    # num_slashes
    X[phishing_mask, 3] = rng.poisson(5, n_phish).astype(float)
    X[~phishing_mask, 3] = rng.poisson(3, n_legit).astype(float)

    # has_https (binary)
    X[phishing_mask, 4] = rng.binomial(1, 0.25, n_phish).astype(float)
    X[~phishing_mask, 4] = rng.binomial(1, 0.85, n_legit).astype(float)

    # has_ip_address (binary)
    X[phishing_mask, 5] = rng.binomial(1, 0.35, n_phish).astype(float)
    X[~phishing_mask, 5] = rng.binomial(1, 0.02, n_legit).astype(float)

    # suspicious_tld (binary)
    X[phishing_mask, 6] = rng.binomial(1, 0.45, n_phish).astype(float)
    X[~phishing_mask, 6] = rng.binomial(1, 0.05, n_legit).astype(float)

    # num_subdomains
    X[phishing_mask, 7] = rng.poisson(3, n_phish).astype(float)
    X[~phishing_mask, 7] = rng.poisson(1, n_legit).astype(float)

    # path_length
    X[phishing_mask, 8] = rng.normal(50, 15, n_phish)
    X[~phishing_mask, 8] = rng.normal(20, 8, n_legit)

    # has_at_symbol (binary)
    X[phishing_mask, 9] = rng.binomial(1, 0.20, n_phish).astype(float)
    X[~phishing_mask, 9] = rng.binomial(1, 0.01, n_legit).astype(float)

    return X


def _generate_temporal_features(n_samples: int, labels: np.ndarray,
                                rng: np.random.Generator) -> np.ndarray:
    """Generate 5 temporal features.

    Phishing emails often arrive at odd hours, from newly-registered
    domains, with high burst frequency.
    """
    X = np.zeros((n_samples, 5), dtype=np.float64)
    phishing_mask = labels == 1
    n_phish = phishing_mask.sum()
    n_legit = n_samples - n_phish

    # hour_sent (0-23)
    X[phishing_mask, 0] = rng.choice(
        np.concatenate([np.arange(0, 6), np.arange(22, 24)]),
        size=n_phish
    ).astype(float) + rng.uniform(-0.5, 0.5, n_phish)
    X[~phishing_mask, 0] = rng.normal(12, 3, n_legit)
    X[:, 0] = np.clip(X[:, 0], 0, 23)

    # day_of_week (0=Mon, 6=Sun)
    X[phishing_mask, 1] = rng.choice(7, n_phish).astype(float)
    X[~phishing_mask, 1] = rng.choice(5, n_legit).astype(float)  # workdays

    # domain_age_days
    X[phishing_mask, 2] = rng.exponential(30, n_phish)
    X[~phishing_mask, 2] = rng.exponential(1500, n_legit)

    # time_since_last_email_hrs
    X[phishing_mask, 3] = rng.exponential(2, n_phish)
    X[~phishing_mask, 3] = rng.exponential(48, n_legit)

    # email_frequency_per_day
    X[phishing_mask, 4] = rng.exponential(8, n_phish)
    X[~phishing_mask, 4] = rng.exponential(1.5, n_legit)

    return X


def generate_synthetic_dataset(n_samples: int = 5000,
                               phishing_ratio: float = 0.4,
                               random_state: int = 42) -> dict:
    """Generate complete multi-modal synthetic phishing dataset.

    Parameters
    ----------
    n_samples : int
        Total number of samples to generate.
    phishing_ratio : float
        Proportion of phishing samples (label=1).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    dict with keys:
        'X_text', 'X_url', 'X_temporal', 'X_combined', 'y',
        'feature_names_text', 'feature_names_url',
        'feature_names_temporal', 'feature_names_all', 'df'
    """
    rng = np.random.default_rng(random_state)

    # Generate labels
    n_phishing = int(n_samples * phishing_ratio)
    n_legit = n_samples - n_phishing
    labels = np.concatenate([np.ones(n_phishing), np.zeros(n_legit)])

    # Shuffle
    shuffle_idx = rng.permutation(n_samples)
    labels = labels[shuffle_idx]

    # Generate each modality
    X_text = _generate_text_features(n_samples, labels, rng)
    X_url = _generate_url_features(n_samples, labels, rng)
    X_temporal = _generate_temporal_features(n_samples, labels, rng)

    # Combined feature matrix (late fusion input)
    X_combined = np.hstack([X_text, X_url, X_temporal])

    # Build DataFrame
    df = pd.DataFrame(X_combined, columns=ALL_FEATURE_NAMES)
    df["label"] = labels.astype(int)

    print(f"[DataLoader] Generated {n_samples} samples "
          f"({n_phishing} phishing, {n_legit} legitimate)")
    print(f"  Text features   : {X_text.shape[1]}")
    print(f"  URL features    : {X_url.shape[1]}")
    print(f"  Temporal features: {X_temporal.shape[1]}")
    print(f"  Combined        : {X_combined.shape[1]}")

    return {
        "X_text": X_text,
        "X_url": X_url,
        "X_temporal": X_temporal,
        "X_combined": X_combined,
        "y": labels.astype(int),
        "feature_names_text": TEXT_FEATURE_NAMES,
        "feature_names_url": URL_FEATURE_NAMES,
        "feature_names_temporal": TEMPORAL_FEATURE_NAMES,
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

    for key, X, names in [
        ("text", data["X_text"], data["feature_names_text"]),
        ("url", data["X_url"], data["feature_names_url"]),
        ("temporal", data["X_temporal"], data["feature_names_temporal"]),
        ("combined", data["X_combined"], data["feature_names_all"]),
    ]:
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
    data = generate_synthetic_dataset(n_samples=5000)
    splits = prepare_splits(data)
    print("\nModality shapes (train):")
    for mod in ["text", "url", "temporal", "combined"]:
        print(f"  {mod:10s}: {splits[mod]['X_train'].shape}")
