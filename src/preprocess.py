"""
preprocess.py

Loads CICIDS2017 CSVs, cleans them, encodes labels,
and saves a processed dataset ready for training.

Usage:
    python src/preprocess.py
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

# Features to drop (identifiers, not useful for ML)
DROP_COLS = [
    "Flow ID", "Source IP", "Destination IP",
    "Source Port", "Destination Port", "Timestamp",
    " Flow ID", " Source IP", " Destination IP",
    " Source Port", " Destination Port", " Timestamp",
]

# Map verbose attack labels to cleaner names
LABEL_MAP = {
    "BENIGN": "Benign",
    "DoS Hulk": "DoS",
    "PortScan": "PortScan",
    "DDoS": "DDoS",
    "DoS GoldenEye": "DoS",
    "FTP-Patator": "Brute Force",
    "SSH-Patator": "Brute Force",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "Bot": "Bot",
    "Web Attack \x96 Brute Force": "Web Attack",
    "Web Attack \x96 XSS": "Web Attack",
    "Web Attack \x96 Sql Injection": "Web Attack",
    "Infiltration": "Infiltration",
    "Heartbleed": "Heartbleed",
}


def load_csvs(raw_dir: str) -> pd.DataFrame:
    """Load and concatenate all CICIDS2017 CSV files."""
    files = glob.glob(os.path.join(raw_dir, "*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}. "
            "See data/README.md for download instructions."
        )
    print(f"Found {len(files)} CSV files:")
    dfs = []
    for f in files:
        print(f"  Loading {os.path.basename(f)}...")
        df = pd.read_csv(f, encoding="utf-8", low_memory=False)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total rows loaded: {len(combined):,}")
    return combined


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier columns, handle inf/NaN, strip whitespace from headers."""
    # Strip leading/trailing spaces from column names
    df.columns = df.columns.str.strip()

    # Drop identifier columns if present
    cols_to_drop = [c for c in DROP_COLS if c.strip() in df.columns]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # Replace inf values with NaN then drop
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    before = len(df)
    df.dropna(inplace=True)
    print(f"Dropped {before - len(df):,} rows with NaN/inf values.")

    return df


def encode_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    """Map verbose attack labels and encode as integers."""
    label_col = "Label"
    df[label_col] = df[label_col].map(LABEL_MAP).fillna(df[label_col])

    le = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df[label_col])
    print(f"Classes: {list(le.classes_)}")
    print(f"Label distribution:\n{df[label_col].value_counts()}\n")
    return df, le


def scale_features(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Separate features/labels and apply StandardScaler."""
    label_cols = ["Label", "label_encoded"]
    feature_cols = [c for c in df.columns if c not in label_cols]

    X = df[feature_cols].values.astype(np.float32)
    y = df["label_encoded"].values.astype(np.int64)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, scaler


def save_processed(
    X: np.ndarray,
    y: np.ndarray,
    scaler: StandardScaler,
    le: LabelEncoder,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X.npy"), X)
    np.save(os.path.join(out_dir, "y.npy"), y)
    with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(out_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    print(f"Saved processed data to {out_dir}/")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")


def main():
    df = load_csvs(RAW_DIR)
    df = clean(df)
    df, le = encode_labels(df)
    X, y, scaler = scale_features(df)
    save_processed(X, y, scaler, le, PROCESSED_DIR)
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
