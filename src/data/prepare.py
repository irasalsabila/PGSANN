import json
import os
import sys
import warnings

# Make `import src` work when running this file directly (no PYTHONPATH needed).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Silence benign mendeleev "multiple allotropes" warnings during Magpie featurization.
warnings.filterwarnings("ignore", message=".*allotropes.*")
warnings.filterwarnings("ignore", message=".*multiple allotropes.*")

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.features import add_elemental_descriptors, add_physics_features
from src.data.loaders import DatasetInfo, load_dataset
from src.data.magpie import add_magpie_features
from src.utils.config import PROJECT_ROOT

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
SPLITS_DIR = os.path.join(PROJECT_ROOT, "data", "splits")


def _stratify_bands(y, n_bins: int = 4):
    """Stratification labels for band-gap target (handles zero-heavy skew)."""
    y = np.asarray(y, dtype=float)
    # Bin 0 = metallic (0); remaining into quartiles.
    labels = np.zeros(len(y), dtype=int)
    non_zero = y > 0
    labels[non_zero] = 1 + pd.qcut(
        y[non_zero], q=n_bins, labels=False, duplicates="drop"
    )
    return labels


def prepare_dataset(name: str, seed: int = 42, val_frac: float = 0.15, test_frac: float = 0.15):
    """Load, clean, add physics features, and create stratified splits."""
    print(f"[{name}] loading raw data ...", flush=True)
    info = load_dataset(name)
    print(f"  rows={len(info.df)} raw_features={len(info.feature_cols)} target={info.target_col}", flush=True)

    print(f"[{name}] feature engineering (t, mu, elemental, Magpie) ...", flush=True)
    info = add_physics_features(info)
    info = add_elemental_descriptors(info)
    # Add Magpie features (matching B/C schema) for formula-only datasets (D, E).
    info.df = add_magpie_features(info)
    for col in info.df.columns:
        if col.endswith("_mean") or col.endswith("_std"):
            if col not in info.feature_cols:
                info.feature_cols = info.feature_cols + [col]
    print(f"  final features={len(info.feature_cols)}", flush=True)

    df = info.df.reset_index(drop=True)
    X = df[info.feature_cols].astype(float)
    y = df[info.target_col].astype(float)

    # Handle any remaining NaN (should be none for A/B; variant has some).
    X = X.fillna(X.median(numeric_only=True))
    if y.isna().any():
        y = y.fillna(y.median())

    # Split: train / val / test, stratified on band-gap bins.
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_frac, random_state=seed, stratify=_stratify_bands(y)
    )
    # Split the remaining (train+val) into train/val.
    remaining_idx = X.index.difference(Xte.index) if not isinstance(Xte, np.ndarray) else None
    if isinstance(X, pd.DataFrame) and remaining_idx is not None:
        Xtr_val = X.loc[remaining_idx]
        ytr_val = y.loc[remaining_idx]
    else:
        Xtr_val, ytr_val = X, y
    Xtr, Xval, ytr, yval = train_test_split(
        Xtr_val, ytr_val, test_size=val_frac / (1 - test_frac),
        random_state=seed, stratify=_stratify_bands(ytr_val),
    )

    # Fit StandardScaler on TRAIN ONLY to avoid leakage; save it for reuse.
    out_dir = os.path.join(PROCESSED_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = pd.DataFrame(scaler.transform(Xtr), columns=Xtr.columns, index=Xtr.index)
    Xval_s = pd.DataFrame(scaler.transform(Xval), columns=Xval.columns, index=Xval.index)
    Xte_s = pd.DataFrame(scaler.transform(Xte), columns=Xte.columns, index=Xte.index)
    dump(scaler, os.path.join(out_dir, "scaler.joblib"))

    meta = {
        "dataset": name,
        "n_train": len(Xtr),
        "n_val": len(Xval),
        "n_test": len(Xte),
        "feature_cols": info.feature_cols,
        "target": info.target_col,
        "formula_col": info.formula_col,
        "scaled": True,
    }

    np.save(os.path.join(out_dir, "X_train.npy"), Xtr_s.values)
    np.save(os.path.join(out_dir, "X_val.npy"), Xval_s.values)
    np.save(os.path.join(out_dir, "X_test.npy"), Xte_s.values)
    np.save(os.path.join(out_dir, "y_train.npy"), ytr.values)
    np.save(os.path.join(out_dir, "y_val.npy"), yval.values)
    np.save(os.path.join(out_dir, "y_test.npy"), yte.values)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Cache full processed frame too.
    X.to_csv(os.path.join(out_dir, "X.csv"), index=False)
    y.to_csv(os.path.join(out_dir, "y.csv"), index=False)
    return meta


def load_splits(name: str):
    """Load cached splits for a dataset."""
    out_dir = os.path.join(PROCESSED_DIR, name)
    with open(os.path.join(out_dir, "meta.json")) as f:
        meta = json.load(f)
    return {
        "X_train": np.load(os.path.join(out_dir, "X_train.npy")),
        "X_val": np.load(os.path.join(out_dir, "X_val.npy")),
        "X_test": np.load(os.path.join(out_dir, "X_test.npy")),
        "y_train": np.load(os.path.join(out_dir, "y_train.npy")),
        "y_val": np.load(os.path.join(out_dir, "y_val.npy")),
        "y_test": np.load(os.path.join(out_dir, "y_test.npy")),
        "feature_cols": meta["feature_cols"],
        "target": meta["target"],
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Prepare a dataset into processed splits.")
    ap.add_argument("--dataset", default="dataset_a",
                    help="dataset_a | dataset_b | dataset_b_variant")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    meta = prepare_dataset(args.dataset, seed=args.seed)
    print(f"Prepared {args.dataset}: {meta}")
