"""Shared helpers for analysis/visualization scripts."""
import json
import os

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.prepare import load_splits
from src.models.losses import build_loss
from src.models.pg_sann import PG_SANN
from src.utils.config import load_config, PROJECT_ROOT
from src.utils.metrics import evaluate

FIGURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
ANALYSIS_DIR = os.path.join(PROJECT_ROOT, "outputs", "analysis")
MODELS_DIR = os.path.join(PROJECT_ROOT, "outputs", "models")

# Publication style
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def ensure_dirs():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)


def locate_indices(feature_cols):
    t_idx, chi_b_idx = None, None
    lower = [c.lower() for c in feature_cols]
    for i, name in enumerate(lower):
        if name in ("t", "gtf") or "tolerance" in name:
            t_idx = i
        if name in ("electronegativity_b", "x_b", "chi_b", "en_b", "x_mean"):
            chi_b_idx = i
    return t_idx, chi_b_idx


def predict_model(model, X, device="cpu", clip_min=0.0):
    """Run a trained model on an array -> numpy predictions.

    Predictions are clamped to >= clip_min (band gaps cannot be negative).
    """
    model.eval()
    X = torch.tensor(np.asarray(X, dtype=np.float32)).to(device)
    with torch.no_grad():
        preds = model(X)
        if clip_min is not None:
            preds = torch.clamp(preds, min=clip_min)
        preds = preds.cpu().numpy()
    return preds


def load_trained_model(dataset, tag=None, device="cpu"):
    """Load a trained PG-SANN checkpoint + its config.

    If tag is None, auto-select the best available checkpoint via
    get_best_checkpoint_tag. Expects outputs/models/{dataset}/{tag}/best.pt
    and config.yaml.
    """
    if tag is None:
        tag = get_best_checkpoint_tag(dataset)
        print(f"  auto-selected tag='{tag}'", flush=True)
    run_dir = os.path.join(MODELS_DIR, dataset, tag)
    ckpt = os.path.join(run_dir, "best.pt")
    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"No checkpoint at {ckpt}. Train first, e.g. "
            f"python src/models/train_pg_sann.py --config configs/pg_sann.yaml "
            f"--dataset {dataset} --tag {tag}"
        )
    cfg = load_config(cfg_path)
    splits = load_splits(dataset)
    n_features = splits["X_train"].shape[1]
    m = cfg.get("model", {})
    model = PG_SANN(
        n_features=n_features,
        d_model=int(m.get("d_model", 64)),
        n_heads=int(m.get("n_heads", 4)),
        n_layers=int(m.get("n_layers", 3)),
        d_ffn=int(m.get("d_ffn", 128)),
        dropout=float(m.get("dropout", 0.1)),
        pool=str(m.get("pool", "mean")),
    )
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    return model, cfg, splits


def get_best_checkpoint_tag(dataset):
    """Return the tag of the most-recently-created 'physics' run if present, else 'physics'."""
    physics_dir = os.path.join(MODELS_DIR, dataset, "physics")
    if os.path.isdir(physics_dir) and os.path.exists(os.path.join(physics_dir, "best.pt")):
        return "physics"
    # fall back to any subdir containing best.pt
    if os.path.isdir(os.path.join(MODELS_DIR, dataset)):
        for tag in sorted(os.listdir(os.path.join(MODELS_DIR, dataset))):
            if os.path.exists(os.path.join(MODELS_DIR, dataset, tag, "best.pt")):
                return tag
    return "physics"


def save_summary_json(name, data):
    ensure_dirs()
    with open(os.path.join(ANALYSIS_DIR, name), "w") as f:
        json.dump(data, f, indent=2)
