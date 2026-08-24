"""Evaluate physics-consistency proof metrics WITHOUT inference clamping.

Computes, per checkpoint tag:
  - raw NP%  (fraction of unclamped predictions < 0)
  - monotonicity violation %  (fraction of samples with dEg/dchi_b > 0)
  - clamped R2 / RMSE / MAE  (standard metrics, clip_min=0)

Usage:
    python src/analysis/physics_proof.py --dataset dataset_b \
        --phys-tags pg_seed1 pg_seed2 pg_seed3 \
        --std-tags st_seed1 st_seed2 st_seed3
"""
import argparse
import json
import os
import statistics
import sys

# Make `import src` work when running this file directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import yaml

from src.analysis.helpers import ANALYSIS_DIR
from src.data.prepare import load_splits
from src.models.pg_sann import build_model
from src.utils.metrics import evaluate, nonphys_rate


def _col_idx(feature_cols, names):
    for i, c in enumerate(feature_cols):
        if c.lower() in names:
            return i
    return None


def _load(dataset, tag):
    run_dir = os.path.join("outputs", "models", dataset, tag)
    ckpt = os.path.join(run_dir, "best.pt")
    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(ckpt):
        return None
    cfg = yaml.safe_load(open(cfg_path))
    splits = load_splits(dataset)
    X = np.asarray(splits["X_test"], dtype=np.float32)
    y = np.asarray(splits["y_test"], dtype=np.float32)
    model = build_model(X.shape[1], {"model": cfg["model"]})
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    return model, splits, X, y


def _analyze(model, X, y, chi_idx, nsample=400):
    with torch.no_grad():
        pred = model(torch.tensor(X)).cpu().numpy().ravel()
    raw_np = nonphys_rate(pred)
    clamped = evaluate(y, np.clip(pred, 0.0, None))
    viol = float("nan")
    if chi_idx is not None:
        idx = np.random.RandomState(0).choice(len(X), min(nsample, len(X)), replace=False)
        Xg = torch.tensor(X[idx], requires_grad=True)
        yh = model(Xg).sum()
        grads = torch.autograd.grad(yh, Xg)[0][:, chi_idx].cpu().numpy()
        viol = float(np.mean(grads > 0))
    return {
        "raw_np": raw_np,
        "mono_viol": viol,
        "r2": clamped["r2"],
        "rmse": clamped["rmse"],
        "mae": clamped["mae"],
    }


def _summarize(rows):
    out = {}
    for key in ("raw_np", "mono_viol", "r2", "rmse", "mae"):
        vals = [r[key] for r in rows if r is not None and r[key] == r[key]]
        if not vals:
            out[key] = None
        else:
            out[key] = {
                "mean": statistics.mean(vals),
                "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "n": len(vals),
            }
    return out


def _fmt(stat, pct=False, dp=3):
    if stat is None:
        return "    --   "
    s = 100.0 if pct else 1.0
    return f"{stat['mean'] * s:7.3f} ± {stat['std'] * s:.3f}"


def run(dataset, phys_tags, std_tags):
    splits = load_splits(dataset)
    chi_idx = _col_idx(splits["feature_cols"], ["electronegativity_b", "x_b", "chi_b", "x_mean"])

    def load_and_analyze(tag):
        loaded = _load(dataset, tag)
        if loaded is None:
            print(f"  (skip {tag}: no checkpoint)", flush=True)
            return None
        model, splits_, X, y = loaded
        return _analyze(model, X, y, chi_idx)

    phys_rows = [load_and_analyze(t) for t in phys_tags]
    std_rows = [load_and_analyze(t) for t in std_tags]
    phys = _summarize(phys_rows)
    std = _summarize(std_rows)

    print(f"\n=== {dataset} (raw, unclamped; NP% of preds < 0) ===")
    print(f"  model   rawNP%            monoViol%          R2        RMSE      MAE")
    print(f"  ST      {_fmt(std['raw_np'], pct=True)}      {_fmt(std['mono_viol'], pct=True)}      "
          f"{_fmt(std['r2'])}  {_fmt(std['rmse'])}  {_fmt(std['mae'])}")
    print(f"  PG-SANN {_fmt(phys['raw_np'], pct=True)}      {_fmt(phys['mono_viol'], pct=True)}      "
          f"{_fmt(phys['r2'])}  {_fmt(phys['rmse'])}  {_fmt(phys['mae'])}")

    out = os.path.join(ANALYSIS_DIR, f"{dataset}_physics_proof.json")
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"dataset": dataset, "ST": std, "PG-SANN": phys}, f, indent=2)
    print(f"  saved -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--phys-tags", nargs="+", required=True)
    ap.add_argument("--std-tags", nargs="+", required=True)
    args = ap.parse_args()
    run(args.dataset, args.phys_tags, args.std_tags)


if __name__ == "__main__":
    main()
