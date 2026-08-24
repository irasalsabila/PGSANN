"""Generate core diagnostic figures for PG-SANN.

Usage:
    python src/analysis/make_figures.py --dataset dataset_a [--tag physics]
"""
import argparse
import os
import sys

# Make `import src` work when running this file directly (no PYTHONPATH needed).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.helpers import (
    ANALYSIS_DIR, FIGURES_DIR, ensure_dirs, get_best_checkpoint_tag,
    load_trained_model, predict_model, save_summary_json,
)
from src.data.prepare import load_splits
from src.utils.metrics import evaluate, nonphys_rate


def _bin_name(v):
    if v == 0:
        return "metallic"
    if v <= 1.0:
        return "semi-small"
    if v <= 3.0:
        return "semi-med"
    return "insulator"


def _per_bin_errors(y, pred):
    y = np.asarray(y); pred = np.asarray(pred)
    rows = {}
    for b in sorted(set(_bin_name(v) for v in y)):
        mask = np.array([_bin_name(v) == b for v in y])
        if mask.sum() == 0:
            continue
        m = evaluate(y[mask], pred[mask])
        rows[b] = {"n": int(mask.sum()), **{k: round(v, 4) for k, v in m.items()}}
    return rows


def fig_pred_vs_actual(y, pred, title, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    m = evaluate(y, pred)
    ax.scatter(y, pred, s=12, alpha=0.5, edgecolors="none")
    lo, hi = min(y.min(), pred.min()), max(y.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="ideal")
    ax.set_xlabel("True band gap (eV)")
    ax.set_ylabel("Predicted band gap (eV)")
    ax.set_title(title)
    ax.text(0.05, 0.95, f"R²={m['r2']:.3f}\nRMSE={m['rmse']:.3f} eV\nMAE={m['mae']:.3f} eV",
            transform=ax.transAxes, va="top", bbox=dict(boxstyle="round", fc="white"))
    ax.legend(loc="lower right")
    fig.savefig(path)
    plt.close(fig)
    return m


def fig_residuals(y, pred, title, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    resid = pred - y
    ax.scatter(pred, resid, s=12, alpha=0.5)
    ax.axhline(0, color="r", lw=1.2)
    ax.set_xlabel("Predicted band gap (eV)")
    ax.set_ylabel("Residual (pred - true) (eV)")
    ax.set_title(title)
    fig.savefig(path)
    plt.close(fig)


def fig_prediction_distribution(y, pred, title, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(min(y.min(), pred.min()), max(y.max(), pred.max()), 50)
    ax.hist(y, bins=bins, alpha=0.6, label="true", color="tab:blue")
    ax.hist(pred, bins=bins, alpha=0.6, label="predicted", color="tab:orange")
    nphys = nonphys_rate(pred)
    ax.set_xlabel("Band gap (eV)")
    ax.set_ylabel("Count")
    ax.set_title(title + f"  (NonPhys%={nphys*100:.2f})")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def fig_ablation_nonphys(dataset):
    """Bar chart of NonPhys% + R² across ablation variants from saved metrics.json."""
    from src.analysis.helpers import MODELS_DIR

    variants = {"v0_std": "Variant 0 (std)", "v1_bounds": "Variant 1 (bounds)", "physics": "Full PG-SANN"}
    labels, nonphys, r2s = [], [], []
    for tag, label in variants.items():
        run_dir = os.path.join(MODELS_DIR, dataset, tag)
        met_path = os.path.join(run_dir, "metrics.json")
        if not os.path.exists(met_path):
            print(f"  (skip {tag}: no metrics.json — not trained yet)", flush=True)
            continue
        import json
        with open(met_path) as f:
            m = json.load(f)
        labels.append(label); nonphys.append(m.get("nonphys_rate", 0.0)); r2s.append(m.get("r2", 0.0))
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(labels))
    ax.bar(x - 0.2, nonphys, width=0.4, label="NonPhys% (lower better)", color="tab:red")
    ax.bar(x + 0.2, r2s, width=0.4, label="R² (higher better)", color="tab:green")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Score")
    ax.set_title(f"Ablation comparison — {dataset}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(FIGURES_DIR, f"{dataset}_ablation.png"))
    plt.close(fig)
    print(f"  ablation chart -> {FIGURES_DIR}/{dataset}_ablation.png", flush=True)


def run(dataset, tag):
    ensure_dirs()
    if tag is None:
        tag = get_best_checkpoint_tag(dataset)
    print(f"[{dataset}] loading model (tag='{tag}') ...", flush=True)
    model, cfg, splits = load_trained_model(dataset, tag=tag)
    y_test = splits["y_test"]
    pred_test = predict_model(model, splits["X_test"])
    print(f"  test r2={evaluate(y_test, pred_test)['r2']:.4f} "
          f"rmse={evaluate(y_test, pred_test)['rmse']:.4f}", flush=True)

    tag_label = "PG-SANN" if tag in ("physics", "full") else f"PG-SANN ({tag})"
    fig_pred_vs_actual(y_test, pred_test, f"{tag_label} — {dataset}",
                       os.path.join(FIGURES_DIR, f"{dataset}_pred_vs_actual_{tag}.png"))
    fig_residuals(y_test, pred_test, f"{tag_label} residuals — {dataset}",
                  os.path.join(FIGURES_DIR, f"{dataset}_residuals_{tag}.png"))
    fig_prediction_distribution(y_test, pred_test, f"{tag_label} — {dataset}",
                                os.path.join(FIGURES_DIR, f"{dataset}_dist_{tag}.png"))

    per_bin = _per_bin_errors(y_test, pred_test)
    print(f"  per-bin errors: {per_bin}", flush=True)
    # Save per-bin table as CSV
    pd.DataFrame.from_dict(per_bin, orient="index").to_csv(
        os.path.join(ANALYSIS_DIR, f"{dataset}_perbin_{tag}.csv")
    )
    save_summary_json(f"{dataset}_figures_{tag}.json", {
        "dataset": dataset,
        "tag": tag,
        "test": evaluate(y_test, pred_test),
        "per_bin": per_bin,
    })
    fig_ablation_nonphys(dataset)
    print(f"  figures -> {FIGURES_DIR}/", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tag", default=None, help="checkpoint tag (default: physics)")
    args = ap.parse_args()
    run(args.dataset, args.tag)


if __name__ == "__main__":
    main()
