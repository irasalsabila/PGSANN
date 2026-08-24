"""Generate the Discussion figures for PG-SANN vs Standard Transformer.

Produces two combined publication figures across Datasets B, C, and D:
  - fig2_parity.png    : 2x3 grid of raw pred-vs-actual (Dataset B top row,
                         C middle row, D bottom row; ST left, PG-SANN right),
                         with the non-physical zone (Eg<0) shaded red + inset
                         negative-histogram
  - fig3_gradient.png  : 3-panel KDE of dEg/dchi_B for Datasets B, C, and D,
                         line at 0, forbidden zone shaded

Usage:
    python src/analysis/discussion_figures.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy import stats

from src.analysis.helpers import FIGURES_DIR, ensure_dirs
from src.data.prepare import load_splits
from src.models.pg_sann import build_model
from src.utils.metrics import nonphys_rate

# Publication style
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

NEG_COLOR = "tab:red"
POS_COLOR = "tab:blue"
PHYS_COLOR = "tab:green"

# Map internal dataset names to manuscript labels.
DATASET_LABELS = {
    "dataset_b": "A",
    "dataset_c": "B",
    "dataset_d": "C",
}


def _dlabel(dataset):
    return DATASET_LABELS.get(dataset, dataset.replace("dataset_", "").upper())


def _load(dataset, tag):
    run_dir = os.path.join("outputs", "models", dataset, tag)
    ckpt = os.path.join(run_dir, "best.pt")
    cfg_path = os.path.join(run_dir, "config.yaml")
    cfg = yaml.safe_load(open(cfg_path))
    splits = load_splits(dataset)
    X = np.asarray(splits["X_test"], dtype=np.float32)
    y = np.asarray(splits["y_test"], dtype=np.float32)
    model = build_model(X.shape[1], {"model": cfg["model"]})
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    return model, cfg, splits, X, y


def _col_idx(feature_cols, names):
    for i, c in enumerate(feature_cols):
        if c.lower() in names:
            return i
    return None


def _raw_pred(model, X):
    with torch.no_grad():
        return model(torch.tensor(X)).cpu().numpy().ravel()


def _grad_chi(model, X, chi_idx, nsample=600):
    idx = np.random.RandomState(0).choice(len(X), min(nsample, len(X)), replace=False)
    Xg = torch.tensor(X[idx], requires_grad=True)
    yh = model(Xg).sum()
    grads = torch.autograd.grad(yh, Xg)[0][:, chi_idx].cpu().numpy()
    return grads


def _parity_ax(ax, y, pred, title):
    lo = min(y.min(), pred.min(), 0.0)
    hi = max(y.max(), pred.max(), 0.01)
    ax.axhspan(lo, 0, color=NEG_COLOR, alpha=0.10)
    ax.axvspan(lo, 0, color=NEG_COLOR, alpha=0.10)
    ax.scatter(y, pred, s=10, alpha=0.5, edgecolors="none", color=POS_COLOR)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="ideal")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_title(title)
    ax.axhline(0, color=NEG_COLOR, lw=0.8, ls=":")
    neg = pred[pred < 0]
    ax_ins = ax.inset_axes([0.55, 0.06, 0.4, 0.3])
    ax_ins.hist(pred, bins=60, range=(lo, 0.0), density=True,
                color=NEG_COLOR, alpha=0.5)
    if len(neg):
        xk = np.linspace(lo, 0, 200)
        k = stats.gaussian_kde(neg)
        ax_ins.plot(xk, k(xk), color=NEG_COLOR)
    ax_ins.set_xlim(lo, 0)
    ax_ins.set_title(f"neg. {100*nonphys_rate(pred):.1f}%", fontsize=9)
    ax.legend(loc="upper left", fontsize=9)


def fig2_parity(datasets):
    rows, cols = 2, len(datasets)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 8.2), sharex="col", sharey="row")
    for c, d in enumerate(datasets):
        label = _dlabel(d)
        pg_model, _, _, X, y = _load(d, "pg_seed1")
        st_model, _, _, _, _ = _load(d, "st_seed1")
        pg_pred = _raw_pred(pg_model, X)
        st_pred = _raw_pred(st_model, X)
        _parity_ax(axes[0, c], y, st_pred, f"Standard Transformer — Dataset {label}")
        _parity_ax(axes[1, c], y, pg_pred, f"PG-SANN — Dataset {label}")
        axes[0, c].set_xlabel("")
        axes[1, c].set_xlabel("True band gap (eV)")
        axes[0, c].set_ylabel("Predicted band gap (eV)" if c == 0 else "")
        axes[1, c].set_ylabel("Predicted band gap (eV)" if c == 0 else "")
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig2_parity.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  fig2 -> {out}", flush=True)


def _grad_ax(ax, st_g, pg_g, title):
    lo = min(st_g.min(), pg_g.min())
    hi = max(st_g.max(), pg_g.max())
    ax.axvspan(0, hi, color=NEG_COLOR, alpha=0.10)
    x = np.linspace(lo, hi, 400)
    ax.plot(x, stats.gaussian_kde(st_g)(x), color=POS_COLOR,
            label=f"ST (viol. {100*np.mean(st_g>0):.0f}%)")
    ax.plot(x, stats.gaussian_kde(pg_g)(x), color=PHYS_COLOR,
            label=f"PG-SANN (viol. {100*np.mean(pg_g>0):.0f}%)")
    ax.axvline(0, color=NEG_COLOR, lw=1.2, ls="--")
    ax.set_xlabel(r"$\partial\hat{E}_g / \partial\chi_B$")
    ax.set_title(title)
    ax.legend(fontsize=10)


def fig3_gradient(datasets):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.5))
    for i, d in enumerate(datasets):
        label = _dlabel(d)
        pg_model, _, splits, X, _ = _load(d, "pg_seed1")
        st_model, _, _, _, _ = _load(d, "st_seed1")
        chi_idx = _col_idx(splits["feature_cols"], ["electronegativity_b", "x_b", "chi_b", "x_mean"])
        st_g = _grad_chi(st_model, X, chi_idx)
        pg_g = _grad_chi(pg_model, X, chi_idx)
        _grad_ax(axes[i], st_g, pg_g, f"Dataset {label}")
        if i == 0:
            axes[i].set_ylabel("Density")
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig3_gradient.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  fig3 -> {out}", flush=True)


def run():
    ensure_dirs()
    datasets = ["dataset_b", "dataset_c", "dataset_d"]
    fig2_parity(datasets)
    fig3_gradient(datasets)


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    run()


if __name__ == "__main__":
    main()

