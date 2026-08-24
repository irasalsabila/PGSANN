"""Extract and visualize PG-SANN attention across A/B/X sites.

Usage:
    python src/analysis/attention_maps.py --dataset dataset_a [--tag physics]
"""
import argparse
import os
import sys

# Make `import src` work when running this file directly (no PYTHONPATH needed).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.analysis.helpers import (
    FIGURES_DIR, ensure_dirs, get_best_checkpoint_tag, locate_indices,
    load_trained_model, save_summary_json,
)
from src.data.prepare import load_splits

# Site heuristic defined in _site_of (keyword-based).


def _site_of(feature):
    fl = feature.lower()
    # Physics descriptors
    if fl in ("t", "mu", "gtf", "of", "r_a", "r_b", "r_x") or "tolerance" in fl or "octahedral" in fl:
        return "physics"
    # A-site: organic cations + alkali/alkaline + A-site element counts
    if fl in ("ma", "fa", "dma", "ea", "ga") or fl in (
        "cs", "rb", "k", "na", "li", "ba", "sr", "ca", "la", "ce", "pr", "nd",
        "sm", "eu", "gd", "tb", "dy", "ho", "er", "tm", "bi"):
        return "A"
    # B-site: transition/post-transition metals + B descriptors
    if fl in ("ti", "zr", "hf", "v", "nb", "ta", "cr", "mo", "w", "mn", "fe", "co",
              "ni", "cu", "zn", "cd", "al", "ga", "in", "tl", "ge", "sn", "pb", "sb",
              "sc", "y", "ru", "rh", "pd", "ag", "ir", "pt", "au", "u", "th") or \
       "electronegativity_b" in fl:
        return "B"
    # X-site: halides/chalcogenides + X descriptors
    if fl in ("cl", "br", "i", "f", "o", "s", "se", "te", "n") or \
       "electronegativity_x" in fl or "ionization_energy_x" in fl or "electron_affinity_x" in fl:
        return "X"
    # composition weight / structural
    if "weight" in fl or "structure" in fl:
        return "other"
    return "other"


def extract_attention(model, X, n_layers, n_heads):
    """Return per-layer attention maps aggregated over samples.
    attention shape: (n_layers, B, n_tokens, n_tokens) -> (n_layers, n_tokens, n_tokens) mean.
    """
    device = next(model.parameters()).device
    X = torch.tensor(np.asarray(X, dtype=np.float32)).to(device)
    model.eval()
    with torch.no_grad():
        _, attn_maps = model(X, return_attention=True)
    # attn_maps: list of (B, n_heads, n_tokens, n_tokens) if average_attn_weights=False
    # In our model we used average_attn_weights=True -> (B, n_tokens, n_tokens)
    aggregated = []
    for layer_attn in attn_maps:
        # layer_attn: (B, n_tokens, n_tokens) or (B, n_heads, n_tokens, n_tokens)
        if layer_attn.dim() == 4:
            layer_attn = layer_attn.mean(dim=1)  # avg over heads
        aggregated.append(layer_attn.mean(dim=0).cpu().numpy())  # avg over batch
    return aggregated


def _plot_site_heatmap(mat, labels, title, path):
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.4), max(6, len(labels) * 0.4)))
    im = ax.imshow(mat, cmap="viridis")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=6)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run(dataset, tag):
    ensure_dirs()
    if tag is None:
        tag = get_best_checkpoint_tag(dataset)
    print(f"[{dataset}] loading model (tag='{tag}') ...", flush=True)
    model, cfg, splits = load_trained_model(dataset, tag=tag)
    feature_cols = splits["feature_cols"]
    n_layers = int(cfg.get("model", {}).get("n_layers", 3))
    n_heads = int(cfg.get("model", {}).get("n_heads", 4))
    n_features = len(feature_cols)
    print(f"  features={n_features} layers={n_layers} heads={n_heads}", flush=True)

    # Use a subset of the test set for attention aggregation (faster).
    X = splits["X_test"]
    if len(X) > 512:
        X = X[:512]
    attn_layers = extract_attention(model, X, n_layers, n_heads)
    print(f"  extracted attention for {len(X)} samples, {n_layers} layers", flush=True)

    # Feature labels (map raw feature names to site tags)
    site_of = [_site_of(c) for c in feature_cols]
    labels = [f"{c[:12]}" for c in feature_cols]

    # Aggregate attention across layers (mean) -> single feature x feature map
    agg = np.mean(attn_layers, axis=0)
    _plot_site_heatmap(agg, labels, f"{dataset} — attention (mean over layers, {tag})",
                       os.path.join(FIGURES_DIR, f"{dataset}_attention_{tag}.png"))

    # Per-layer heatmaps
    for li, layer_attn in enumerate(attn_layers):
        _plot_site_heatmap(layer_attn, labels,
                           f"{dataset} — attention layer {li+1} ({tag})",
                           os.path.join(FIGURES_DIR, f"{dataset}_attention_layer{li+1}_{tag}.png"))

    # Site-level aggregation: mean attention between site groups
    sites = sorted(set(site_of))
    site_mat = np.zeros((len(sites), len(sites)))
    for i, si in enumerate(sites):
        for j, sj in enumerate(sites):
            ii = [k for k, s in enumerate(site_of) if s == si]
            jj = [k for k, s in enumerate(site_of) if s == sj]
            site_mat[i, j] = agg[np.ix_(ii, jj)].mean()
    _plot_site_heatmap(site_mat, sites, f"{dataset} — site-level attention ({tag})",
                       os.path.join(FIGURES_DIR, f"{dataset}_attention_sites_{tag}.png"))

    save_summary_json(f"{dataset}_attention_{tag}.json", {
        "dataset": dataset, "tag": tag, "site_of": dict(zip(feature_cols, site_of)),
        "site_attention": {s1: {s2: float(site_mat[i, j])
                                for j, s2 in enumerate(sites)}
                           for i, s1 in enumerate(sites)},
    })
    print(f"  attention figures -> {FIGURES_DIR}/", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    run(args.dataset, args.tag)


if __name__ == "__main__":
    main()
