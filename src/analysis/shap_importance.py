"""SHAP feature-importance analysis on the best baseline (LightGBM).

Uses the same splits as PG-SANN for a fair comparison. Produces a SHAP
summary plot + mean |SHAP| bar chart, and verifies B-site electronegativity
ranks highly (PRD physics claim).

Usage:
    python src/analysis/shap_importance.py --dataset dataset_a [--model lightgbm]
"""
import argparse
import os
import sys

# Make `import src` work when running this file directly (no PYTHONPATH needed).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib.pyplot as plt
import numpy as np

from src.analysis.helpers import FIGURES_DIR, ANALYSIS_DIR, ensure_dirs, save_summary_json
from src.data.prepare import load_splits

BASELINE_FACTORY = {
    "lightgbm": lambda **kw: __import__("lightgbm").LGBMRegressor(
        n_estimators=400, learning_rate=0.05, num_leaves=63, random_state=42,
        verbose=-1, **kw),
    "xgboost": lambda **kw: __import__("xgboost").XGBRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=6, random_state=42,
        verbosity=0, **kw),
    "catboost": lambda **kw: __import__("catboost").CatBoostRegressor(
        iterations=400, learning_rate=0.05, depth=6, random_seed=42, verbose=0, **kw),
}


def run(dataset, model_name="lightgbm", n_shap_samples=512):
    ensure_dirs()
    import shap

    print(f"[{dataset}] loading splits ...", flush=True)
    s = load_splits(dataset)
    X_train, y_train = s["X_train"], s["y_train"]
    X_test = s["X_test"]
    feature_cols = s["feature_cols"]
    print(f"  train={len(y_train)} test={len(X_test)} features={len(feature_cols)}", flush=True)

    print(f"[{dataset}] training {model_name} ...", flush=True)
    factory = BASELINE_FACTORY[model_name]
    model = factory()
    model.fit(X_train, y_train)
    print(f"  trained {model_name}", flush=True)

    print(f"[{dataset}] computing SHAP (sample={n_shap_samples}) ...", flush=True)
    explainer = shap.TreeExplainer(model)
    X_sample = X_test[:n_shap_samples]
    shap_values = explainer.shap_values(X_sample)

    # mean |SHAP| per feature, ranked
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    ranked = [(feature_cols[i], float(mean_abs[i])) for i in order]

    # Bar chart top-K
    k = min(20, len(ranked))
    fig, ax = plt.subplots(figsize=(8, 6))
    names = [r[0] for r in ranked[:k]][::-1]
    vals = [r[1] for r in ranked[:k]][::-1]
    ax.barh(names, vals, color="tab:blue")
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title(f"{model_name} feature importance — {dataset}")
    fig.savefig(os.path.join(FIGURES_DIR, f"{dataset}_shap_bar_{model_name}.png"))
    plt.close(fig)

    # SHAP summary plot
    shap.summary_plot(shap_values, X_sample, feature_names=feature_cols,
                      show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"{dataset}_shap_summary_{model_name}.png"))
    plt.close()

    # Verify B-site electronegativity ranks highly
    lower = [c.lower() for c in feature_cols]
    chi_rank = None
    for r, (name, _) in enumerate(ranked):
        if name.lower() in ("electronegativity_b", "x_mean", "x_b", "chi_b"):
            chi_rank = r + 1
            break
    print(f"  B-site electronegativity rank: {chi_rank} of {len(ranked)}", flush=True)
    print(f"  top-5: {ranked[:5]}", flush=True)

    save_summary_json(f"{dataset}_shap_{model_name}.json", {
        "dataset": dataset, "model": model_name,
        "top_features": ranked[:k],
        "chi_b_rank": chi_rank,
    })
    print(f"  SHAP figures -> {FIGURES_DIR}/", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="lightgbm",
                    choices=list(BASELINE_FACTORY.keys()))
    ap.add_argument("--samples", type=int, default=512)
    args = ap.parse_args()
    run(args.dataset, args.model, args.samples)


if __name__ == "__main__":
    main()
