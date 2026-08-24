"""Train all baseline models on a dataset and save a results summary.

Usage:
    python src/baselines/train_baselines.py --dataset dataset_a [--models rf,catboost]
"""
import argparse
import os
import sys

# Make `import src` work when running this file directly (no PYTHONPATH needed).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from src.baselines.models import MODELS, compute_zero_weights, train_and_evaluate
from src.data.prepare import load_splits
from src.utils.logging import OUTPUT_DIR, ExperimentLogger
from src.utils.seeds import set_seed


def run(dataset: str, models: list = None, weighted: bool = True):
    set_seed(42)
    print(f"Loading splits for '{dataset}' ...", flush=True)
    s = load_splits(dataset)
    X_train, y_train = s["X_train"], s["y_train"]
    X_val, y_val = s["X_val"], s["y_val"]
    X_test, y_test = s["X_test"], s["y_test"]
    print(f"  train={len(y_train)} val={len(y_val)} test={len(y_test)} "
          f"| features={X_train.shape[1]}\n", flush=True)

    if models is None:
        models = sorted(MODELS)
    log = ExperimentLogger(name="baselines", run_dir=os.path.join(OUTPUT_DIR, "baselines", dataset))
    log.log("dataset", dataset)

    results = []
    for name in models:
        print(f"[{dataset}] training {name} ...", flush=True)
        sample_weight = None
        model_label = name
        if weighted and name == "lightgbm" and dataset == "dataset_b":
            # Counter metallic zero-gap bias by upweighting semiconductors.
            sample_weight = compute_zero_weights(y_train)
            model_label = "lightgbm_weighted"
            print(f"  weighted: metal=1, semi=3 -> {int((sample_weight==3).sum())} semi samples", flush=True)
        try:
            res, _ = train_and_evaluate(
                name, X_train, y_train, X_val, y_val, X_test, y_test,
                sample_weight=sample_weight,
            )
        except Exception as e:
            print(f"  {name} FAILED: {e}")
            continue
        row = {"dataset": dataset, "model": model_label}
        for split in ("train", "val", "test"):
            for k, v in res[split].items():
                row[f"{split}_{k}"] = round(float(v), 4)
        results.append(row)
        print(f"  test r2={row['test_r2']:.4f} rmse={row['test_rmse']:.4f} "
              f"mae={row['test_mae']:.4f} nonphys={row['test_nonphys_rate']:.4f}")

    summary_csv = os.path.join(OUTPUT_DIR, "baselines", f"summary_{dataset}.csv")
    for row in results:
        log.append_row(row, summary_csv)
    log.log("n_models", len(results))
    run_dir = log.finalize()
    print(f"[{dataset}] results -> {summary_csv} | run log -> {run_dir}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--models", default=None, help="comma-separated model names")
    args = ap.parse_args()
    models = args.models.split(",") if args.models else None
    run(args.dataset, models)
