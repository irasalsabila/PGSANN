"""Train PG-SANN on a prepared dataset.

Usage:
    python src/models/train_pg_sann.py --config configs/pg_sann.yaml
                                       [--dataset dataset_a] [--standard-transformer]
"""
import argparse
import os
import sys

# Make `import src` work when running this file directly (no PYTHONPATH needed).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

from src.data.prepare import load_splits
from src.models.trainer import train
from src.utils.config import load_config
from src.utils.logging import OUTPUT_DIR, ExperimentLogger
from src.utils.seeds import set_seed


def locate_indices(feature_cols, target):
    """Return (t_idx, chi_b_idx) or (None, None) if not found."""
    t_idx = None
    chi_b_idx = None
    lower = [c.lower() for c in feature_cols]
    for i, name in enumerate(lower):
        if name in ("t", "gtf") or "tolerance" in name:
            t_idx = i
        # B-site electronegativity: Dataset A has electronegativity_B;
        # Dataset B uses Magpie x_mean (mean electronegativity) as proxy.
        if name in ("electronegativity_b", "x_b", "chi_b", "en_b", "x_mean"):
            chi_b_idx = i
    return t_idx, chi_b_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pg_sann.yaml")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--standard-transformer", action="store_true",
                    help="disable physics losses (standard transformer ablation baseline)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seed", type=int, default=None, help="override config seed")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataset = args.dataset or cfg.data.get("dataset", "dataset_a")
    seed = int(args.seed if args.seed is not None else cfg.data.get("seed", 42))
    set_seed(seed)

    print(f"Loading splits for '{dataset}' ...", flush=True)
    splits = load_splits(dataset)
    feature_cols = splits["feature_cols"]
    print(f"  train={len(splits['y_train'])} val={len(splits['y_val'])} "
          f"test={len(splits['y_test'])} | features={len(feature_cols)}", flush=True)
    t_idx, chi_b_idx = locate_indices(feature_cols, dataset)
    print(f"  physics indices: t_idx={t_idx} chi_b_idx={chi_b_idx}", flush=True)

    if args.epochs is not None:
        cfg.train["epochs"] = args.epochs

    use_physics = not args.standard_transformer
    tag = args.tag or ("standard" if args.standard_transformer else "physics")
    print(f"  mode={'PHYSICS' if use_physics else 'STANDARD (no physics)'}  seed={seed}  "
          f"epochs={cfg.train['epochs']} batch_size={cfg.train.get('batch_size', 64)} "
          f"lr={cfg.train.get('lr', 1e-3)}", flush=True)
    print(f"  device={'mps' if torch.backends.mps.is_available() else 'cpu'}\n", flush=True)

    log = ExperimentLogger(
        name="pg_sann",
        run_dir=os.path.join(OUTPUT_DIR, "models", dataset, tag),
    )
    log.log("dataset", dataset)
    log.log("seed", seed)
    log.log("use_physics", use_physics)
    log.save_config({"data": cfg.data, "model": cfg.model, "train": cfg.train, "loss": cfg.loss})

    model, test_metrics = train(
        cfg, splits, feature_cols,
        use_physics=use_physics,
        t_idx=t_idx, chi_b_idx=chi_b_idx, logger=log,
    )

    out_model = os.path.join(log.run_dir, "best.pt")
    torch.save(model.state_dict(), out_model)
    print(f"\n[{dataset}] test metrics: {test_metrics}")
    print(f"model saved -> {out_model}")
    log.finalize()


if __name__ == "__main__":
    main()
