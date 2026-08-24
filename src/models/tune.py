"""Optuna hyperparameter tuning for PG-SANN.

Usage:
    python src/models/tune.py --dataset dataset_a [--trials 30] [--standard-transformer]
"""
import argparse
import os
import sys
import time

# Make `import src` work when running this file directly (no PYTHONPATH needed).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import optuna
import torch
from tqdm import tqdm

from src.data.prepare import load_splits
from src.models.pg_sann import build_model
from src.models.dataset import make_loaders
from src.models.losses import build_loss, CompositeLoss
from src.models.trainer import evaluate_epoch
from src.utils.logging import OUTPUT_DIR
from src.utils.metrics import evaluate
from src.utils.seeds import set_seed

# Show per-trial logs from Optuna (trial # / value / pruning).
optuna.logging.set_verbosity(optuna.logging.INFO)


def objective_factory(splits, feature_cols, device, use_physics, t_idx, chi_b_idx,
                      total_trials=None, progress_callback=None):
    _trial_counter = {"n": 0}

    def objective(trial):
        _trial_counter["n"] += 1
        trial_no = _trial_counter["n"]
        total = total_trials or "?"
        t0 = time.time()
        seed = trial.suggest_categorical("seed", [42])
        set_seed(seed)
        n_features = splits["X_train"].shape[1]

        model_cfg = {
            "d_model": trial.suggest_categorical("d_model", [64, 128]),
            "n_heads": trial.suggest_categorical("n_heads", [2, 4, 8]),
            "n_layers": trial.suggest_categorical("n_layers", [2, 3, 4]),
            "d_ffn": trial.suggest_categorical("d_ffn", [128, 256]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.2),
            "pool": "mean",
            "tokenizer": trial.suggest_categorical("tokenizer", ["periodic", "linear"]),
            "use_reglu": True,
            "layer_scale": 1e-2,
            "deep_head": True,
            "head_width": trial.suggest_categorical("head_width", [128, 256]),
        }
        train_cfg = {
            "batch_size": trial.suggest_categorical("batch_size", [64, 128]),
            "epochs": trial.suggest_categorical("epochs", [150, 300]),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "patience": 20,
            "clip_min": 0.0,
        }
        loss_cfg = {}
        if use_physics:
            loss_cfg = {
                "lam_bounds": trial.suggest_float("lam_bounds", 0.1, 2.0),
                "lam_tolerance": trial.suggest_float("lam_tolerance", 0.0, 1.0),
                "lam_monotonicity": trial.suggest_float("lam_monotonicity", 0.0, 1.0),
            }
        cfg = {"model": model_cfg, "train": train_cfg, "loss": loss_cfg}

        if progress_callback is not None:
            progress_callback(trial_no, total, model_cfg, train_cfg, loss_cfg, use_physics)

        train_loader, val_loader, test_loader = make_loaders(
            splits, t_idx=t_idx, chi_b_idx=chi_b_idx, batch_size=train_cfg["batch_size"]
        )
        model = build_model(n_features, {"model": model_cfg}).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=train_cfg["patience"] // 2
        )

        if use_physics:
            loss_fn = build_loss({"loss": loss_cfg}, n_features, feature_cols)
        else:
            loss_fn = CompositeLoss(0.0, 0.0, 0.0)

        best_val_rmse = float("inf")
        best_state = None
        patience_counter = 0
        epoch_bar = tqdm(range(train_cfg["epochs"]), leave=False, desc=f"trial {trial_no}/{total}",
                         ncols=100)
        for epoch in epoch_bar:
            model.train()
            for batch in train_loader:
                x, y, t, chi_b = batch
                x, y = x.to(device), y.to(device)
                t = t.to(device)
                if use_physics and chi_b_idx is not None:
                    x = x.requires_grad_(True)
                    chi_b = x[:, chi_b_idx].to(device)
                else:
                    chi_b = chi_b.to(device)
                optimizer.zero_grad()
                y_hat = model(x)
                if use_physics:
                    t_use = t if t_idx is not None else None
                    chi_use = chi_b if chi_b_idx is not None else None
                    loss, _ = loss_fn(y, y_hat, t=t_use, chi_b=chi_use)
                else:
                    loss, _ = loss_fn(y, y_hat)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            val_metrics = evaluate_epoch(model, val_loader, loss_fn, device, t_idx, chi_b_idx,
                                         clip_min=train_cfg.get("clip_min"))
            scheduler.step(val_metrics["rmse"])
            trial.report(val_metrics["rmse"], epoch)
            epoch_bar.set_postfix(val_rmse=f"{val_metrics['rmse']:.4f}",
                                  best=f"{best_val_rmse:.4f}")
            if trial.should_prune():
                epoch_bar.close()
                raise optuna.TrialPruned()

            if val_metrics["rmse"] < best_val_rmse:
                best_val_rmse = val_metrics["rmse"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= train_cfg["patience"]:
                    break
        epoch_bar.close()

        model.load_state_dict(best_state)
        test_metrics = evaluate_epoch(model, test_loader, loss_fn, device, t_idx, chi_b_idx,
                                      clip_min=train_cfg.get("clip_min"))
        trial.set_user_attr("test_r2", float(test_metrics["r2"]))
        trial.set_user_attr("test_rmse", float(test_metrics["rmse"]))
        trial.set_user_attr("test_mae", float(test_metrics["mae"]))
        elapsed = time.time() - t0
        print(f"[{trial_no}/{total}] done in {elapsed:.0f}s | val_rmse={best_val_rmse:.4f} "
              f"| test: r2={test_metrics['r2']:.4f} rmse={test_metrics['rmse']:.4f} "
              f"mae={test_metrics['mae']:.4f}", flush=True)
        return best_val_rmse

    return objective


def _trial_banner(trial_no, total, model_cfg, train_cfg, loss_cfg, use_physics):
    mode = "physics" if use_physics else "standard (no physics)"
    print(f"\n{'='*70}", flush=True)
    print(f"  TRIAL {trial_no}/{total}  ({mode})", flush=True)
    print(f"  arch : d_model={model_cfg['d_model']} n_heads={model_cfg['n_heads']} "
          f"n_layers={model_cfg['n_layers']} d_ffn={model_cfg['d_ffn']} "
          f"dropout={model_cfg['dropout']:.3f}", flush=True)
    print(f"  train: bs={train_cfg['batch_size']} lr={train_cfg['lr']:.2e} "
          f"wd={train_cfg['weight_decay']:.1e} epochs={train_cfg['epochs']}", flush=True)
    if loss_cfg:
        print(f"  loss : lam_bounds={loss_cfg['lam_bounds']:.3f} "
              f"lam_tol={loss_cfg['lam_tolerance']:.3f} "
              f"lam_mono={loss_cfg['lam_monotonicity']:.3f}", flush=True)
    print(f"{'='*70}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset_a")
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--standard-transformer", action="store_true",
                    help="disable physics losses (standard transformer ablation)")
    args = ap.parse_args()

    set_seed(42)
    print(f"Loading splits for '{args.dataset}' ...", flush=True)
    splits = load_splits(args.dataset)
    feature_cols = splits["feature_cols"]
    print(f"  loaded: train={len(splits['y_train'])} val={len(splits['y_val'])} "
          f"test={len(splits['y_test'])} | features={len(feature_cols)}", flush=True)
    t_idx, chi_b_idx = None, None
    for i, c in enumerate(feature_cols):
        if c.lower() in ("t", "gtf"):
            t_idx = i
        if c.lower() in ("electronegativity_b", "x_mean", "x_b", "chi_b"):
            chi_b_idx = i
    print(f"  physics feature indices: t_idx={t_idx} chi_b_idx={chi_b_idx}", flush=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    use_physics = not args.standard_transformer
    print(f"  device={device}  mode={'physics' if use_physics else 'STANDARD (no physics)'}  "
          f"trials={args.trials}\n", flush=True)

    tag = f"{'physics' if use_physics else 'standard'}"
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        objective_factory(splits, feature_cols, device, use_physics, t_idx, chi_b_idx,
                          total_trials=args.trials, progress_callback=_trial_banner),
        n_trials=args.trials, show_progress_bar=False,
    )

    out = os.path.join(OUTPUT_DIR, "optuna", f"{args.dataset}_{tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    best = study.best_trial
    result = {
        "dataset": args.dataset,
        "use_physics": use_physics,
        "best_val_rmse": best.value,
        "best_params": best.params,
        "test_r2": best.user_attrs.get("test_r2"),
        "test_rmse": best.user_attrs.get("test_rmse"),
        "test_mae": best.user_attrs.get("test_mae"),
    }
    import json
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{args.dataset}] best val rmse={best.value:.4f}")
    print(f"  params: {best.params}")
    print(f"  test: r2={result['test_r2']:.4f} rmse={result['test_rmse']:.4f} mae={result['test_mae']:.4f}")
    print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
