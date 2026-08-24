"""Compare PG-SANN v2 architecture variants on a dataset (diagnostic).

Run from repo root:
    python src/models/compare_v2.py --dataset dataset_a

Trains a few PG-SANN v2 configs (pure MSE) and a no-attention reference to see
whether the improved architecture is competitive with baselines. Use fewer
epochs with --epochs to iterate faster.
"""
import argparse
import time

import numpy as np
import torch
from tqdm import tqdm

from src.data.prepare import load_splits
from src.models.dataset import make_loaders
from src.models.pg_sann import PG_SANN
from src.utils.metrics import mae, r2, rmse
from src.utils.seeds import set_seed


def evaluate_splits(model, splits, device):
    model.eval()
    Xtr = torch.tensor(splits["X_train"], dtype=torch.float32).to(device)
    Xte = torch.tensor(splits["X_test"], dtype=torch.float32).to(device)
    with torch.no_grad():
        p = model(Xtr).cpu().numpy()
        pt = model(Xte).cpu().numpy()
    return {
        "train_r2": r2(splits["y_train"], p),
        "test_r2": r2(splits["y_test"], pt),
        "test_rmse": rmse(splits["y_test"], pt),
        "test_mae": mae(splits["y_test"], pt),
    }


def train_pure_mse(model, splits, device, epochs, lr, batch_size=64):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    mse = torch.nn.MSELoss()
    tl, _, _ = make_loaders(splits, batch_size=batch_size)
    model.train()
    for ep in tqdm(range(epochs), desc="training", ncols=90, leave=True):
        for x, y, _, _ in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = mse(model(x), y)
            loss.backward()
            opt.step()
        if ep == 0:
            # quick sanity print so the user knows training started
            print(f"  [start] ep0 loss={loss.item():.4f}", flush=True)
    return model


def run(dataset, epochs, lr):
    set_seed(42)
    splits = load_splits(dataset)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    nf = splits["X_train"].shape[1]
    print(f"[{dataset}] features={nf} | epochs={epochs} lr={lr} | device={dev}\n", flush=True)

    configs = {
        "v2 periodic+deephead (2L)": dict(tokenizer="periodic", deep_head=True, n_layers=2),
        "v2 linear+deephead (2L)": dict(tokenizer="linear", deep_head=True, n_layers=2),
        "v2 periodic no-deephead (2L)": dict(tokenizer="periodic", deep_head=False, n_layers=2),
        "v2 periodic+deephead (4L)": dict(tokenizer="periodic", deep_head=True, n_layers=4),
    }

    results = []
    for i, (name, kw) in enumerate(configs.items()):
        print(f"\n=== [{i+1}/{len(configs)}] {name} ===", flush=True)
        t0 = time.time()
        model = PG_SANN(n_features=nf, d_model=64, n_heads=4, d_ffn=128,
                        dropout=0.1, pool="mean", use_reglu=True, layer_scale=1e-2,
                        head_width=256, **kw)
        model = train_pure_mse(model, splits, dev, epochs, lr)
        m = evaluate_splits(model, splits, dev)
        m.update({"name": name, "secs": int(time.time() - t0)})
        results.append(m)
        print(f"{name:32s} train_r2={m['train_r2']:.3f} test_r2={m['test_r2']:.3f} "
              f"rmse={m['test_rmse']:.3f} mae={m['test_mae']:.3f} ({m['secs']}s)", flush=True)

    print("\n=== reference baselines (test) ===")
    print(f"{'XGBoost (best on A)':32s} test_r2=0.970 rmse=0.238")
    print(f"{'MLP baseline':32s} test_r2=0.932 rmse=0.357")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset_a")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()
    run(args.dataset, args.epochs, args.lr)


if __name__ == "__main__":
    main()
