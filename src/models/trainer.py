import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.models.losses import build_loss
from src.models.pg_sann import build_model
from src.utils.metrics import evaluate


def _to_device(batch, device):
    x, y, t, chi_b = batch
    x = x.to(device)
    y = y.to(device)
    t = t.to(device)
    chi_b = chi_b.to(device)
    return x, y, t, chi_b


def evaluate_epoch(model, loader, loss_fn, device, t_idx=None, chi_b_idx=None,
                   clip_min=None):
    """Compute val/test metrics + composite loss over a loader.

    If clip_min is not None, predictions are clamped to >= clip_min before
    computing metrics (physically, band gaps cannot be negative).
    """
    model.eval()
    all_y, all_pred = [], []
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            x, y, t, chi_b = _to_device(batch, device)
            y_hat = model(x)
            if clip_min is not None:
                y_hat = torch.clamp(y_hat, min=clip_min)
            loss = torch.mean((y - y_hat) ** 2)
            if loss_fn.lam_bounds > 0:
                loss = loss + loss_fn.lam_bounds * loss_fn.bounds(y_hat)
            if loss_fn.lam_tolerance > 0 and t_idx is not None:
                loss = loss + loss_fn.lam_tolerance * loss_fn.tolerance(t, y_hat)
            total_loss += loss.item()
            n_batches += 1
            all_y.append(y.detach().cpu().numpy())
            all_pred.append(y_hat.detach().cpu().numpy())
    y_all = np.concatenate(all_y)
    pred_all = np.concatenate(all_pred)
    metrics = evaluate(y_all, pred_all)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


def train(cfg, splits, feature_cols, device=None, use_physics=True,
          t_idx=None, chi_b_idx=None, logger=None):
    """Train PG-SANN and return (model, best_test_metrics)."""
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    n_features = splits["X_train"].shape[1]
    batch_size = int(train_cfg.get("batch_size", 64))
    epochs = int(train_cfg.get("epochs", 50))
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    patience = int(train_cfg.get("patience", 10))
    clip_min = train_cfg.get("clip_min", None)

    from src.models.dataset import make_loaders

    train_loader, val_loader, test_loader = make_loaders(
        splits, t_idx=t_idx, chi_b_idx=chi_b_idx, batch_size=batch_size
    )

    model = build_model(n_features, cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 2
    )

    if use_physics:
        loss_fn = build_loss(cfg, n_features, feature_cols)
    else:
        loss_fn = _NoPhysicsLoss()

    best_val_rmse = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0

    epoch_bar = tqdm(range(epochs), desc="training", ncols=110, unit="ep")
    for epoch in epoch_bar:
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            x, y, t, chi_b = _to_device(batch, device)
            if use_physics and chi_b_idx is not None:
                # Enable grad on inputs so autograd.grad wrt chi_b works.
                x = x.requires_grad_(True)
                chi_b = x[:, chi_b_idx]
            optimizer.zero_grad()
            y_hat = model(x)
            if use_physics:
                t_use = t if t_idx is not None else None
                chi_use = chi_b if chi_b_idx is not None else None
                loss, _ = loss_fn(y, y_hat, t=t_use, chi_b=chi_use)
            else:
                loss = loss_fn(y, y_hat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        val_metrics = evaluate_epoch(model, val_loader, loss_fn, device, t_idx, chi_b_idx,
                                     clip_min=clip_min)
        scheduler.step(val_metrics["rmse"])
        epoch_bar.set_postfix(train_loss=f"{total_loss/max(len(train_loader),1):.4f}",
                              val_rmse=f"{val_metrics['rmse']:.4f}",
                              val_r2=f"{val_metrics['r2']:.4f}",
                              best=f"{best_val_rmse:.4f}")
        if logger:
            logger.log_dict({
                "epoch": epoch + 1,
                "train_loss": round(total_loss / max(len(train_loader), 1), 4),
                "val_rmse": round(val_metrics["rmse"], 4),
                "val_r2": round(val_metrics["r2"], 4),
            })

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    epoch_bar.close()

    model.load_state_dict(best_state)
    test_metrics = evaluate_epoch(model, test_loader, loss_fn, device, t_idx, chi_b_idx,
                                  clip_min=clip_min)
    if logger:
        logger.log_dict({"best_epoch": best_epoch})
        logger.save_metrics(test_metrics)
    return model, test_metrics


class _NoPhysicsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.lam_bounds = 0.0
        self.lam_tolerance = 0.0
        self.lam_monotonicity = 0.0

    def forward(self, y, y_hat):
        return torch.mean((y - y_hat) ** 2)
