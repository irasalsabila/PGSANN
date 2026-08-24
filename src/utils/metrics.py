import numpy as np


def r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def nonphys_rate(y_pred):
    """Fraction of predictions below 0 eV (non-physical)."""
    y_pred = np.asarray(y_pred, dtype=float)
    if y_pred.size == 0:
        return 0.0
    return float(np.mean(y_pred < 0))


def evaluate(y_true, y_pred):
    """Compute the full metric set used across the project."""
    return {
        "r2": r2(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "nonphys_rate": nonphys_rate(y_pred),
    }
