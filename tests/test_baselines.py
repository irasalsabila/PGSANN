import numpy as np
import pytest

from src.baselines.models import compute_zero_weights, get_model, train_and_evaluate
from src.data.prepare import load_splits


def test_zero_weights_ratio():
    y = np.array([0.0, 1.0, 2.0, 0.0])
    w = compute_zero_weights(y, metal_weight=1.0, semi_weight=3.0)
    assert w[0] == 1.0
    assert w[1] == 3.0
    assert w[2] == 3.0
    assert w[3] == 1.0


def test_get_model_available():
    for name in ["random_forest", "svr", "xgboost", "lightgbm", "catboost", "mlp"]:
        model = get_model(name)
        assert model is not None


def test_train_evaluate_smoke():
    s = load_splits("dataset_a")
    res, model = train_and_evaluate(
        "random_forest", s["X_train"], s["y_train"],
        s["X_val"], s["y_val"], s["X_test"], s["y_test"],
        n_estimators=50,
    )
    assert 0.0 <= res["test"]["r2"] <= 1.0
    assert "rmse" in res["test"] and "mae" in res["test"]
