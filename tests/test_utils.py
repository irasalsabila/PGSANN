import numpy as np
import pytest

from src.utils.metrics import evaluate, mae, nonphys_rate, r2, rmse
from src.utils.seeds import set_seed


def test_metrics_perfect_prediction():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    m = evaluate(y, y)
    assert m["r2"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["mae"] == pytest.approx(0.0)
    assert m["nonphys_rate"] == pytest.approx(0.0)


def test_metrics_known_values():
    y = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.0, 2.0, 2.0])
    assert mae(y, pred) == pytest.approx(1 / 3)
    assert rmse(y, pred) == pytest.approx(np.sqrt(1 / 3))


def test_nonphys_rate():
    pred = np.array([-0.5, 0.0, 1.0, 2.0])
    assert nonphys_rate(pred) == pytest.approx(0.25)


def test_r2_constant_target_is_nan():
    y = np.array([1.0, 1.0, 1.0])
    assert np.isnan(r2(y, y))


def test_seed_reproducibility():
    set_seed(42)
    a = np.random.rand(5)
    set_seed(42)
    b = np.random.rand(5)
    assert np.array_equal(a, b)
