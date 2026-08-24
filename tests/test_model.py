import numpy as np
import torch
import torch.nn as nn

from src.models.losses import CompositeLoss
from src.models.pg_sann import PG_SANN


def test_model_forward_shape():
    model = PG_SANN(n_features=10, d_model=32, n_heads=4, n_layers=2)
    x = torch.randn(8, 10)
    out = model(x)
    assert out.shape == (8,)


def test_model_attention_extraction():
    model = PG_SANN(n_features=6, d_model=16, n_heads=2, n_layers=2)
    x = torch.randn(4, 6)
    out, attn = model(x, return_attention=True)
    assert out.shape == (4,)
    assert len(attn) == 2  # one per layer
    # mean pooling: attention over the n_features tokens (no CLS token)
    assert attn[0].shape[-1] == 6


def test_bounds_loss_positive_predictions_zero():
    loss_fn = CompositeLoss(lam_bounds=1.0)
    y_hat = torch.tensor([0.5, 1.0, 2.0])
    assert loss_fn.bounds(y_hat).item() == 0.0


def test_bounds_loss_negative_predictions_penalized():
    loss_fn = CompositeLoss(lam_bounds=1.0)
    y_hat = torch.tensor([-0.5, 1.0, -2.0])
    val = loss_fn.bounds(y_hat).item()
    assert val > 0.0
    # relu(-y): (0.5) + (0) + (2) = 2.5 ; /3
    assert np.isclose(val, 2.5 / 3)


def test_bounds_loss_linear_gradient_at_zero():
    # Linear hinge has non-zero gradient just below zero (unlike the old quadratic
    # clamp, whose gradient vanished at y=0), which actively prevents negative
    # drift on near-zero metallic gaps.
    loss_fn = CompositeLoss(lam_bounds=1.0)
    y_hat = torch.tensor([-0.001, 1.0, -2.0], requires_grad=True)
    loss_fn.bounds(y_hat).backward()
    assert y_hat.grad is not None
    assert y_hat.grad[0].item() < 0.0


def test_tolerance_loss_stable_zero():
    loss_fn = CompositeLoss(lam_tolerance=1.0)
    t = torch.tensor([0.925, 0.925, 0.925])  # ideal -> no penalty
    y_hat = torch.ones(3)
    assert loss_fn.tolerance(t, y_hat).item() == 0.0


def test_tolerance_loss_unstable_penalized():
    loss_fn = CompositeLoss(lam_tolerance=1.0)
    t = torch.tensor([0.6, 1.3, 0.925])  # two unstable
    y_hat = torch.ones(3)
    val = loss_fn.tolerance(t, y_hat).item()
    assert val > 0.0


def test_monotonicity_loss_zero_when_decreasing():
    """If Eg decreases with chi_b (physical), penalty should be ~0."""
    loss_fn = CompositeLoss(lam_monotonicity=1.0)
    chi_b = torch.randn(4, requires_grad=True)
    # y_hat = -2*chi_b  -> gradient w.r.t. chi_b is -2 (negative) -> no penalty
    y_hat = -2.0 * chi_b.sum()
    val = loss_fn.monotonicity(y_hat, chi_b).item()
    assert val == 0.0


def test_monotonicity_loss_penalizes_positive_gradient():
    loss_fn = CompositeLoss(lam_monotonicity=1.0)
    chi_b = torch.randn(4, requires_grad=True)
    y_hat = 3.0 * chi_b.sum()  # positive gradient -> penalized
    val = loss_fn.monotonicity(y_hat, chi_b).item()
    assert val > 0.0


def test_composite_loss_backward():
    """Full composite loss must backprop without autograd errors."""
    torch.manual_seed(0)
    model = PG_SANN(n_features=8, d_model=32, n_heads=2, n_layers=2)
    loss_fn = CompositeLoss(lam_bounds=0.1, lam_tolerance=0.1, lam_monotonicity=0.1)
    x = torch.randn(16, 8, requires_grad=True)
    t = torch.rand(16)
    y = torch.rand(16)
    y_hat = model(x)
    chi_b = x[:, 0]  # use first feature as B-site electronegativity proxy
    loss, terms = loss_fn(y, y_hat, t=t, chi_b=chi_b)
    loss.backward()
    assert loss.item() > 0
    assert all(k in terms for k in ("mse", "bounds", "tolerance", "monotonicity"))
    assert all(g is not None and torch.isfinite(g).all()
               for g in model.parameters())


def test_ablation_by_config_zero_weights():
    """Setting all lambda to 0 -> pure MSE (standard transformer variant)."""
    loss_fn = CompositeLoss(lam_bounds=0.0, lam_tolerance=0.0, lam_monotonicity=0.0)
    y = torch.tensor([1.0, 2.0])
    y_hat = torch.tensor([1.5, 2.5])
    loss, terms = loss_fn(y, y_hat, t=torch.tensor([1.0, 1.0]), chi_b=None)
    assert loss.item() == loss_fn.mse(y, y_hat).item()
    assert set(terms) == {"mse"}


def test_no_physics_training_path_backward():
    """Regression: the no-physics (ablation) training path must unpack the
    (loss, terms) tuple and backprop. Previously this crashed with
    'tuple' object has no attribute 'backward' in tune.py."""
    torch.manual_seed(0)
    model = PG_SANN(n_features=8, d_model=32, n_heads=2, n_layers=2)
    # CompositeLoss with all lambdas 0 == the no-physics loss used in tune.py.
    loss_fn = CompositeLoss(lam_bounds=0.0, lam_tolerance=0.0, lam_monotonicity=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(16, 8)
    y = torch.randn(16)
    w0 = model.head[-1].weight.detach().clone()  # snapshot BEFORE stepping
    for _ in range(2):
        opt.zero_grad()
        y_hat = model(x)
        loss, _ = loss_fn(y, y_hat)  # must unpack like the training loop
        loss.backward()
        opt.step()
        assert loss.item() > 0
    # optimizer step actually changed weights
    assert not torch.equal(w0, model.head[-1].weight.detach())
