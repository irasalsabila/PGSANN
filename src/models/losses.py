"""Physics-informed loss functions for PG-SANN.

L_total = L_MSE + lam1*L_bounds + lam2*L_tolerance + lam3*L_monotonicity
"""
import torch
import torch.nn as nn


class CompositeLoss(nn.Module):
    """Composite physics-informed loss.

    Args:
        lam_bounds (float): weight for non-negative prediction penalty.
        lam_tolerance (float): weight for crystallographic (tolerance factor) penalty.
        lam_monotonicity (float): weight for B-site electronegativity monotonicity.
        t_idx (int): column index of tolerance factor `t` in the input features.
        chi_b_idx (int): column index of B-site electronegativity in the input features.
    """

    def __init__(self, lam_bounds=0.1, lam_tolerance=0.1, lam_monotonicity=0.1,
                 t_idx=None, chi_b_idx=None):
        super().__init__()
        self.lam_bounds = lam_bounds
        self.lam_tolerance = lam_tolerance
        self.lam_monotonicity = lam_monotonicity
        self.t_idx = t_idx
        self.chi_b_idx = chi_b_idx

    def mse(self, y, y_hat):
        return torch.mean((y - y_hat) ** 2)

    def bounds(self, y_hat):
        """Linear hinge penalty on negative predictions (non-physical band gaps).

        relu(-y_hat) has constant (non-zero) gradient for y_hat < 0 and zero
        gradient for y_hat > 0, so it actively pushes predictions above zero
        without biasing positive predictions (critical for the zero-gap
        metallic skew in Datasets B/D, where the quadratic clamp version had
        vanishing gradient at y_hat ~ 0).
        """
        return torch.mean(torch.relu(-y_hat))

    def tolerance(self, t, y_hat):
        """Crystallographic instability penalty.

        Stable perovskites: 0.8 <= t <= 1.05 (we use 0.925 +/- 0.175).
        Penalize |t - 0.925| - 0.175 beyond 0, scaled by sigmoid(y_hat).
        """
        dist = torch.clamp(torch.abs(t - 0.925) - 0.175, min=0.0)
        return torch.mean(dist * torch.sigmoid(y_hat))

    def monotonicity(self, y_hat, chi_b):
        """Penalize positive d(y_hat)/d(chi_b). Higher B-site electronegativity
        should depress Eg (decrease), so positive gradient is non-physical."""
        if chi_b is None or chi_b.numel() == 0:
            return y_hat.sum() * 0.0  # zero contribution if feature absent
        grads = torch.autograd.grad(
            outputs=y_hat, inputs=chi_b,
            grad_outputs=torch.ones_like(y_hat),
            create_graph=True, retain_graph=True, allow_unused=True,
        )[0]
        if grads is None:
            return y_hat.sum() * 0.0
        return torch.mean(torch.clamp(grads, min=0.0) ** 2)

    def forward(self, y, y_hat, t=None, chi_b=None):
        loss = self.mse(y, y_hat)
        terms = {"mse": loss.item()}
        if self.lam_bounds > 0:
            lb = self.bounds(y_hat)
            loss = loss + self.lam_bounds * lb
            terms["bounds"] = lb.item()
        if self.lam_tolerance > 0 and t is not None:
            lt = self.tolerance(t, y_hat)
            loss = loss + self.lam_tolerance * lt
            terms["tolerance"] = lt.item()
        if self.lam_monotonicity > 0 and chi_b is not None:
            lm = self.monotonicity(y_hat, chi_b)
            loss = loss + self.lam_monotonicity * lm
            terms["monotonicity"] = lm.item()
        return loss, terms


def build_loss(cfg, n_features: int, feature_cols) -> CompositeLoss:
    loss_cfg = cfg.get("loss", {})
    t_idx = None
    chi_b_idx = None
    # Locate tolerance-factor and B-site electronegativity columns by name.
    for i, name in enumerate(feature_cols):
        if name.lower() in ("t", "gtf") or "tolerance" in name.lower():
            t_idx = i
        if name.lower() in ("electronegativity_b", "x_b", "chi_b", "x_mean"):
            chi_b_idx = i
    return CompositeLoss(
        lam_bounds=float(loss_cfg.get("lam_bounds", 0.1)),
        lam_tolerance=float(loss_cfg.get("lam_tolerance", 0.1)),
        lam_monotonicity=float(loss_cfg.get("lam_monotonicity", 0.1)),
        t_idx=t_idx,
        chi_b_idx=chi_b_idx,
    )
