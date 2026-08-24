import numpy as np
import torch
from torch.utils.data import Dataset


class PerovskiteDataset(Dataset):
    """PyTorch dataset wrapping cached numpy splits.

    Returns (x, y, t, chi_b) where t is the tolerance factor column and
    chi_b the B-site electronegativity column (used by physics losses).
    """

    def __init__(self, X, y, t_idx=None, chi_b_idx=None):
        self.X = torch.tensor(np.asarray(X, dtype=np.float32))
        self.y = torch.tensor(np.asarray(y, dtype=np.float32))
        self.t_idx = t_idx
        self.chi_b_idx = chi_b_idx

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        t = x[self.t_idx] if self.t_idx is not None else torch.tensor(float("nan"))
        chi_b = x[self.chi_b_idx] if self.chi_b_idx is not None else torch.tensor(float("nan"))
        return x, self.y[idx], t, chi_b


def make_loaders(splits, t_idx=None, chi_b_idx=None, batch_size=64, seed=42):
    from torch.utils.data import DataLoader

    from src.utils.seeds import seed_worker

    g = torch.Generator()
    g.manual_seed(seed)
    train_ds = PerovskiteDataset(splits["X_train"], splits["y_train"], t_idx, chi_b_idx)
    val_ds = PerovskiteDataset(splits["X_val"], splits["y_val"], t_idx, chi_b_idx)
    test_ds = PerovskiteDataset(splits["X_test"], splits["y_test"], t_idx, chi_b_idx)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker,
        generator=g, drop_last=False,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
