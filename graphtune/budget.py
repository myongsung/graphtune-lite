from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def make_budgeted_train_loader(
    train_loader: DataLoader,
    frac: float,
    mode: str = "subset",   # "subset" | "steps" | "both"
    seed: int = 42,
) -> Tuple[DataLoader, int]:
    """
    Budgeted loader for few-shot / low-budget tuning.

    frac in (0,1]:
      - subset mode: sample frac * |train_ds|
      - steps mode: keep full dataset but cap steps/epoch to frac * (#batches)
      - both: do both (most budget-strict)

    Returns:
      (new_train_loader, max_batches_per_epoch)
    """
    frac = float(np.clip(frac, 1e-6, 1.0))
    ds = train_loader.dataset
    n = len(ds)

    full_batches = len(train_loader)
    max_batches = max(1, int(round(full_batches * frac)))

    if mode in ["subset", "both"]:
        k = max(1, int(round(n * frac)))
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=k, replace=False)
        sub_ds = Subset(ds, idx)
        new_loader = DataLoader(
            sub_ds,
            batch_size=train_loader.batch_size,
            shuffle=True,
            num_workers=train_loader.num_workers,
            pin_memory=getattr(train_loader, "pin_memory", True),
            drop_last=getattr(train_loader, "drop_last", False),
        )
    else:
        new_loader = train_loader

    return new_loader, max_batches


class LossGradientBudgetScheduler:
    """
    Loss-Gradient Budget Scheduler (lite version).

    We monitor validation RMSE improvement per spent budget fraction.
    If marginal gain rate becomes too small for 'patience' consecutive steps, stop early.

    gain_rate = (prev_val_rmse - new_val_rmse) / delta_frac
    rel_gain  = (prev_val_rmse - new_val_rmse) / prev_val_rmse
    """

    def __init__(
        self,
        fractions: List[float],
        min_gain_rate: float = 0.02,     # absolute RMSE gain per 1.0 budget
        min_rel_improve: float = 0.005,  # 0.5% relative gain
        patience: int = 1,
    ):
        fracs = sorted(set([float(f) for f in fractions if f > 0]))
        if fracs[-1] != 1.0:
            fracs.append(1.0)
        self.fractions = fracs
        self.min_gain_rate = float(min_gain_rate)
        self.min_rel_improve = float(min_rel_improve)
        self.patience = int(patience)
        self._low_gain_count = 0

    def should_continue(
        self,
        prev_val_rmse: float,
        new_val_rmse: float,
        delta_frac: float,
    ) -> bool:
        if delta_frac <= 0:
            return True
        gain = prev_val_rmse - new_val_rmse
        gain_rate = gain / delta_frac
        rel_gain = gain / max(prev_val_rmse, 1e-9)

        ok = (gain_rate >= self.min_gain_rate) or (rel_gain >= self.min_rel_improve)

        if ok:
            self._low_gain_count = 0
            return True
        else:
            self._low_gain_count += 1
            return self._low_gain_count <= self.patience

    def planned_fractions(self) -> List[float]:
        return self.fractions
