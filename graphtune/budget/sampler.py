from typing import Tuple
import numpy as np
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

    NOTE: legacy behavior preserved:
      even in "subset" mode we still return a max_batches cap based on frac.
      (run_experiment passes it through)
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
