from typing import List


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
