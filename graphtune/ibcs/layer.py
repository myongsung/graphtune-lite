# graphtune/ibcs/layer.py
from typing import Callable, Sequence, Optional
import numpy as np

class InfoBudgetCoresetLayer:
    def __init__(
        self,
        budget: int,
        score_fn: Callable[[Sequence[int]], np.ndarray],
        stochastic: bool = False,
    ):
        """
        budget: 몇 개를 남길지 (샘플 수, 토큰 수 등)
        score_fn: 아이템 인덱스 리스트 -> 점수 벡터 반환
        stochastic: 필요하면 확률적 선택(옵션)
        """
        self.budget = budget
        self.score_fn = score_fn
        self.stochastic = stochastic

    def select_indices(self, indices: Sequence[int]) -> np.ndarray:
        scores = self.score_fn(indices)       # shape [len(indices)]
        # 여기서 서브모듈러 greedy / top-k / diversity-aware 선택 로직
        # (처음엔 simple top-k로 시작해도 됨)
        sorted_idx = np.argsort(-scores)      # 내림차순
        chosen = np.array(indices)[sorted_idx[: self.budget]]
        return chosen
