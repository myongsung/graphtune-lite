# graphtune/ibcs/scoring_graph.py
import numpy as np

def make_graph_sample_scorer(A_np, X_train, Y_train, difficulty_weight: float = 1.0):
    """
    (대략적인 예시)
    - 그래프 구조(A_np)와 시계열(X_train, Y_train)을 보고
    - 샘플 인덱스별 난이도/대표성 score를 계산하는 함수 생성
    - 반환된 함수는 indices -> scores 형태를 가짐
    """
    # 예: 노드 degree, 과거 RMSE, gradient norm 등을 미리 precompute 해도 됨

    def score_fn(indices):
        # indices: [i0, i1, ...] (train_ds에서의 샘플 인덱스)
        # 여기서 coverage / diversity / difficulty 결합한 점수를 계산
        scores = np.ones(len(indices), dtype=np.float32)  # placeholder
        return scores

    return score_fn
