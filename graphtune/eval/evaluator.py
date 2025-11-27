# graphtune/eval/evaluator.py (혹은 비슷한 위치)

import torch
import numpy as np
from .metrics import masked_mae_loss, masked_rmse_loss  # 실제 경로에 맞게

def evaluate_model(model, loader, scaler, device="cuda"):
    model.eval()
    maes = []
    rmses = []

    with torch.no_grad():
        for batch in loader:
            # (X, Y, mask) 구조라고 가정
            x, y, mask = batch
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)

            # 1) 모델 예측
            y_hat = model(x)  # [B, T_out, N]

            # 2) 역정규화
            y_hat = scaler.inverse_transform(y_hat)
            y = scaler.inverse_transform(y)

            # 3) NaN/Inf 제거 + 마스크와 AND
            finite_mask = torch.isfinite(y_hat) & torch.isfinite(y)
            final_mask = (mask > 0) & finite_mask

            if final_mask.sum() == 0:
                # 이 배치는 유효 포인트 없음 → 그냥 스킵
                continue

            # 4) 마스킹된 값만 사용해서 MAE/RMSE 계산
            mae = torch.abs(y_hat - y)[final_mask].mean()
            rmse = ((y_hat - y) ** 2)[final_mask].mean().sqrt()

            maes.append(mae.item())
            rmses.append(rmse.item())

    if len(maes) == 0:
        # 전체가 다 비어 있으면 NaN 리턴 (혹은 큰 값 리턴)
        return float("nan"), float("nan")

    return float(np.mean(maes)), float(np.mean(rmses))
