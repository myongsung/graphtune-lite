# graphtune/eval/evaluator.py
import numpy as np
import torch

def evaluate_model(model, loader, scaler, device="cuda", is_bigst=False):
    """
    model(x) -> pred 얻고, scaler로 denorm 후 masked MAE/RMSE 계산.
    torch 텐서 기반으로 NaN/Inf 를 안전하게 처리한다.
    """
    model.eval()
    maes = []
    rmses = []

    with torch.no_grad():
        for x, y, mask in loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)

            # 1) 모델 예측
            pred = model(x)  # [B, T_out, N]

            # 2) 역정규화 (scaler 가 torch 텐서를 지원하도록 수정해 둔 상태)
            pred_denorm = scaler.inverse_transform(pred)
            y_denorm = scaler.inverse_transform(y)

            # 3) 차이 + 유효한 위치 마스크
            diff = pred_denorm - y_denorm

            # 유한한 값만 사용 (NaN/Inf 제거)
            finite = torch.isfinite(diff) & torch.isfinite(y_denorm) & torch.isfinite(pred_denorm)

            # 원래 마스크(mask>0) 와 AND
            m = (mask > 0) & finite

            if m.sum() == 0:
                # 이 배치는 유효 포인트 없음 → 스킵
                continue

            # 4) MAE / RMSE 계산
            mae = torch.abs(diff[m]).mean()
            rmse = torch.sqrt((diff[m] ** 2).mean())

            maes.append(mae.item())
            rmses.append(rmse.item())

    if not maes:
        # 전체가 다 비어 있으면 NaN 리턴
        return float("nan"), float("nan")

    return float(np.mean(maes)), float(np.mean(rmses))
