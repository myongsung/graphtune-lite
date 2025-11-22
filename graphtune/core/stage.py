# graphtune/core/stage.py
import time
import torch

from ..utils import masked_mae_loss
from ..eval import evaluate_model


def get_teacher_forcing_ratio(epoch, tf_start=1.0, tf_end=0.0, decay_epochs=50):
    """
    legacy에서 쓰던 DCRNN teacher forcing 스케줄 유지. :contentReference[oaicite:1]{index=1}
    """
    if epoch >= decay_epochs:
        return tf_end
    alpha = (epoch - 1) / (decay_epochs - 1)
    return tf_start * (1 - alpha) + tf_end * alpha


def train_one_stage(
    model,
    train_loader,
    val_loader,
    scaler,
    num_epochs=50,
    lr=1e-3,
    max_grad_norm=5.0,
    device="cuda",
    name="stage",
    is_dcrnn=False,
    profile=True,
    max_batches_per_epoch=None,  # budget/steps cap 지원
):
    """
    Train for one stage (or mini-stage).
    Returns (model, stats_dict).
    동작/로그/리턴 형식 legacy와 동일 유지. :contentReference[oaicite:2]{index=2}
    """
    model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(trainable_params, lr=lr)

    best_rmse = float("inf")
    best_state = None

    if profile and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for epoch in range(1, num_epochs + 1):
        model.train()
        run_loss, n_batch = 0.0, 0

        tf_ratio = get_teacher_forcing_ratio(epoch) if is_dcrnn else 0.0

        for bidx, (x, y, mask) in enumerate(train_loader):
            if max_batches_per_epoch is not None and bidx >= max_batches_per_epoch:
                break

            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optim.zero_grad()

            out = model(x, y=y, teacher_forcing_ratio=tf_ratio) if is_dcrnn else model(x)
            pred = out[0] if isinstance(out, tuple) else out

            loss = masked_mae_loss(pred, y, mask)
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optim.step()

            run_loss += loss.item()
            n_batch += 1

        train_loss = run_loss / max(n_batch, 1)
        val_mae, val_rmse = evaluate_model(model, val_loader, scaler, device=device)

        print(
            f"[{name}][{epoch:03d}] "
            f"train_loss={train_loss:.4f} val_MAE={val_mae:.4f} val_RMSE={val_rmse:.4f}"
        )

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    stats = {}
    if profile:
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            peak_mem = 0.0

        dt = time.perf_counter() - t0
        stats = {
            "train_time_sec": float(dt),
            "peak_mem_mb": float(peak_mem),
            "epochs": int(num_epochs),
            "lr": float(lr),
            "max_batches_per_epoch": None if max_batches_per_epoch is None else int(max_batches_per_epoch),
        }

    return model, stats
