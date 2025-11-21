import torch
from .utils import masked_mae_loss
from .eval import evaluate_model

def get_teacher_forcing_ratio(epoch, tf_start=1.0, tf_end=0.0, decay_epochs=50):
    if epoch >= decay_epochs:
        return tf_end
    alpha = (epoch - 1) / (decay_epochs - 1)
    return tf_start*(1-alpha) + tf_end*alpha

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
):
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    best_rmse = float("inf")
    best_state = None

    for epoch in range(1, num_epochs+1):
        model.train()
        run_loss, n_batch = 0.0, 0

        tf_ratio = get_teacher_forcing_ratio(epoch) if is_dcrnn else 0.0

        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optim.zero_grad()

            out = model(x, y=y, teacher_forcing_ratio=tf_ratio) if is_dcrnn else model(x)
            pred = out[0] if isinstance(out, tuple) else out

            loss = masked_mae_loss(pred, y, mask)
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optim.step()

            run_loss += loss.item(); n_batch += 1

        train_loss = run_loss / max(n_batch, 1)
        val_mae, val_rmse = evaluate_model(model, val_loader, scaler, device=device)

        print(f"[{name}][{epoch:03d}] train_loss={train_loss:.4f} val_MAE={val_mae:.4f} val_RMSE={val_rmse:.4f}")

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model
