"""
utils/trainer.py
----------------
PyTorch eğitim ve değerlendirme döngüsü.
"""

import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    epochs: int = 30,
    patience: int = 5,
    device=None,
):
    """
    Modeli eğitir, val_accuracy'yi izler, early stopping uygular.

    Returns
    -------
    model     : en iyi ağırlıklara restore edilmiş model
    history   : dict  {train_loss, train_acc, val_loss, val_acc}
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                  patience=2)

    history = {"train_loss": [], "train_acc": [],
               "val_loss":   [], "val_acc":   []}

    best_val_acc  = 0.0
    best_weights  = None
    patience_cnt  = 0

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out  = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)
            correct      += (out.argmax(1) == y).sum().item()
            total        += X.size(0)

        train_loss = running_loss / total
        train_acc  = correct / total

        # ── Val ────────────────────────────────────────────────────────────
        val_loss, val_acc = evaluate_loss_acc(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step(val_acc)

        print(f"  Epoch {epoch:>3}/{epochs}  "
              f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        # Early stopping
        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  Early stopping (epoch {epoch})")
                break

    # En iyi ağırlıkları geri yükle
    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, history, best_val_acc


def evaluate_loss_acc(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out   = model(X)
            loss  = criterion(out, y)
            loss_sum += loss.item() * X.size(0)
            correct  += (out.argmax(1) == y).sum().item()
            total    += X.size(0)
    return loss_sum / total, correct / total