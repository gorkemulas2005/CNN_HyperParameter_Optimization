"""
utils/trainer.py
----------------
PyTorch training and evaluation loop with early stopping,
cosine annealing learning rate scheduling, and optional
knowledge-guided pruning for hyperparameter optimization trials.
"""

import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR


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
    prune_threshold: float = None,
):
    """
    Train the model with early stopping based on validation accuracy.

    Parameters
    ----------
    model : nn.Module
        The neural network to train.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    optimizer : torch.optim.Optimizer
        Optimizer instance.
    epochs : int
        Maximum number of training epochs (default: 30).
    patience : int
        Number of epochs without improvement before stopping (default: 5).
    device : torch.device, optional
        Computation device; auto-detected if None.
    prune_threshold : float, optional
        If set, trials with validation accuracy at or below this value
        after epoch 1 are pruned (knowledge-guided pruning).

    Returns
    -------
    model : nn.Module
        Model restored to the best validation weights.
    history : dict
        Training history with keys: train_loss, train_acc, val_loss, val_acc.
    best_val_acc : float
        Best validation accuracy achieved during training.
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "train_acc": [],
               "val_loss":   [], "val_acc":   []}

    best_val_acc  = 0.0
    best_weights  = None
    patience_cnt  = 0

    for epoch in range(1, epochs + 1):
        # -- Training phase --------------------------------------------------
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

        # -- Validation phase -------------------------------------------------
        val_loss, val_acc = evaluate_loss_acc(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step()

        print(f"  Epoch {epoch:>3}/{epochs}  "
              f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        # Knowledge-Guided Pruning (Epoch 1 check)
        if prune_threshold is not None and epoch == 1:
            if val_acc <= prune_threshold:
                print(f"  [Knowledge-Guided Pruning] val_acc ({val_acc:.4f}) <= threshold ({prune_threshold:.4f}). Pruning trial at epoch 1!")
                # Stop immediately and return current bad score so optimizer avoids it
                best_val_acc = val_acc
                break

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

    # Restore best model weights
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