"""
utils/metrics.py
----------------
Evaluation metrics for multi-class classification performance.

Computes standard metrics following IMRAD reporting conventions:
Accuracy, Precision (macro), Recall (macro), and F1-Score (macro),
along with a per-class classification report and confusion matrix.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)


def evaluate_model(model, test_loader, class_names, device=None):
    """
    Evaluate the trained model on the test set and compute all metrics.

    Parameters
    ----------
    model : nn.Module
        Trained model to evaluate.
    test_loader : DataLoader
        Test data loader.
    class_names : list of str
        Human-readable class names for the classification report.
    device : torch.device, optional
        Computation device; auto-detected if None.

    Returns
    -------
    metrics : dict
        Dictionary with keys: accuracy, precision, recall, f1_score.
    report : str
        Per-class classification report (sklearn format).
    cm : ndarray
        Confusion matrix of shape (num_classes, num_classes).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            out = model(X)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)

    metrics = {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_score":  round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
    }
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    cm     = confusion_matrix(y_true, y_pred)

    return metrics, report, cm
