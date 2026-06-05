"""
utils/metrics.py
----------------
IMRAD standartları: Accuracy, Precision, Recall, F1-Score.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)


def evaluate_model(model, test_loader, class_names, device=None):
    """
    Test seti üzerinde tüm metrikleri hesaplar.

    Returns
    -------
    metrics : dict  {accuracy, precision, recall, f1_score}
    report  : str   (per-class classification report)
    cm      : ndarray (confusion matrix)
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
