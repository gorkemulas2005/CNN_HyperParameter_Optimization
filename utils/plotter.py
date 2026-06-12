"""
utils/plotter.py
----------------
Visualization utilities for training curves, confusion matrices,
and final comparison bar charts (GA vs. Bayesian TPE).
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

PLOT_DIR = "results/plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def plot_history(history: dict, model_name: str, optimizer_name: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name} — {optimizer_name}", fontsize=14)

    axes[0].plot(history["train_acc"], label="Train Acc")
    axes[0].plot(history["val_acc"],   label="Val Acc")
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("Epoch"); axes[0].legend()

    axes[1].plot(history["train_loss"], label="Train Loss")
    axes[1].plot(history["val_loss"],   label="Val Loss")
    axes[1].set_title("Loss"); axes[1].set_xlabel("Epoch"); axes[1].legend()

    fname = f"{PLOT_DIR}/{model_name}_{optimizer_name}_history.png"
    plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()
    print(f"  [Plot saved] {fname}")


def plot_confusion_matrix(cm, class_names, model_name: str, optimizer_name: str):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix — {model_name} ({optimizer_name})")
    plt.xlabel("Predicted"); plt.ylabel("True")
    fname = f"{PLOT_DIR}/{model_name}_{optimizer_name}_cm.png"
    plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()
    print(f"  [Plot saved] {fname}")


def plot_final_comparison(results: dict):
    labels = list(results.keys())
    accs   = [v["accuracy"]  for v in results.values()]
    f1s    = [v["f1_score"]  for v in results.values()]
    times  = [v.get("time_sec", 0) / 60 for v in results.values()]

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("GA vs Bayesian TPE — Performance Comparison", fontsize=15)

    for ax, vals, title, color in zip(
        axes,
        [accs, f1s, times],
        ["Accuracy", "F1-Score (Macro)", "Duration (minutes)"],
        ["#4C72B0", "#DD8452", "#55A868"]
    ):
        bars = ax.bar(x, vals, color=color, alpha=0.85)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, max(vals) * 1.15 + 0.01)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    fname = f"{PLOT_DIR}/final_comparison.png"
    plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()
    print(f"  [Plot saved] {fname}")
