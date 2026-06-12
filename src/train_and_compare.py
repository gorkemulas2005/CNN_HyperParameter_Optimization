"""
train_and_compare.py -- DAGM 2007 Texture Defect Classification
=================================================================
Main execution script for CNN hyperparameter optimization.

Compares three optimization strategies:
  1. Genetic Algorithm (GA)
  2. Bayesian TPE (Tree-structured Parzen Estimator)
  3. Repulsive Hybrid (GA-guided Bayesian with MAP-Elites pruning)

Across four CNN architectures:
  - VGG16 (transfer learning)
  - ResNet50 (transfer learning)
  - Custom CNN (depthwise separable convolutions + SE attention)
  - Proposed CNN (5-block architecture, Acici et al. 2020 inspired)

Dataset: DAGM 2007 -- 10 texture defect classes
Hardware: NVIDIA RTX 4050 Laptop GPU (6 GB VRAM)

Reference:
  Acici, K., Beyaz, S., Sumer, E. (2020). Femoral neck fracture detection
  in X-ray images using deep learning and genetic algorithm approaches.
  Joint Diseases and Related Surgery, 31(2), 175-183.
"""

import os
import sys
import time
import json
import csv
import warnings
warnings.filterwarnings("ignore")

import torch

# ---------------------------------------------------------------------------
# Experiment Configuration
# ---------------------------------------------------------------------------
CONFIG = {
    # Genetic Algorithm parameters
    "ga_generations":         8,
    "ga_pop_size":           10,
    "ga_epochs_per_eval":     5,

    # Bayesian TPE parameters
    "bayes_n_trials":        20,
    "bayes_epochs_per_trial":  5,

    # Final training parameters
    "final_epochs":          15,
    "final_batch_size":      32,
    "final_patience":         5,

    # Models and optimizers to evaluate
    "models_to_run":     ["vgg16", "resnet50", "custom", "custom_v2"],
    "optimizers_to_run": ["ga", "bayesian", "repulsive_hybrid"],

    # Dataset parameters
    "img_size":     224,
    "num_classes":   10,    # DAGM 2007: 10 classes
    "num_workers":    0,    # Set to 0 for Windows compatibility
}


def setup_device():
    """Initialize and return the computation device (GPU or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        vram = props.total_memory / 1024**3
        print(f"  Device : {props.name}")
        print(f"  VRAM   : {vram:.1f} GB")
        torch.backends.cudnn.benchmark = True
        return device
    print("  Device : CPU (no GPU detected)")
    return torch.device("cpu")


# Create output directories
for d in ["results/ga", "results/bayesian", "results/repulsive_hybrid",
          "results/plots", "results/models", "logs"]:
    os.makedirs(d, exist_ok=True)

from utils.data_loader import get_loaders, NUM_CLASSES
from utils.trainer     import train_model
from utils.metrics     import evaluate_model
from utils.plotter     import (plot_history, plot_confusion_matrix,
                                plot_final_comparison)
from optimization.genetic_algorithm import run_ga
from optimization.bayesian_tpe      import run_bayesian
from optimization.repulsive_bayesian import run_repulsive_bayesian
from models.vgg16_model      import build_vgg16,      get_optimizer as opt_vgg
from models.resnet50_model   import build_resnet50,   get_optimizer as opt_res
from models.custom_cnn_model import build_custom_cnn, get_optimizer as opt_cus
from models.custom_cnn_v2_model import build_custom_cnn_v2, get_optimizer as opt_cus_v2


def build_final_model(model_name, params):
    """Construct model and optimizer from the best hyperparameters found."""
    lr      = params.get("learning_rate", 1e-4)
    dropout = params.get("dropout_rate",  0.4)
    dense   = params.get("dense_units",   256)
    opt_n   = params.get("optimizer",     "adam")
    nc      = CONFIG["num_classes"]

    if model_name == "vgg16":
        model = build_vgg16(num_classes=nc, dropout_rate=dropout, dense_units=dense)
        optim = opt_vgg(model, opt_n, lr)
    elif model_name == "resnet50":
        model = build_resnet50(num_classes=nc, dropout_rate=dropout, dense_units=dense)
        optim = opt_res(model, opt_n, lr)
    elif model_name == "custom":
        model = build_custom_cnn(
            num_classes=nc,
            base_filters=params.get("base_filters", 32),
            num_blocks=params.get("num_blocks", 3),
            kernel_size=params.get("kernel_size", 3),
            dropout_rate=dropout,
        )
        optim = opt_cus(model, opt_n, lr)
    else:  # custom_v2
        model = build_custom_cnn_v2(
            num_classes=nc,
            base_filters=params.get("base_filters", 32),
            kernel_size=params.get("kernel_size", 3),
            dropout_rate=dropout,
        )
        optim = opt_cus_v2(model, opt_n, lr)
    return model, optim


def main():
    print("\n" + "=" * 65)
    print("  DAGM 2007 -- CNN Hyperparameter Optimization")
    print("  GA vs Bayesian TPE vs Repulsive Hybrid  |  10 Classes")
    print("=" * 65)

    device = setup_device()

    train_loader, val_loader, test_loader, class_names = get_loaders(
        img_size=CONFIG["img_size"],
        batch_size=CONFIG["final_batch_size"],
        num_workers=CONFIG["num_workers"],
    )
    print(f"\n  Classes : {class_names}")
    print(f"  Train: {len(train_loader.dataset)} | "
          f"Val: {len(val_loader.dataset)} | "
          f"Test: {len(test_loader.dataset)}\n")

    best_params_all = {}
    checkpoint_file = "results/best_params_checkpoint.json"
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            try:
                best_params_all = json.load(f)
            except json.JSONDecodeError:
                pass
    all_results = {}

    # -- Phase 1: Hyperparameter Optimization --------------------------------
    for model_name in CONFIG["models_to_run"]:
        if model_name not in best_params_all:
            best_params_all[model_name] = {}

        if "ga" in CONFIG["optimizers_to_run"]:
            if "ga" not in best_params_all[model_name]:
                params, _, val_acc, t_sec, _ = run_ga(
                    model_name=model_name,
                    n_generations=CONFIG["ga_generations"],
                    pop_size=CONFIG["ga_pop_size"],
                    epochs_per_eval=CONFIG["ga_epochs_per_eval"],
                    num_classes=CONFIG["num_classes"],
                )
                best_params_all[model_name]["ga"] = {
                    "params": params, "val_acc": val_acc, "time_sec": t_sec
                }
                with open(checkpoint_file, "w") as f:
                    json.dump(best_params_all, f, indent=2)

        if "bayesian" in CONFIG["optimizers_to_run"]:
            if "bayesian" not in best_params_all[model_name]:
                params, val_acc, t_sec, _ = run_bayesian(
                    model_name=model_name,
                    n_trials=CONFIG["bayes_n_trials"],
                    epochs_per_trial=CONFIG["bayes_epochs_per_trial"],
                    num_classes=CONFIG["num_classes"],
                )
                best_params_all[model_name]["bayesian"] = {
                    "params": params, "val_acc": val_acc, "time_sec": t_sec
                }
                with open(checkpoint_file, "w") as f:
                    json.dump(best_params_all, f, indent=2)

        if "repulsive_hybrid" in CONFIG["optimizers_to_run"]:
            if "repulsive_hybrid" not in best_params_all[model_name]:
                params, val_acc, t_sec, _ = run_repulsive_bayesian(
                    model_name=model_name,
                    n_trials=CONFIG["bayes_n_trials"],
                    epochs_per_trial=CONFIG["bayes_epochs_per_trial"],
                    num_classes=CONFIG["num_classes"],
                )
                best_params_all[model_name]["repulsive_hybrid"] = {
                    "params": params, "val_acc": val_acc, "time_sec": t_sec
                }
                with open(checkpoint_file, "w") as f:
                    json.dump(best_params_all, f, indent=2)

    with open("results/best_params_all.json", "w") as f:
        json.dump(best_params_all, f, indent=2)

    # -- Phase 2: Final Training with Best Parameters ------------------------
    print("\n" + "-" * 65)
    print("  Final Training Phase")
    print("-" * 65)

    for model_name in CONFIG["models_to_run"]:
        for opt_name in CONFIG["optimizers_to_run"]:
            if opt_name not in best_params_all.get(model_name, {}):
                continue

            tag      = f"{model_name.upper()}_{opt_name.upper()}"
            params   = best_params_all[model_name][opt_name]["params"]
            opt_time = best_params_all[model_name][opt_name]["time_sec"]

            print(f"\n  [{tag}] Final training with optimized parameters...")

            batch = params.get("batch_size", CONFIG["final_batch_size"])
            tr_l, va_l, te_l, cnames = get_loaders(
                img_size=CONFIG["img_size"],
                batch_size=batch,
                num_workers=CONFIG["num_workers"],
            )

            model, optim = build_final_model(model_name, params)

            t0 = time.time()
            model, history, _ = train_model(
                model, tr_l, va_l, optim,
                epochs=CONFIG["final_epochs"],
                patience=CONFIG["final_patience"],
                device=device,
            )
            train_time = time.time() - t0

            torch.save(model.state_dict(), f"results/models/{tag}_best.pt")
            plot_history(history, model_name.upper(), opt_name.upper())

            metrics, report, cm = evaluate_model(model, te_l, cnames, device)
            plot_confusion_matrix(cm, cnames, model_name.upper(), opt_name.upper())

            print(f"\n  [{tag}] Test Results:")
            print(f"    Accuracy  : {metrics['accuracy']:.4f}")
            print(f"    Precision : {metrics['precision']:.4f}")
            print(f"    Recall    : {metrics['recall']:.4f}")
            print(f"    F1-Score  : {metrics['f1_score']:.4f}")
            print(f"\n{report}")

            rpath = f"results/{opt_name}/{model_name}_final_report.txt"
            with open(rpath, "w", encoding="utf-8") as rf:
                rf.write(f"Model    : {tag}\n")
                rf.write(f"Dataset  : DAGM 2007 (10 classes)\n")
                rf.write(f"Params   : {json.dumps(params, indent=2)}\n\n")
                rf.write(f"Optimization Time : {opt_time/60:.1f} min\n")
                rf.write(f"Training Time     : {train_time/60:.1f} min\n\n")
                rf.write("Test Metrics:\n")
                for k, v in metrics.items():
                    rf.write(f"  {k}: {v}\n")
                rf.write(f"\nClassification Report:\n{report}")

            all_results[tag] = {
                **metrics,
                "time_sec":    round(opt_time + train_time, 1),
                "best_params": params,
            }

    # -- Phase 3: Comparative Analysis ---------------------------------------
    if all_results:
        plot_final_comparison(all_results)
        _print_summary(all_results)
        _save_csv(all_results)

    print("\n  All experiments completed.")
    print("  Results : results/")
    print("  Plots   : results/plots/")


def _print_summary(results):
    """Print a summary table of all experimental results."""
    print("\n" + "=" * 75)
    print("  SUMMARY TABLE -- DAGM 2007 (Test Set)")
    print("=" * 75)
    print(f"  {'Model':<30}{'Acc':>8}{'Prec':>8}{'Recall':>8}{'F1':>8}{'Time(min)':>10}")
    print("-" * 75)
    for tag, m in sorted(results.items()):
        print(f"  {tag:<30}"
              f"{m['accuracy']:>8.4f}"
              f"{m['precision']:>8.4f}"
              f"{m['recall']:>8.4f}"
              f"{m['f1_score']:>8.4f}"
              f"{m['time_sec']/60:>10.1f}")
    print("=" * 75)


def _save_csv(results):
    """Save comparison results to CSV."""
    path = "results/final_comparison_table.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Optimizer", "Accuracy", "Precision", "Recall",
                     "F1_Score", "Time_min"])
        for tag, m in sorted(results.items()):
            parts = tag.rsplit("_", 1)
            model_part = parts[0] if len(parts) == 2 else tag
            opt_part = parts[1] if len(parts) == 2 else ""
            w.writerow([model_part, opt_part, m["accuracy"], m["precision"],
                        m["recall"], m["f1_score"],
                        round(m["time_sec"] / 60, 1)])
    print(f"\n  [CSV] {path}")


if __name__ == "__main__":
    main()
