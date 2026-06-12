"""
Bayesian Hyperparameter Optimization using Tree-structured Parzen Estimator (TPE).

This module implements Optuna-based TPE optimization for CNN hyperparameters
on the DAGM 2007 dataset. It supports optional grid-based pruning thresholds
derived from a prior genetic algorithm search, enabling knowledge-guided
early stopping of unpromising trials.

Reference:
    Acici et al. (2020) - Bone fracture detection using GA+CNN optimization.
"""

import time, json, os
import optuna
from optuna.samplers import TPESampler

from utils.data_loader import get_loaders
from utils.trainer     import train_model, get_device

RESULT_DIR = "results/bayesian"
os.makedirs(RESULT_DIR, exist_ok=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_objective(model_name: str, epochs: int = 5, num_classes: int = 10, grid_thresholds: dict = None):
    """Create an Optuna objective function for Bayesian TPE optimization.

    The returned callable builds and trains a CNN model with trial-suggested
    hyperparameters, returning the best validation accuracy as the objective
    value to maximize.

    Args:
        model_name: Target model architecture name.
        epochs: Number of training epochs per trial.
        num_classes: Number of output classes for the classification task.
        grid_thresholds: Optional per-optimizer pruning thresholds from GA.

    Returns:
        An objective function compatible with Optuna's optimization interface.
    """
    from models.vgg16_model      import build_vgg16,      get_optimizer as opt_vgg
    from models.resnet50_model   import build_resnet50,   get_optimizer as opt_res
    from models.custom_cnn_model import build_custom_cnn, get_optimizer as opt_cus
    from models.custom_cnn_v2_model import build_custom_cnn_v2, get_optimizer as opt_cus_v2

    device = get_device()

    def objective(trial):
        lr      = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        dropout = trial.suggest_float("dropout_rate",  0.2, 0.6)
        dense   = trial.suggest_categorical("dense_units",  [128, 256, 384, 512])
        opt_n   = trial.suggest_categorical("optimizer",    ["adam", "sgd", "rmsprop"])
        batch   = trial.suggest_categorical("batch_size",   [16, 32, 64])

        train_loader, val_loader, _, _ = get_loaders(img_size=224, batch_size=batch)

        try:
            if model_name == "vgg16":
                model = build_vgg16(num_classes=num_classes,
                                    dropout_rate=dropout, dense_units=dense)
                optim = opt_vgg(model, opt_n, lr)
            elif model_name == "resnet50":
                model = build_resnet50(num_classes=num_classes,
                                       dropout_rate=dropout, dense_units=dense)
                optim = opt_res(model, opt_n, lr)
            elif model_name == "custom":
                base_f = trial.suggest_categorical("base_filters", [16, 32, 64])
                n_blk  = trial.suggest_int("num_blocks", 2, 5)
                kern   = trial.suggest_categorical("kernel_size", [3, 5])
                model  = build_custom_cnn(
                    num_classes=num_classes,
                    base_filters=base_f, num_blocks=n_blk,
                    kernel_size=kern, dropout_rate=dropout,
                )
                optim = opt_cus(model, opt_n, lr)
            else:
                base_f = trial.suggest_categorical("base_filters", [8, 16, 32])
                kern   = trial.suggest_categorical("kernel_size", [3, 5])
                model  = build_custom_cnn_v2(
                    num_classes=num_classes,
                    base_filters=base_f,
                    kernel_size=kern, dropout_rate=dropout,
                )
                optim = opt_cus_v2(model, opt_n, lr)

            current_threshold = None
            if grid_thresholds is not None:
                current_threshold = grid_thresholds.get(opt_n, 0.0)

            _, _, best_val_acc = train_model(
                model, train_loader, val_loader, optim,
                epochs=epochs, patience=3, device=device, prune_threshold=current_threshold
            )
        except Exception as e:
            print(f"  [Bayesian Error] {e}")
            best_val_acc = 0.0

        return best_val_acc

    return objective


def run_bayesian(
    model_name:       str,
    n_trials:         int = 15,
    epochs_per_trial: int = 5,
    num_classes:      int = 10,
):
    """Execute Bayesian TPE hyperparameter optimization.

    Creates an Optuna study with TPE sampling and optimizes CNN
    hyperparameters over the specified number of trials.

    Args:
        model_name: Target model architecture name.
        n_trials: Number of optimization trials.
        epochs_per_trial: Training epochs per trial evaluation.
        num_classes: Number of output classes.

    Returns:
        Tuple of (best_params, best_val_acc, elapsed, study).
    """
    print(f"\n{'='*55}")
    print(f" Bayesian TPE Optimization - Model: {model_name.upper()}")
    print(f" Trials: {n_trials} | Epochs/Trial: {epochs_per_trial}")
    print(f"{'='*55}")

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        study_name=f"{model_name}_tpe"
    )

    t0 = time.time()
    study.optimize(
        make_objective(model_name, epochs_per_trial, num_classes),
        n_trials=n_trials,
        show_progress_bar=True
    )
    elapsed = time.time() - t0

    best_params  = study.best_params
    best_val_acc = study.best_value

    result = {
        "model": model_name,
        "best_params": best_params,
        "best_val_accuracy": round(best_val_acc, 4),
        "elapsed_sec": round(elapsed, 1),
        "trials": [
            {"number": t.number,
             "val_accuracy": round(t.value, 4) if t.value else 0.0,
             "params": t.params}
            for t in study.trials if t.value is not None
        ]
    }
    path = f"{RESULT_DIR}/{model_name}_bayesian_result.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  [Bayesian] Best val_acc={best_val_acc:.4f} | Time={elapsed/60:.1f} min")
    return best_params, best_val_acc, elapsed, study
