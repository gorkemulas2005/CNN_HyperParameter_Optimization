"""
optimization/bayesian_tpe.py  —  DAGM 2007
--------------------------------------------
Optuna TPE — num_classes parametresi dışarıdan alınır.
"""

import time, json, os
import optuna
from optuna.samplers import TPESampler

from utils.data_loader import get_loaders
from utils.trainer     import train_model, get_device

RESULT_DIR = "results/bayesian"
os.makedirs(RESULT_DIR, exist_ok=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_objective(model_name: str, epochs: int = 5, num_classes: int = 10):
    from models.vgg16_model      import build_vgg16,      get_optimizer as opt_vgg
    from models.resnet50_model   import build_resnet50,   get_optimizer as opt_res
    from models.custom_cnn_model import build_custom_cnn, get_optimizer as opt_cus

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
            else:
                base_f = trial.suggest_categorical("base_filters", [16, 32, 64])
                n_blk  = trial.suggest_int("num_blocks", 2, 5)
                kern   = trial.suggest_categorical("kernel_size", [3, 5])
                model  = build_custom_cnn(
                    num_classes=num_classes,
                    base_filters=base_f, num_blocks=n_blk,
                    kernel_size=kern, dropout_rate=dropout,
                )
                optim = opt_cus(model, opt_n, lr)

            _, _, best_val_acc = train_model(
                model, train_loader, val_loader, optim,
                epochs=epochs, patience=3, device=device
            )
        except Exception as e:
            print(f"  [Bayesian Hata] {e}")
            best_val_acc = 0.0

        return best_val_acc

    return objective


def run_bayesian(
    model_name:       str,
    n_trials:         int = 15,
    epochs_per_trial: int = 5,
    num_classes:      int = 10,
):
    print(f"\n{'='*55}")
    print(f" Bayesian TPE — Model: {model_name.upper()}")
    print(f" Deneme: {n_trials} | Epoch/Deneme: {epochs_per_trial}")
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

    print(f"\n  [Bayesian] En iyi val_acc={best_val_acc:.4f} | Süre={elapsed/60:.1f} dk")
    return best_params, best_val_acc, elapsed, study
