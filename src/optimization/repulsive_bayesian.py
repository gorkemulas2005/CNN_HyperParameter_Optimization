"""
Repulsive Bayesian Optimization (Hybrid GA + TPE Approach).

This module implements a two-phase hybrid optimization strategy. In Phase 1,
a short genetic algorithm run identifies per-optimizer performance thresholds
using a MAP-Elites-inspired grid decomposition. In Phase 2, these thresholds
are passed to Optuna's TPE sampler as pruning criteria, enabling early
termination of trials that fall within poorly-performing regions of the
hyperparameter space.

Reference:
    Acici et al. (2020) - Bone fracture detection using GA+CNN optimization.
"""

import time
import json
import os
import optuna
from optuna.samplers import TPESampler
from optimization.genetic_algorithm import run_ga
from optimization.bayesian_tpe import make_objective

RESULT_DIR = "results/repulsive_hybrid"
os.makedirs(RESULT_DIR, exist_ok=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)

def run_repulsive_bayesian(model_name: str, n_trials: int = 15, epochs_per_trial: int = 5, num_classes: int = 10):
    """Execute the repulsive hybrid optimization strategy.

    Combines a brief GA exploration phase with Bayesian TPE optimization.
    The GA-derived grid thresholds guide the TPE phase by pruning trials
    whose performance falls below the per-optimizer median threshold.

    Args:
        model_name: Target model architecture name.
        n_trials: Number of Bayesian optimization trials in Phase 2.
        epochs_per_trial: Training epochs per trial evaluation.
        num_classes: Number of output classes for the classification task.

    Returns:
        Tuple of (best_params, best_val_acc, total_elapsed, study).
    """
    print(f"\n{'='*60}")
    print(f" Repulsive Hybrid (GA + BO) - Model: {model_name.upper()}")
    print(f" Trials: {n_trials} | Epochs/Trial: {epochs_per_trial}")
    print(f"{'='*60}")

    total_t0 = time.time()

    # Phase 1: Brief GA exploration to compute per-optimizer grid thresholds
    print("\n  [Phase 1] Running short GA to identify grid thresholds (median performance)...")
    _, grid_thresholds, ga_val_acc, ga_elapsed, _ = run_ga(
        model_name=model_name,
        n_generations=2, 
        pop_size=5, 
        cx_prob=0.5, 
        mut_prob=0.5, 
        epochs_per_eval=3, 
        num_classes=num_classes
    )

    print(f"  -> MAP-Elites grid thresholds identified by GA: {grid_thresholds}")

    # Phase 2: Bayesian TPE optimization with grid-based pruning
    print("\n  [Phase 2] Starting Bayesian TPE optimization with MAP-Elites grid pruning...")
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        study_name=f"{model_name}_hybrid"
    )

    # Pass grid thresholds for dynamic early stopping of unpromising trials
    objective_fn = make_objective(model_name, epochs_per_trial, num_classes, grid_thresholds=grid_thresholds)

    # Execute the Bayesian optimization trials
    t1 = time.time()
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=True)
    bo_elapsed = time.time() - t1

    total_elapsed = time.time() - total_t0

    best_params = study.best_params
    best_val_acc = study.best_value

    result = {
        "model": model_name,
        "best_params": best_params,
        "grid_thresholds_used": grid_thresholds,
        "best_val_accuracy": round(best_val_acc, 4),
        "elapsed_sec": round(total_elapsed, 1),
        "trials": [
            {"number": t.number, "val_accuracy": round(t.value, 4) if t.value else 0.0, "params": t.params}
            for t in study.trials if t.value is not None
        ]
    }
    
    path = f"{RESULT_DIR}/{model_name}_repulsive_result.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  [Repulsive Hybrid] Best val_acc={best_val_acc:.4f} | Total time={total_elapsed/60:.1f} min")
    return best_params, best_val_acc, total_elapsed, study
