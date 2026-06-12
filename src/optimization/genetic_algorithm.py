"""
Genetic Algorithm-based Hyperparameter Optimization for CNN Models.

This module implements a DEAP-based genetic algorithm (GA) for optimizing
CNN hyperparameters on the DAGM 2007 dataset. The GA evolves a population
of candidate hyperparameter configurations using tournament selection,
two-point crossover, and uniform integer mutation. A MAP-Elites-inspired
quality-diversity mechanism partitions the population by optimizer type
to compute per-niche performance thresholds.

Reference:
    Acici et al. (2020) - Bone fracture detection using GA+CNN optimization.
"""

import random, time, json, os
import numpy as np
from deap import base, creator, tools, algorithms

from utils.data_loader import get_loaders
from utils.trainer     import train_model, get_device

LR_OPTIONS      = [1e-5, 1e-4, 5e-4, 1e-3, 1e-2]
DROPOUT_OPTIONS = [0.2, 0.3, 0.4, 0.5, 0.6]
DENSE_OPTIONS   = [128, 256, 384, 512]
OPT_OPTIONS     = ["adam", "sgd", "rmsprop"]
BATCH_OPTIONS   = [16, 32, 64]
FILTER_OPTIONS  = [16, 32, 64]
BLOCK_OPTIONS   = [2, 3, 4, 5]
KERNEL_OPTIONS  = [3, 5]

COMMON_LEN = 5
CUSTOM_LEN = 8

RESULT_DIR = "results/ga"
os.makedirs(RESULT_DIR, exist_ok=True)


def _decode(ind, model_name):
    """Decode an integer-encoded individual into a hyperparameter dictionary.

    Args:
        ind: List of integer genes representing encoded hyperparameters.
        model_name: Name of the target model architecture.

    Returns:
        Dictionary mapping hyperparameter names to their decoded values.
    """
    p = {
        "learning_rate": LR_OPTIONS[     ind[0] % len(LR_OPTIONS)],
        "dropout_rate":  DROPOUT_OPTIONS[ind[1] % len(DROPOUT_OPTIONS)],
        "dense_units":   DENSE_OPTIONS[  ind[2] % len(DENSE_OPTIONS)],
        "optimizer":     OPT_OPTIONS[    ind[3] % len(OPT_OPTIONS)],
        "batch_size":    BATCH_OPTIONS[  ind[4] % len(BATCH_OPTIONS)],
    }
    if model_name in ["custom", "custom_v2"]:
        p["base_filters"] = FILTER_OPTIONS[ind[5] % len(FILTER_OPTIONS)]
        p["num_blocks"]   = BLOCK_OPTIONS[ ind[6] % len(BLOCK_OPTIONS)]
        p["kernel_size"]  = KERNEL_OPTIONS[ind[7] % len(KERNEL_OPTIONS)]
    return p


def make_fitness_fn(model_name: str, epochs: int = 5, num_classes: int = 10):
    """Create a fitness evaluation function for the genetic algorithm.

    The returned callable trains a CNN model with the decoded hyperparameters
    and returns the best validation accuracy as the fitness value.

    Args:
        model_name: Target model architecture name.
        epochs: Number of training epochs per evaluation.
        num_classes: Number of output classes for the classification task.

    Returns:
        A fitness function compatible with DEAP's evaluation interface.
    """
    from models.vgg16_model      import build_vgg16,      get_optimizer as opt_vgg
    from models.resnet50_model   import build_resnet50,   get_optimizer as opt_res
    from models.custom_cnn_model import build_custom_cnn, get_optimizer as opt_cus
    from models.custom_cnn_v2_model import build_custom_cnn_v2, get_optimizer as opt_cus_v2

    device = get_device()

    def fitness(individual):
        p = _decode(individual, model_name)
        train_loader, val_loader, _, _ = get_loaders(
            img_size=224, batch_size=p["batch_size"]
        )
        try:
            if model_name == "vgg16":
                model = build_vgg16(num_classes=num_classes,
                                    dropout_rate=p["dropout_rate"],
                                    dense_units=p["dense_units"])
                optim = opt_vgg(model, p["optimizer"], p["learning_rate"])
            elif model_name == "resnet50":
                model = build_resnet50(num_classes=num_classes,
                                       dropout_rate=p["dropout_rate"],
                                       dense_units=p["dense_units"])
                optim = opt_res(model, p["optimizer"], p["learning_rate"])
            elif model_name == "custom":
                model = build_custom_cnn(
                    num_classes=num_classes,
                    base_filters=p.get("base_filters", 32),
                    num_blocks=p.get("num_blocks", 3),
                    kernel_size=p.get("kernel_size", 3),
                    dropout_rate=p["dropout_rate"],
                )
                optim = opt_cus(model, p["optimizer"], p["learning_rate"])
            else:
                model = build_custom_cnn_v2(
                    num_classes=num_classes,
                    base_filters=p.get("base_filters", 32),
                    kernel_size=p.get("kernel_size", 3),
                    dropout_rate=p["dropout_rate"],
                )
                optim = opt_cus_v2(model, p["optimizer"], p["learning_rate"])

            _, _, best_val_acc = train_model(
                model, train_loader, val_loader, optim,
                epochs=epochs, patience=3, device=device
            )
        except Exception as e:
            print(f"  [GA Error] {e}")
            best_val_acc = 0.0

        return (best_val_acc,)

    return fitness


def run_ga(
    model_name:      str,
    n_generations:   int   = 5,
    pop_size:        int   = 10,
    cx_prob:         float = 0.7,
    mut_prob:        float = 0.2,
    epochs_per_eval: int   = 5,
    num_classes:     int   = 10,
):
    """Execute genetic algorithm optimization for CNN hyperparameters.

    Evolves a population of hyperparameter configurations over multiple
    generations using DEAP. After evolution, computes MAP-Elites-style
    grid thresholds partitioned by optimizer type.

    Args:
        model_name: Target model architecture name.
        n_generations: Number of GA generations.
        pop_size: Population size.
        cx_prob: Crossover probability.
        mut_prob: Mutation probability.
        epochs_per_eval: Training epochs per fitness evaluation.
        num_classes: Number of output classes.

    Returns:
        Tuple of (best_params, grid_thresholds, best_val_acc, elapsed, logbook).
    """
    print(f"\n{'='*55}")
    print(f" GA Optimization - Model: {model_name.upper()}")
    print(f" Population: {pop_size} | Generations: {n_generations} | Epochs/Individual: {epochs_per_eval}")
    print(f"{'='*55}")

    gene_len = CUSTOM_LEN if model_name in ["custom", "custom_v2"] else COMMON_LEN

    for name in ["FitnessMax", "Individual"]:
        if name in creator.__dict__:
            delattr(creator, name)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    tb = base.Toolbox()
    tb.register("attr_int",   random.randint, 0, 9)
    tb.register("individual", tools.initRepeat, creator.Individual, tb.attr_int, n=gene_len)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("evaluate",   make_fitness_fn(model_name, epochs_per_eval, num_classes))
    tb.register("mate",       tools.cxTwoPoint)
    tb.register("mutate",     tools.mutUniformInt, low=0, up=9, indpb=0.2)
    tb.register("select",     tools.selTournament, tournsize=3)

    pop  = tb.population(n=pop_size)
    hof  = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    t0 = time.time()
    pop, logbook = algorithms.eaSimple(
        pop, tb, cxpb=cx_prob, mutpb=mut_prob,
        ngen=n_generations, stats=stats, halloffame=hof, verbose=True
    )
    elapsed = time.time() - t0

    best_ind     = hof[0]
    best_val_acc = best_ind.fitness.values[0]
    best_params  = _decode(best_ind, model_name)

    # MAP-Elites quality-diversity: partition search space into grids by optimizer type
    grid_thresholds = {"adam": 0.0, "sgd": 0.0, "rmsprop": 0.0}
    grid_pops = {"adam": [], "sgd": [], "rmsprop": []}
    
    # Global fallback if a grid is completely empty
    all_scores = []
    
    for ind in pop:
        p = _decode(ind, model_name)
        opt_type = p["optimizer"]
        score = float(ind.fitness.values[0])
        grid_pops[opt_type].append(score)
        all_scores.append(score)
        
    all_scores.sort()
    global_median = all_scores[len(all_scores) // 2] if all_scores else 0.0

    for opt_type in grid_thresholds.keys():
        if len(grid_pops[opt_type]) > 0:
            grid_pops[opt_type].sort()
            median_idx = len(grid_pops[opt_type]) // 2
            grid_thresholds[opt_type] = grid_pops[opt_type][median_idx]
        else:
            # Use global median as fallback when this optimizer was not sampled
            grid_thresholds[opt_type] = global_median

    result = {
        "model": model_name,
        "best_params": best_params,
        "grid_thresholds": {k: round(v, 4) for k, v in grid_thresholds.items()},
        "best_val_accuracy": round(best_val_acc, 4),
        "elapsed_sec": round(elapsed, 1),
        "logbook": [{"gen": r["gen"], "max": round(float(r["max"]), 4),
                     "avg": round(float(r["avg"]), 4)} for r in logbook]
    }
    path = f"{RESULT_DIR}/{model_name}_ga_result.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  [GA] Best val_acc={best_val_acc:.4f} | Grid thresholds={grid_thresholds} | Time={elapsed/60:.1f} min")
    return best_params, grid_thresholds, best_val_acc, elapsed, logbook
