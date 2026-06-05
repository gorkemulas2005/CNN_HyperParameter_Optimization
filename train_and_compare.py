"""
train_and_compare.py  —  DAGM 2007
=====================================
ANA ÇALIŞTIRMA DOSYASI — PyTorch + RTX 4060

10 doku sınıfı | GA vs Bayesian TPE | 3 model
"""

import os, sys, time, json, csv, warnings
warnings.filterwarnings("ignore")

import torch

# ── CONFIG ────────────────────────────────────────────────────────────────────
CONFIG = {
    "ga_generations":        5,
    "ga_pop_size":          10,
    "ga_epochs_per_eval":    5,

    "bayes_n_trials":       15,
    "bayes_epochs_per_trial": 5,

    "final_epochs":         30,
    "final_batch_size":     32,
    "final_patience":        7,

    "models_to_run":     ["vgg16", "resnet50", "custom"],
    "optimizers_to_run": ["ga", "bayesian"],

    "img_size":     224,
    "num_classes":   10,   # DAGM: 10 sınıf
    "num_workers":    0,   # Windows'ta 0 bırak
}
# ──────────────────────────────────────────────────────────────────────────────

def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props  = torch.cuda.get_device_properties(0)
        vram   = props.total_memory / 1024**3
        print(f"  GPU : {props.name}")
        print(f"  VRAM: {vram:.1f} GB")
        torch.backends.cudnn.benchmark = True
        return device
    print("  GPU bulunamadı — CPU kullanılıyor.")
    return torch.device("cpu")

for d in ["results/ga", "results/bayesian", "results/plots",
          "results/models", "logs"]:
    os.makedirs(d, exist_ok=True)

from utils.data_loader import get_loaders, NUM_CLASSES
from utils.trainer     import train_model
from utils.metrics     import evaluate_model
from utils.plotter     import (plot_history, plot_confusion_matrix,
                                plot_final_comparison)
from optimization.genetic_algorithm import run_ga
from optimization.bayesian_tpe      import run_bayesian
from models.vgg16_model      import build_vgg16,      get_optimizer as opt_vgg
from models.resnet50_model   import build_resnet50,   get_optimizer as opt_res
from models.custom_cnn_model import build_custom_cnn, get_optimizer as opt_cus


def build_final_model(model_name, params):
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
    else:
        model = build_custom_cnn(
            num_classes=nc,
            base_filters=params.get("base_filters", 32),
            num_blocks=params.get("num_blocks",     3),
            kernel_size=params.get("kernel_size",   3),
            dropout_rate=dropout,
        )
        optim = opt_cus(model, opt_n, lr)
    return model, optim


def main():
    print("\n" + "═"*60)
    print("  DAGM 2007 — CNN Hiperparametre Optimizasyonu")
    print("  GA vs Bayesian TPE  |  10 Sınıf  |  PyTorch")
    print("═"*60)

    device = setup_device()

    train_loader, val_loader, test_loader, class_names = get_loaders(
        img_size=CONFIG["img_size"],
        batch_size=CONFIG["final_batch_size"],
        num_workers=CONFIG["num_workers"],
    )
    print(f"\n  Sınıflar : {class_names}")
    print(f"  Train: {len(train_loader.dataset)} | "
          f"Val: {len(val_loader.dataset)} | "
          f"Test: {len(test_loader.dataset)}\n")

    best_params_all = {}
    all_results     = {}

    # ── 1. OPTİMİZASYON ──────────────────────────────────────────────────────
    for model_name in CONFIG["models_to_run"]:
        best_params_all[model_name] = {}

        if "ga" in CONFIG["optimizers_to_run"]:
            params, val_acc, t_sec, _ = run_ga(
                model_name=model_name,
                n_generations=CONFIG["ga_generations"],
                pop_size=CONFIG["ga_pop_size"],
                epochs_per_eval=CONFIG["ga_epochs_per_eval"],
                num_classes=CONFIG["num_classes"],
            )
            best_params_all[model_name]["ga"] = {
                "params": params, "val_acc": val_acc, "time_sec": t_sec
            }

        if "bayesian" in CONFIG["optimizers_to_run"]:
            params, val_acc, t_sec, _ = run_bayesian(
                model_name=model_name,
                n_trials=CONFIG["bayes_n_trials"],
                epochs_per_trial=CONFIG["bayes_epochs_per_trial"],
                num_classes=CONFIG["num_classes"],
            )
            best_params_all[model_name]["bayesian"] = {
                "params": params, "val_acc": val_acc, "time_sec": t_sec
            }

    with open("results/best_params_all.json", "w") as f:
        json.dump(best_params_all, f, indent=2)

    # ── 2. FİNAL EĞİTİM ──────────────────────────────────────────────────────
    print("\n" + "─"*55)
    print("  Final Eğitim Başlıyor...")
    print("─"*55)

    for model_name in CONFIG["models_to_run"]:
        for opt_name in CONFIG["optimizers_to_run"]:
            if opt_name not in best_params_all.get(model_name, {}):
                continue

            tag      = f"{model_name.upper()}_{opt_name.upper()}"
            params   = best_params_all[model_name][opt_name]["params"]
            opt_time = best_params_all[model_name][opt_name]["time_sec"]

            print(f"\n  [{tag}] Final eğitim...")
            print(f"  Parametreler: {params}")

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

            print(f"\n  [{tag}] ── Test Sonuçları ──")
            print(f"    Accuracy  : {metrics['accuracy']:.4f}")
            print(f"    Precision : {metrics['precision']:.4f}")
            print(f"    Recall    : {metrics['recall']:.4f}")
            print(f"    F1-Score  : {metrics['f1_score']:.4f}")
            print(f"\n{report}")

            rpath = f"results/{opt_name}/{model_name}_final_report.txt"
            with open(rpath, "w", encoding="utf-8") as rf:
                rf.write(f"Model    : {tag}\n")
                rf.write(f"Dataset  : DAGM 2007 (10 sınıf)\n")
                rf.write(f"Params   : {json.dumps(params, indent=2)}\n\n")
                rf.write(f"Opt Süresi   : {opt_time/60:.1f} dk\n")
                rf.write(f"Train Süresi : {train_time/60:.1f} dk\n\n")
                rf.write("Test Metrikleri:\n")
                for k, v in metrics.items():
                    rf.write(f"  {k}: {v}\n")
                rf.write(f"\nClassification Report:\n{report}")

            all_results[tag] = {
                **metrics,
                "time_sec":    round(opt_time + train_time, 1),
                "best_params": params,
            }

    # ── 3. KARŞILAŞTIRMA ─────────────────────────────────────────────────────
    if all_results:
        plot_final_comparison(all_results)
        _print_summary(all_results)
        _save_csv(all_results)

    print("\n✓ Tüm işlemler tamamlandı!")
    print("  Sonuçlar  : results/")
    print("  Grafikler : results/plots/")


def _print_summary(results):
    print("\n" + "═"*72)
    print("  ÖZET TABLO — DAGM 2007 (Test Seti)")
    print("═"*72)
    print(f"  {'Model':<25}{'Acc':>8}{'Prec':>8}{'Recall':>8}{'F1':>8}{'Süre(dk)':>10}")
    print("─"*72)
    for tag, m in sorted(results.items()):
        print(f"  {tag:<25}"
              f"{m['accuracy']:>8.4f}"
              f"{m['precision']:>8.4f}"
              f"{m['recall']:>8.4f}"
              f"{m['f1_score']:>8.4f}"
              f"{m['time_sec']/60:>10.1f}")
    print("═"*72)


def _save_csv(results):
    path = "results/comparison_table.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Accuracy", "Precision", "Recall", "F1_Score",
                    "Time_min", "Best_Params"])
        for tag, m in sorted(results.items()):
            w.writerow([tag, m["accuracy"], m["precision"], m["recall"],
                        m["f1_score"], round(m["time_sec"]/60, 1),
                        json.dumps(m.get("best_params", {}))])
    print(f"\n  [CSV] {path}")


if __name__ == "__main__":
    main()
