"""
prepare_dataset.py  —  DAGM 2007
----------------------------------
Ham yapı:
  data/DAGM2007/raw/
    Class1/
      Train/   ← 575 görüntü
      Test/    ← 575 görüntü
    Class2/ ... Class10/

Strateji:
  Train + Test birleştirilir, toplam 1150 görüntü/sınıf
  %70 train / %15 val / %15 test olarak yeniden bölünür.

Çıktı:
  data/DAGM2007/train/   (~805 görüntü/sınıf)
  data/DAGM2007/val/     (~173 görüntü/sınıf)
  data/DAGM2007/test/    (~172 görüntü/sınıf)
"""

import shutil, random
from pathlib import Path
from collections import defaultdict

RAW_DIR   = Path("data/DAGM2007/raw")
BASE_DIR  = Path("data/DAGM2007")
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR   = BASE_DIR / "val"
TEST_DIR  = BASE_DIR / "test"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
SEED        = 42
CLASSES     = [f"Class{i}" for i in range(1, 11)]


def is_valid_image(f: Path) -> bool:
    if f.is_dir():
        return False
    if "_label" in f.stem.lower():
        return False
    if f.suffix.upper() != ".PNG":
        return False
    return True


def collect_all(raw_dir: Path):
    """Train + Test klasörlerini birleştirerek tüm görüntüleri toplar."""
    class_files = defaultdict(list)
    for cls in CLASSES:
        cls_dir = raw_dir / cls
        if not cls_dir.exists():
            print(f"  [UYARI] {cls_dir} bulunamadı.")
            continue
        images = []
        for split_name in ["Train", "Test"]:
            split_dir = cls_dir / split_name
            if split_dir.exists():
                images += [f for f in split_dir.iterdir() if is_valid_image(f)]
        class_files[cls] = images
    return class_files


def split_and_copy(class_files: dict):
    random.seed(SEED)
    total_train = total_val = total_test = 0

    print(f"\n  {'Sınıf':<12} {'Toplam':>8} {'Train':>8} {'Val':>6} {'Test':>7}")
    print("  " + "─" * 45)

    for cls in CLASSES:
        images = class_files.get(cls, [])
        if not images:
            print(f"  [UYARI] '{cls}' için görüntü yok.")
            continue

        random.shuffle(images)
        n       = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)
        n_test  = n - n_train - n_val

        splits = {
            TRAIN_DIR: images[:n_train],
            VAL_DIR:   images[n_train:n_train + n_val],
            TEST_DIR:  images[n_train + n_val:],
        }
        for dest_base, files in splits.items():
            dest = dest_base / cls
            dest.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, dest / f.name)

        print(f"  {cls:<12} {n:>8} {n_train:>8} {n_val:>6} {n_test:>7}")
        total_train += n_train
        total_val   += n_val
        total_test  += n_test

    print("  " + "─" * 45)
    total = total_train + total_val + total_test
    print(f"  {'TOPLAM':<12} {total:>8} {total_train:>8} {total_val:>6} {total_test:>7}")
    return total_train, total_val, total_test


def main():
    print("=" * 55)
    print("  DAGM 2007 Veri Hazırlama  (10 sınıf)")
    print("  %70 train / %15 val / %15 test")
    print("=" * 55)

    if not RAW_DIR.exists():
        print(f"\n[HATA] Klasör bulunamadı: {RAW_DIR.resolve()}")
        print("\n  Beklenen yapı:")
        print("    data/DAGM2007/raw/Class1/Train/0001.PNG")
        print("    data/DAGM2007/raw/Class1/Test/0001.PNG")
        print("    ...")
        return

    print("\n  Tüm görüntüler toplanıyor (Train + Test birleştiriliyor)...")
    class_files = collect_all(RAW_DIR)

    total = sum(len(v) for v in class_files.values())
    if total == 0:
        print("\n[HATA] Hiç görüntü bulunamadı.")
        return

    print(f"  Toplam {total} görüntü bulundu.\n")
    tr, va, te = split_and_copy(class_files)

    print(f"\n  Train : {tr}")
    print(f"  Val   : {va}")
    print(f"  Test  : {te}")
    print("\n✓ Hazır! Şimdi: python train_and_compare.py")
    print("=" * 55)


if __name__ == "__main__":
    main()