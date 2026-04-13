import csv
import json
import random
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_ROOT = REPO_ROOT / "datasets/gtsrb/train"
OUT_TRAIN = REPO_ROOT / "artifacts/gtsrb_train_full.jsonl"
OUT_VAL = REPO_ROOT / "artifacts/gtsrb_val_full.jsonl"

VAL_RATIO = 0.1
SEED = 42

GROUP_MAP = {
    "speed_limit": {0, 1, 2, 3, 4, 5, 6, 7, 8, 32, 41, 42},
    "stop": {14},
    "yield": {13},
    "no_entry": {15, 16, 17},
    "warning": {18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
    "direction": {9, 10, 11, 12, 33, 34, 35, 36, 37, 38, 39, 40},
}

def map_class_id(class_id: int) -> str:
    for k, v in GROUP_MAP.items():
        if class_id in v:
            return k
    raise ValueError(f"Unmapped class id: {class_id}")

def main():
    if not TRAIN_ROOT.exists():
        raise FileNotFoundError(f"Training root not found: {TRAIN_ROOT.resolve()}")

    rows = []

    class_dirs = [p for p in TRAIN_ROOT.iterdir() if p.is_dir()]
    print(f"Found {len(class_dirs)} class folders under {TRAIN_ROOT.resolve()}")

    for class_dir in sorted(class_dirs):
        class_name = class_dir.name

        if not class_name.isdigit():
            print(f"Skipping non-numeric folder: {class_name}")
            continue

        gt_csv = class_dir / f"GT-{class_name}.csv"
        if not gt_csv.exists():
            print(f"Skipping {class_name}: missing {gt_csv.name}")
            continue

        with open(gt_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                try:
                    class_id = int(row["ClassId"])
                except Exception:
                    continue

                image_path = class_dir / row["Filename"]
                if not image_path.exists():
                    continue

                rows.append({
                    "image": str(image_path.resolve()),
                    "label": map_class_id(class_id),
                    "class_id": class_id,
                })

    print(f"Total usable rows: {len(rows)}")

    if len(rows) == 0:
        raise RuntimeError("No rows found.")

    random.seed(SEED)
    random.shuffle(rows)

    n_val = int(len(rows) * VAL_RATIO)
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    OUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(OUT_VAL, "w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved train: {len(train_rows)} -> {OUT_TRAIN}")
    print(f"Saved val:   {len(val_rows)} -> {OUT_VAL}")

if __name__ == "__main__":
    main()