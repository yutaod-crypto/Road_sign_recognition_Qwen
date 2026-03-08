# 03_prepare_gtsrb_train_small.py

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

TRAIN_ROOT = Path("datasets/gtsrb/train")
OUT_TRAIN = Path("artifacts/gtsrb_train_small.jsonl")
OUT_VAL = Path("artifacts/gtsrb_val_small.jsonl")

TRAIN_PER_CLASS = 10
VAL_PER_CLASS = 5
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

    random.seed(SEED)
    grouped_rows = defaultdict(list)

    class_dirs = [p for p in TRAIN_ROOT.iterdir() if p.is_dir()]
    print(f"Found {len(class_dirs)} class folders under {TRAIN_ROOT.resolve()}")

    total_rows = 0
    total_csv_found = 0

    for class_dir in sorted(class_dirs):
        class_name = class_dir.name

        if not class_name.isdigit():
            print(f"Skipping non-numeric folder: {class_name}")
            continue

        gt_csv = class_dir / f"GT-{class_name}.csv"
        if not gt_csv.exists():
            print(f"Skipping {class_name}: missing {gt_csv.name}")
            continue

        total_csv_found += 1

        with open(gt_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")

            for row in reader:
                try:
                    class_id = int(row["ClassId"])
                except Exception:
                    continue

                grouped_label = map_class_id(class_id)
                image_path = class_dir / row["Filename"]

                if not image_path.exists():
                    continue

                grouped_rows[grouped_label].append({
                    "image": str(image_path.resolve()),
                    "label": grouped_label,
                    "class_id": class_id
                })
                total_rows += 1

    print(f"Found class CSVs: {total_csv_found}")
    print(f"Total usable rows found: {total_rows}")

    if total_rows == 0:
        raise RuntimeError("No training rows found. Check your folder structure and CSV files.")

    train_rows = []
    val_rows = []

    for label in sorted(grouped_rows.keys()):
        rows = grouped_rows[label]
        random.shuffle(rows)

        needed = TRAIN_PER_CLASS + VAL_PER_CLASS
        selected = rows[:needed]

        if len(selected) < needed:
            print(f"Warning: {label} only has {len(selected)} examples, wanted {needed}")

        val_part = selected[:min(VAL_PER_CLASS, len(selected))]
        train_part = selected[min(VAL_PER_CLASS, len(selected)):]

        val_rows.extend(val_part)
        train_rows.extend(train_part)

        print(f"{label}: train={len(train_part)}, val={len(val_part)}, total_available={len(rows)}")

    if len(train_rows) == 0 or len(val_rows) == 0:
        raise RuntimeError(f"Empty split detected: train={len(train_rows)}, val={len(val_rows)}")

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