import csv
import json
import random
from pathlib import Path

TRAIN_ROOT = Path("datasets/gtsrb/train")
OUT_TRAIN = Path("artifacts/gtsrb_train_grouped.jsonl")
OUT_VAL = Path("artifacts/gtsrb_val_grouped.jsonl")

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
    rows = []

    for class_dir in sorted(TRAIN_ROOT.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        gt_csv = class_dir / f"GT-{class_name}.csv"
        if not gt_csv.exists():
            print(f"Warning: missing {gt_csv}")
            continue

        with open(gt_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                filename = row["Filename"]
                class_id = int(row["ClassId"])
                grouped_label = map_class_id(class_id)

                image_path = class_dir / filename
                if not image_path.exists():
                    continue

                rows.append({
                    "image": str(image_path.resolve()),
                    "label": grouped_label,
                    "class_id": class_id
                })

    print("Total training rows:", len(rows))

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