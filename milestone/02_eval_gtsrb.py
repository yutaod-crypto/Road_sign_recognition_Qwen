import argparse
import json
from collections import defaultdict, Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELS = ["speed_limit", "stop", "yield", "no_entry", "warning", "direction"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_jsonl", default=str(REPO_ROOT / "artifacts/gtsrb_preds_baseline.jsonl"))
    ap.add_argument("--out_confusion_csv", default=str(REPO_ROOT / "artifacts/gtsrb_confusion.csv"))
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.preds_jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]

    total = len(rows)
    correct = 0
    valid = 0

    confusion = defaultdict(lambda: Counter())

    for row in rows:
        gt = row["gt"]
        pred = row["pred"]["label"]

        if row.get("ok", False):
            valid += 1

        confusion[gt][pred] += 1

        if gt == pred:
            correct += 1

    accuracy = correct / total if total else 0.0
    valid_rate = valid / total if total else 0.0

    df = pd.DataFrame(0, index=LABELS, columns=LABELS)

    for gt in confusion:
        for pred, count in confusion[gt].items():
            if gt in df.index and pred in df.columns:
                df.loc[gt, pred] += count

    out_csv = Path(args.out_confusion_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv)

    per_class_acc = {}
    for label in LABELS:
        row_sum = df.loc[label].sum()
        if row_sum == 0:
            per_class_acc[label] = None
        else:
            per_class_acc[label] = int(df.loc[label, label]) / int(row_sum)

    result = {
        "n": total,
        "accuracy": accuracy,
        "valid_json_rate": valid_rate,
        "per_class_accuracy": per_class_acc,
        "confusion_csv": str(out_csv.resolve())
    }

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()