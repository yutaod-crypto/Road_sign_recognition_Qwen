import csv
import json
import random
from pathlib import Path

CSV_PATH = Path("datasets/gtsrb/GT-final_test.csv")
IMAGE_DIR = Path("datasets/gtsrb/test_images")
OUT_JSONL = Path("artifacts/gtsrb_test_120.jsonl")

SAMPLE_SIZE = 120
SEED = 42

GROUP_MAP = {
    "speed_limit": {0,1,2,3,4,5,6,7,8,32,41,42},
    "stop": {14},
    "yield": {13},
    "no_entry": {15,16,17},
    "warning": {18,19,20,21,22,23,24,25,26,27,28,29,30,31},
    "direction": {9,10,11,12,33,34,35,36,37,38,39,40},
}

def map_class(class_id):
    for k,v in GROUP_MAP.items():
        if class_id in v:
            return k
    return None

rows = []

with open(CSV_PATH) as f:
    reader = csv.DictReader(f, delimiter=";")

    for r in reader:
        class_id = int(r["ClassId"])

        rows.append({
            "image": str((IMAGE_DIR / r["Filename"]).resolve()),
            "label": map_class(class_id),
            "class_id": class_id
        })

print("Total images:", len(rows))

random.seed(SEED)
sample = random.sample(rows, SAMPLE_SIZE)

OUT_JSONL.parent.mkdir(exist_ok=True)

with open(OUT_JSONL,"w") as f:
    for r in sample:
        f.write(json.dumps(r) + "\n")

print("Saved", SAMPLE_SIZE, "samples →", OUT_JSONL)