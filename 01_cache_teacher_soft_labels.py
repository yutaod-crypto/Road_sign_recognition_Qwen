#!/usr/bin/env python3
"""
Cache teacher (Qwen2.5-VL-32B) soft labels for GTSRB coarse classification.

Uses the same JSON completion protocol as milestone (`{"label":"..."}`) andscores each of the 6 completions via summed token log-probabilities, then
stores softmax(logits / T) as teacher_probs.

Usage:
  CUDA_VISIBLE_DEVICES=0 python 01_cache_teacher_soft_labels.py \\
    --input_jsonl artifacts/gtsrb_train_small.jsonl \\
    --out_jsonl artifacts/gtsrb_teacher_soft_small.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from final_common import LABELS_6, REPO_ROOT
from final_qwen25_scoring import load_qwen25_vl, score_label_completions


def load_image(path: str, max_side: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_jsonl",
        default=str(REPO_ROOT / "artifacts/gtsrb_train_small.jsonl"),
    )
    ap.add_argument(
        "--out_jsonl",
        default=str(REPO_ROOT / "artifacts/gtsrb_teacher_soft_train_small.jsonl"),
    )
    ap.add_argument(
        "--teacher_model_id",
        default="Qwen/Qwen2.5-VL-32B-Instruct",
    )
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--max_image_side", type=int, default=224)
    ap.add_argument("--max_samples", type=int, default=0, help="0 = all")
    ap.add_argument("--load_in_4bit", action="store_true", help="Strongly recommended for32B")
    ap.add_argument("--prompt_version", default="v1")
    args = ap.parse_args()

    in_path = Path(args.input_jsonl)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        json.loads(line)
        for line in in_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    done_images: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            done_images.add(json.loads(line)["image"])

    completions = [json.dumps({"label": lab}, ensure_ascii=False) for lab in LABELS_6]
    instruction = (
        "Classify the traffic sign.\n"
        f"Labels: {LABELS_6}\n"
        'Return only JSON: {"label":"stop"}'
    )

    print(f"Loading teacher {args.teacher_model_id} ...")
    model, processor = load_qwen25_vl(
        args.teacher_model_id,
        load_in_4bit=args.load_in_4bit,
        device_map="auto",
    )
    model.eval()

    mode = "a" if out_path.exists() else "w"
    n_written = 0
    with open(out_path, mode, encoding="utf-8") as f_out:
        for ex in tqdm(rows):
            image_path = ex["image"]
            if image_path in done_images:
                continue
            try:
                img = load_image(image_path, args.max_image_side)
            except Exception as e:
                rec = {
                    **ex,
                    "ok": False,
                    "error": f"image_load_error: {e}",
                    "teacher_log_scores": None,
                    "teacher_probs": None,
                }
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f_out.flush()
                continue

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": instruction},
                    ],
                }
            ]

            with torch.inference_mode():
                scores = score_label_completions(
                    model,
                    processor,
                    messages=messages,
                    completion_texts=completions,
                )
                probs = F.softmax(scores / args.temperature, dim=-1)

            rec = {
                **ex,
                "ok": True,
                "teacher_log_scores": [float(x) for x in scores.cpu().tolist()],
                "teacher_probs": [float(x) for x in probs.cpu().tolist()],
                "teacher_meta": {
                    "model_id": args.teacher_model_id,
                    "temperature": args.temperature,
                    "prompt_version": args.prompt_version,
                    "label_space": "gtsrb_coarse_6",
                },
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f_out.flush()
            n_written += 1

    print(f"Done. Wrote {n_written} new rows -> {out_path}")


if __name__ == "__main__":
    main()
