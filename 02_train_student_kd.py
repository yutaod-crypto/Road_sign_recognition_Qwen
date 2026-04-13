#!/usr/bin/env python3
"""
Train Qwen2.5-VL student with output-level KD: L = λ L_CE + (1-λ) T^2 KL(p_t || p_s).

Expects JSONL from `01_cache_teacher_soft_labels.py` (teacher_probs aligned with each image).

Default student is Qwen2.5-VL-3B-Instruct (smallest public Qwen2.5-VL checkpoint; plan textmentions 2B but that size is not in the Qwen2.5-VL release series).

Usage:
  CUDA_VISIBLE_DEVICES=0 python 02_train_student_kd.py \\
    --train_jsonl artifacts/gtsrb_teacher_soft_train_small.jsonl \\
    --out_dir artifacts/gtsrb_student_kd_run1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from tqdm import tqdm
from final_common import LABELS_6, REPO_ROOT, label_to_index
from final_qwen25_scoring import infer_model_device, load_qwen25_vl, score_label_completions


def load_image(path: str, max_side: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train_jsonl",
        default=str(REPO_ROOT / "artifacts/gtsrb_teacher_soft_train_small.jsonl"),
    )
    ap.add_argument(
        "--out_dir",
        default=str(REPO_ROOT / "artifacts/gtsrb_student_kd"),
    )
    ap.add_argument(
        "--student_model_id",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
    )
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda_ce", type=float, default=0.5, help="Weight on hard-label CE")
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--max_image_side", type=int, default=224)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--max_samples", type=int, default=0, help="0 = all ok rows")
    ap.add_argument("--save_every", type=int, default=50)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for line in Path(args.train_jsonl).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("ok", True):
            continue
        if r.get("teacher_probs") is None:
            continue
        rows.append(r)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    if not rows:
        raise SystemExit(f"No usable rows in {args.train_jsonl}")

    completions = [json.dumps({"label": lab}, ensure_ascii=False) for lab in LABELS_6]
    instruction = (
        "Classify the traffic sign.\n"
        f"Labels: {LABELS_6}\n"
        'Return only JSON: {"label":"stop"}'
    )

    print(f"Loading student {args.student_model_id} ...")
    model, processor = load_qwen25_vl(
        args.student_model_id,
        load_in_4bit=False,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    opt = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    T = args.temperature
    lam = args.lambda_ce

    global_step = 0
    for epoch in range(args.epochs):
        pbar = tqdm(rows, desc=f"epoch {epoch+1}/{args.epochs}")
        for ex in pbar:
            img = load_image(ex["image"], args.max_image_side)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": instruction},
                    ],
                }
            ]

            dev = infer_model_device(model)
            p_t = torch.tensor(ex["teacher_probs"], dtype=torch.float32, device=dev)
            p_t = p_t / (p_t.sum() + 1e-12)
            gt_idx = label_to_index(ex["label"])

            opt.zero_grad(set_to_none=True)
            scores = score_label_completions(
                model,
                processor,
                messages=messages,
                completion_texts=completions,
            )
            log_p_s = F.log_softmax(scores / T, dim=-1)
            p_s = log_p_s.exp()

            loss_kd = (T**2) * F.kl_div(log_p_s, p_t, reduction="sum")
            loss_ce = -log_p_s[gt_idx]
            loss = lam * loss_ce + (1.0 - lam) * loss_kd
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            opt.step()

            pred = int(torch.argmax(p_s).item())
            acc = 1.0 if pred == gt_idx else 0.0
            pbar.set_postfix(loss=float(loss.item()), acc=acc)
            global_step += 1

            if global_step % args.save_every == 0:
                ckpt = out_dir / f"checkpoint-{global_step}"
                ckpt.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(ckpt)

    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    print(f"Saved LoRA adapter -> {final_dir}")


if __name__ == "__main__":
    main()
