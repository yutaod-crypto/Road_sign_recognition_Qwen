# 01_run_gtsrb_qwen3vl.py
#
# Baseline inference for GTSRB grouped traffic-sign classification
# using Qwen3-VL-2B on a Windows machine with an RTX 4060.
#
# Supports both 4-bit quantization (Linux/WSL) and fp16 (Windows native).
# Auto-detects bitsandbytes availability.
#
# Usage (from any cwd):
#   python milestone/01_run_gtsrb_qwen3vl.py
#
# Then evaluate with:
#   python milestone/02_eval_gtsrb.py

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# Try importing bitsandbytes; if unavailable, fall back to fp16
try:
    from transformers import BitsAndBytesConfig
    import bitsandbytes as _bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

REPO_ROOT = Path(__file__).resolve().parent.parent
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
LABELS = ["speed_limit", "stop", "yield", "no_entry", "warning", "direction"]


def load_image(path: str, max_side: int):
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return img


def extract_json(text: str):
    m = JSON_RE.search(text)
    if not m:
        raise ValueError("No JSON found in model output")
    return json.loads(m.group(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_jsonl", default=str(REPO_ROOT / "artifacts/gtsrb_test_120.jsonl"))
    ap.add_argument("--out_preds", default=str(REPO_ROOT / "artifacts/gtsrb_preds_baseline.jsonl"))
    ap.add_argument("--model_id", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--max_image_side", type=int, default=168)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--force_fp16", action="store_true",
                    help="Skip 4-bit quantization, use fp16 instead")
    args = ap.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.eval_jsonl).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"Loaded {len(rows)} test examples from {args.eval_jsonl}")

    # Limit Qwen3-VL vision token count to speed up inference
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        min_pixels=28 * 28,
        max_pixels=168 * 168,
    )

    use_bnb = HAS_BNB and not args.force_fp16

    if use_bnb:
        print("Loading model with 4-bit quantization (bitsandbytes) ...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_id,
            device_map="cuda:0",
            torch_dtype="auto",
            quantization_config=bnb_config,
        )
    else:
        if not args.force_fp16:
            print("bitsandbytes not available, falling back to fp16 ...")
        else:
            print("Using fp16 mode (--force_fp16) ...")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_id,
            device_map="cuda:0",
            torch_dtype=torch.float16,
        )

    model.eval()

    # Check device placement
    total_params = sum(p.numel() for p in model.parameters())
    cuda_params = sum(p.numel() for p in model.parameters() if p.is_cuda)
    cpu_params = total_params - cuda_params
    print(f"Model params: {total_params/1e6:.0f}M total, {cuda_params/1e6:.0f}M on GPU, {cpu_params/1e6:.0f}M on CPU")
    if cpu_params > 0:
        print("WARNING: Some params on CPU, inference will be slow!")

    if torch.cuda.is_available():
        mem_alloc = torch.cuda.memory_allocated() / 1024**3
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU memory: {mem_alloc:.1f}GB used / {mem_total:.1f}GB total")

    out_path = Path(args.out_preds)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f_out:
        for ex in tqdm(rows):
            image_path = ex["image"]
            gt_label = ex["label"]

            try:
                img = load_image(image_path, args.max_image_side)
            except Exception as e:
                record = {
                    "image": image_path,
                    "gt": gt_label,
                    "ok": False,
                    "pred": {"label": "warning", "error": f"image_load_error: {str(e)}"},
                    "raw": ""
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            prompt = (
                "Classify the traffic sign.\n"
                "Labels: [speed_limit, stop, yield, no_entry, warning, direction]\n"
                'Return only JSON: {"label":"stop"}'
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )

            inputs.pop("token_type_ids", None)

            for k, v in list(inputs.items()):
                if hasattr(v, "to"):
                    inputs[k] = v.to(model.device)

            with torch.inference_mode():
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )

            prompt_len = inputs["input_ids"].shape[-1]
            generated = gen_ids[0][prompt_len:]
            decoded = processor.decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            record = {
                "image": image_path,
                "gt": gt_label,
                "ok": False,
                "pred": {"label": "warning"},
                "raw": decoded[:1000]
            }

            try:
                pred_json = extract_json(decoded)
                pred_label = pred_json.get("label", "warning")

                if pred_label not in LABELS:
                    pred_label = "warning"

                record["pred"] = {"label": pred_label}
                record["ok"] = True

            except Exception as e:
                record["pred"] = {"label": "warning", "error": str(e)}

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
