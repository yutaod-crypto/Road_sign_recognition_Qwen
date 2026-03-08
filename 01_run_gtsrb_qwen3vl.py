# 01_run_gtsrb_qwen3vl.py
#
# Fast baseline inference for GTSRB grouped traffic-sign classification
# using Qwen3-VL-2B in 4-bit on a Windows machine with an RTX 4060.
#
# Expected input JSONL format:
# {"image":"...absolute_or_relative_path_to_.ppm_or_other_image...","label":"stop","class_id":14}
#
# Example usage (Windows CMD, single line):
# set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# python 01_run_gtsrb_qwen3vl.py --eval_jsonl artifacts/gtsrb_test_120.jsonl --out_preds artifacts/gtsrb_preds_120.jsonl
#
# Then evaluate with:
# python 02_eval_gtsrb.py --preds_jsonl artifacts/gtsrb_preds_120.jsonl

import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

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
    ap.add_argument("--eval_jsonl", default="artifacts/gtsrb_test_120.jsonl")
    ap.add_argument("--out_preds", default="artifacts/gtsrb_preds_120.jsonl")
    ap.add_argument("--model_id", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--max_image_side", type=int, default=256)   # reduced for speed
    ap.add_argument("--max_new_tokens", type=int, default=16)    # reduced for speed
    args = ap.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.eval_jsonl).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    processor = AutoProcessor.from_pretrained(args.model_id)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype="auto",
        quantization_config=bnb_config,
    )
    model.eval()

    out_path = Path(args.out_preds)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f_out:
        for ex in tqdm(rows):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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

            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
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