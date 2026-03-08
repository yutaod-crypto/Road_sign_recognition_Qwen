import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from peft import PeftModel
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
        raise ValueError("No JSON found")
    return json.loads(m.group(0))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_jsonl", default="artifacts/gtsrb_test_120.jsonl")
    ap.add_argument("--out_preds", default="artifacts/gtsrb_preds_qlora.jsonl")
    ap.add_argument("--base_model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--adapter_dir", default="artifacts/gtsrb_qwen3vl_qlora")
    ap.add_argument("--max_image_side", type=int, default=384)
    ap.add_argument("--max_new_tokens", type=int, default=24)
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.eval_jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]

    processor = AutoProcessor.from_pretrained(args.base_model)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    base = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype="auto",
        quantization_config=bnb_config,
    )

    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    out_path = Path(args.out_preds)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f_out:
        for ex in tqdm(rows):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            img = load_image(ex["image"], args.max_image_side)
            gt_label = ex["label"]

            prompt = (
                "Identify the traffic sign in this image.\n"
                f"Choose exactly one label from: {LABELS}\n"
                'Return ONLY valid JSON like {"label":"stop"}\n'
                "Do not output extra text."
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
                clean_up_tokenization_spaces=False,
            )

            record = {
                "image": ex["image"],
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

    print(f"Saved QLoRA predictions to {out_path}")

if __name__ == "__main__":
    main()