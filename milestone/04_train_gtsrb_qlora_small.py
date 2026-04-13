# 04_train_gtsrb_qlora_small.py

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELS = ["speed_limit", "stop", "yield", "no_entry", "warning", "direction"]


@dataclass
class GTSRBCollator:
    processor: Any
    max_length: int = 256
    max_image_side: int = 224

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pad_id = self.processor.tokenizer.pad_token_id

        input_ids_list = []
        attn_list = []
        labels_list = []
        pixel_values_list = []
        image_grid_thw_list = []
        mm_tok_list = []

        for ex in batch:
            img = Image.open(ex["image"]).convert("RGB")
            img.thumbnail((self.max_image_side, self.max_image_side), Image.Resampling.LANCZOS)

            target_json = json.dumps({"label": ex["label"]}, ensure_ascii=False)

            prompt = (
                "Classify the traffic sign.\n"
                f"Labels: {LABELS}\n"
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

            prompt_inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            prompt_inputs.pop("token_type_ids", None)

            target_inputs = self.processor(
                text=target_json,
                return_tensors="pt",
                add_special_tokens=False,
            )

            input_ids = torch.cat(
                [prompt_inputs["input_ids"], target_inputs["input_ids"]],
                dim=-1
            )

            attention_mask = torch.ones_like(input_ids)

            labels = torch.full_like(input_ids, -100)
            labels[:, prompt_inputs["input_ids"].shape[-1]:] = target_inputs["input_ids"]

            prompt_mm = prompt_inputs["mm_token_type_ids"]
            tgt_len = target_inputs["input_ids"].shape[-1]
            target_mm = torch.zeros(
                (1, tgt_len),
                dtype=prompt_mm.dtype,
                device=prompt_mm.device,
            )
            mm_token_type_ids = torch.cat([prompt_mm, target_mm], dim=-1)

            if input_ids.shape[-1] > self.max_length:
                input_ids = input_ids[:, -self.max_length:]
                attention_mask = attention_mask[:, -self.max_length:]
                labels = labels[:, -self.max_length:]
                mm_token_type_ids = mm_token_type_ids[:, -self.max_length:]

            input_ids_list.append(input_ids.squeeze(0))
            attn_list.append(attention_mask.squeeze(0))
            labels_list.append(labels.squeeze(0))
            pixel_values_list.append(prompt_inputs["pixel_values"].squeeze(0))
            image_grid_thw_list.append(prompt_inputs["image_grid_thw"].squeeze(0))
            mm_tok_list.append(mm_token_type_ids.squeeze(0))

        max_len = max(x.shape[-1] for x in input_ids_list)

        def pad1d(x, val):
            if x.shape[-1] == max_len:
                return x
            return torch.cat(
                [x, torch.full((max_len - x.shape[-1],), val, dtype=x.dtype)],
                dim=0
            )

        input_ids = torch.stack([pad1d(x, pad_id) for x in input_ids_list])
        attention_mask = torch.stack([pad1d(x, 0) for x in attn_list])
        labels = torch.stack([pad1d(x, -100) for x in labels_list])
        mm_token_type_ids = torch.stack([pad1d(x, 0) for x in mm_tok_list])

        pixel_values = torch.stack(pixel_values_list)
        image_grid_thw = torch.stack(image_grid_thw_list)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "mm_token_type_ids": mm_token_type_ids,
            "labels": labels,
        }


def main():
    train_jsonl = str(REPO_ROOT / "artifacts/gtsrb_train_small.jsonl")
    val_jsonl = str(REPO_ROOT / "artifacts/gtsrb_val_small.jsonl")
    out_dir = str(REPO_ROOT / "artifacts/gtsrb_qwen3vl_lora_small")
    model_id = "Qwen/Qwen3-VL-2B-Instruct"

    print("Loading dataset...")
    ds = load_dataset("json", data_files={"train": train_jsonl, "val": val_jsonl})

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(model_id)

    print("Loading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        logging_steps=1,
        save_steps=20,
        save_total_limit=2,
        fp16=True,
        bf16=False,
        dataloader_num_workers=0,
        report_to="none",
        remove_unused_columns=False,
    )

    collator = GTSRBCollator(
        processor=processor,
        max_length=256,
        max_image_side=224,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        data_collator=collator,
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model(out_dir)
    processor.save_pretrained(out_dir)
    print(f"Saved adapter to: {out_dir}")


if __name__ == "__main__":
    main()