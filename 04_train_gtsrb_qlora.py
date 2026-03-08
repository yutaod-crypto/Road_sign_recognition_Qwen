import json
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from PIL import Image
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
)

LABELS = ["speed_limit", "stop", "yield", "no_entry", "warning", "direction"]


@dataclass
class GTSRBCollator:
    processor: Any
    max_length: int = 768
    max_image_side: int = 384

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pad_id = self.processor.tokenizer.pad_token_id

        input_ids_list = []
        attn_list = []
        labels_list = []
        pixel_values_list = []

        for ex in batch:
            img = Image.open(ex["image"]).convert("RGB")
            img.thumbnail((self.max_image_side, self.max_image_side), Image.Resampling.LANCZOS)

            target_label = ex["label"]
            target_json = json.dumps({"label": target_label}, ensure_ascii=False)

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

            input_ids = torch.cat([prompt_inputs["input_ids"], target_inputs["input_ids"]], dim=-1)
            attention_mask = torch.ones_like(input_ids)

            labels = torch.full_like(input_ids, -100)
            labels[:, prompt_inputs["input_ids"].shape[-1]:] = target_inputs["input_ids"]

            if "pixel_values" in prompt_inputs:
                pixel_values = prompt_inputs["pixel_values"]
            else:
                pixel_values = self.processor(images=img, return_tensors="pt")["pixel_values"]

            if input_ids.shape[-1] > self.max_length:
                input_ids = input_ids[:, -self.max_length:]
                attention_mask = attention_mask[:, -self.max_length:]
                labels = labels[:, -self.max_length:]

            input_ids_list.append(input_ids.squeeze(0))
            attn_list.append(attention_mask.squeeze(0))
            labels_list.append(labels.squeeze(0))
            pixel_values_list.append(pixel_values.squeeze(0))

        max_len = max(x.shape[-1] for x in input_ids_list)

        def pad1d(x, val):
            if x.shape[-1] == max_len:
                return x
            return torch.cat([x, torch.full((max_len - x.shape[-1],), val, dtype=x.dtype)], dim=0)

        input_ids = torch.stack([pad1d(x, pad_id) for x in input_ids_list])
        attention_mask = torch.stack([pad1d(x, 0) for x in attn_list])
        labels = torch.stack([pad1d(x, -100) for x in labels_list])
        pixel_values = torch.stack(pixel_values_list)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
        }


def main():
    train_jsonl = "artifacts/gtsrb_train_grouped.jsonl"
    val_jsonl = "artifacts/gtsrb_val_grouped.jsonl"
    out_dir = "artifacts/gtsrb_qwen3vl_qlora"
    model_id = "Qwen/Qwen3-VL-2B-Instruct"

    ds = load_dataset("json", data_files={"train": train_jsonl, "val": val_jsonl})

    processor = AutoProcessor.from_pretrained(model_id)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype="auto",
    )

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        fp16=True,
        bf16=False,
        dataloader_num_workers=0,
        report_to="none",
        remove_unused_columns=False,
        evaluation_strategy="steps",
        eval_steps=200,
    )

    collator = GTSRBCollator(
        processor=processor,
        max_length=768,
        max_image_side=384,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(out_dir)
    processor.save_pretrained(out_dir)

    print(f"Saved adapter to: {out_dir}")


if __name__ == "__main__":
    main()
    