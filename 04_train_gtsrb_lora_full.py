import argparse
import json
import os
from dataclasses import dataclass
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
from transformers.trainer_utils import get_last_checkpoint

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


def parse_args():
    p = argparse.ArgumentParser(description="Full GTSRB LoRA fine-tuning for Qwen3-VL")
    p.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="must stay 1: each image yields different Qwen3-VL patch counts; cannot stack a micro-batch in this collator",
    )
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=-1,
        help="default auto: 1 if WORLD_SIZE>1 else 2 (keeps ~2 samples per optimizer step vs original1-GPU recipe)",
    )
    p.add_argument("--num_train_epochs", type=int, default=2)
    p.add_argument("--bf16", action="store_true", help="train in bf16 (faster on H100/H200)")
    p.add_argument("--fp16", action="store_true", help="force fp16 even if bf16 is available")
    p.add_argument("--dataloader_num_workers", type=int, default=0)
    p.add_argument("--no_gradient_checkpointing", action="store_true", help="disable checkpointing for speed if VRAM allows")
    return p.parse_args()


def main():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))

    args = parse_args()
    if args.gradient_accumulation_steps < 0:
        args.gradient_accumulation_steps = 1 if world_size > 1 else 2
    if args.per_device_train_batch_size != 1 or args.per_device_eval_batch_size != 1:
        raise ValueError(
            "per_device_*_batch_size must be 1 for this multimodal collator (variable vision tokens per image). "
            "Speed up with more GPUs (torchrun DDP) or bf16/TF32, not larger micro-batch."
        )

    train_jsonl = "artifacts/gtsrb_train_full.jsonl"
    val_jsonl = "artifacts/gtsrb_val_full.jsonl"
    out_dir = "artifacts/gtsrb_qwen3vl_lora_full"
    model_id = "Qwen/Qwen3-VL-2B-Instruct"
    if world_size > 1:
        if local_rank < 0:
            raise RuntimeError("Multi-GPU requires torchrun (LOCAL_RANK not set).")
        torch.cuda.set_device(local_rank)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    use_bf16 = args.bf16 or (not args.fp16 and torch.cuda.is_bf16_supported())
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16

    print("Loading dataset...")
    ds = load_dataset("json", data_files={"train": train_jsonl, "val": val_jsonl})

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(model_id)

    print(f"Loading model... (world_size={world_size}, dtype={torch_dtype})")
    load_kw: Dict[str, Any] = {"torch_dtype": torch_dtype}
    if world_size <= 1:
        load_kw["device_map"] = "auto"
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **load_kw)
    if world_size > 1:
        model = model.to(torch.device(f"cuda:{local_rank}"))

    if not args.no_gradient_checkpointing:
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

    ta_kwargs: Dict[str, Any] = dict(
        output_dir=out_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=1e-4,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        fp16=not use_bf16,
        bf16=use_bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        report_to="none",
        remove_unused_columns=False,
    )
    #仅多进程 DDP 时设置，否则 accelerate 会走分布式分支导致单卡报错
    if world_size > 1:
        ta_kwargs["ddp_find_unused_parameters"] = False
        ta_kwargs["ddp_backend"] = "nccl"
    training_args = TrainingArguments(**ta_kwargs)

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

    resume_ckpt = get_last_checkpoint(out_dir)
    if resume_ckpt:
        print(f"Resuming from checkpoint: {resume_ckpt}")
    else:
        print("Starting training from scratch...")
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(out_dir)
    processor.save_pretrained(out_dir)

    print(f"Saved adapter to: {out_dir}")

if __name__ == "__main__":
    main()