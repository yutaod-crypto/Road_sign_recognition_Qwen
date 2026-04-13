"""Qwen2.5-VL helpers: completion log-probabilities for coarse label scoring (KD)."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration


def infer_model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def load_qwen25_vl(
    model_id: str,
    *,
    load_in_4bit: bool = False,
    device_map: str | dict | None = "auto",
    torch_dtype: torch.dtype | str | None = None,
) -> tuple[Qwen2_5_VLForConditionalGeneration, Any]:
    """Load processor + model (optional4-bit for large teachers)."""
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": device_map,
    }
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        kwargs["torch_dtype"] = torch.bfloat16
    elif torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    else:
        kwargs["torch_dtype"] = "auto"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
    return model, processor


def completion_token_logprob_sum(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: Any,
    *,
    prompt_inputs: dict,
    completion_text: str,
) -> torch.Tensor:
    """
    Sum of log p(token_t | prefix) over completion tokens (natural log).
    Single sequence, batch size 1.
    """
    device = infer_model_device(model)
    comp = processor(text=completion_text, return_tensors="pt", add_special_tokens=False)
    comp_ids = comp["input_ids"].to(device)
    comp_len = comp_ids.shape[1]
    if comp_len == 0:
        return torch.zeros((), device=device, dtype=torch.float32)

    input_ids = torch.cat([prompt_inputs["input_ids"].to(device), comp_ids], dim=-1)
    attention_mask = torch.cat(
        [prompt_inputs["attention_mask"].to(device), torch.ones_like(comp_ids)],
        dim=-1,
    )

    mm_key = "mm_token_type_ids"
    if mm_key in prompt_inputs:
        prompt_mm = prompt_inputs[mm_key].to(device)
        target_mm = torch.zeros((1, comp_len), dtype=prompt_mm.dtype, device=device)
        mm_token_type_ids = torch.cat([prompt_mm, target_mm], dim=-1)
    else:
        mm_token_type_ids = None

    pixel_values = prompt_inputs.get("pixel_values")
    if pixel_values is not None:
        pixel_values = pixel_values.to(device)
    image_grid_thw = prompt_inputs.get("image_grid_thw")
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.to(device)

    fwd = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        mm_token_type_ids=mm_token_type_ids,
        use_cache=False,
    )
    logits = fwd.logits  # [1, seq, vocab]
    logits = logits.float()

    prompt_len = prompt_inputs["input_ids"].shape[1]
    total_len = input_ids.shape[1]
    log_probs = torch.log_softmax(logits, dim=-1)

    total = input_ids.new_zeros((), dtype=torch.float32)
    for j in range(comp_len):
        pos = prompt_len - 1 + j
        tid = comp_ids[0, j]
        total = total + log_probs[0, pos, tid]
    return total


def score_label_completions(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: Any,
    *,
    messages: list[dict],
    completion_texts: list[str],
) -> torch.Tensor:
    """Stacked log-prob sums, shape [K]. Runs K forward passes."""
    prompt_inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    prompt_inputs.pop("token_type_ids", None)

    scores = []
    for text in completion_texts:
        scores.append(
            completion_token_logprob_sum(
                model,
                processor,
                prompt_inputs=prompt_inputs,
                completion_text=text,
            )
        )
    return torch.stack(scores)
