import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

model_id = "Qwen/Qwen3-VL-2B-Instruct"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_id)
print("Processor loaded.")

print("Loading model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)
print("Model loaded.")
print("Done.")