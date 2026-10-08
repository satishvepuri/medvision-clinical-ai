import json
import os

from PIL import Image
from transformers import AutoProcessor

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET = r"data\processed\train_vlm.jsonl"

print("=" * 60)
print("MedVision - VLM Processor Smoke Test")
print("=" * 60)

print("Loading processor:", MODEL_ID)

processor = AutoProcessor.from_pretrained(
    MODEL_ID
)

print("Processor loaded.")

with open(DATASET, "r", encoding="utf-8") as f:
    record = json.loads(next(f))

image_path = record["image"]

print("Image:", image_path)
print("Exists:", os.path.exists(image_path))

image = Image.open(image_path).convert("RGB")

messages = record["messages"]

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False
)

print()
print("Chat template created.")
print("Prompt characters:", len(text))

inputs = processor(
    text=text,
    images=image,
    return_tensors="pt"
)

print()
print("Processor output:")
for key, value in inputs.items():
    if hasattr(value, "shape"):
        print(f"  {key}: {tuple(value.shape)}")
    else:
        print(f"  {key}: {type(value).__name__}")

print()
print("=" * 60)
print("SMOKE TEST PASSED")
print("=" * 60)
