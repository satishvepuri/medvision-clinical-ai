import json
import os

from datasets import load_dataset, Image as HFImage
from transformers import AutoProcessor
from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET = "data/processed/train_vlm_pc_portable.jsonl"

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28

print("=" * 60)
print("MedVision - Controlled Resolution VLM Smoke Test")
print("=" * 60)

print("Minimum pixels:", MIN_PIXELS)
print("Maximum pixels:", MAX_PIXELS)

dataset = load_dataset(
    "json",
    data_files=DATASET,
    split="train[:2]"
)

dataset = dataset.cast_column(
    "image",
    HFImage()
)

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    min_pixels=MIN_PIXELS,
    max_pixels=MAX_PIXELS
)

print("Processor loaded with controlled resolution.")

collator = DataCollatorForVisionLanguageModeling(
    processor=processor,
    max_length=None,
    completion_only_loss=True
)

examples = [dataset[i] for i in range(2)]

batch = collator(examples)

print()
print("Batch shapes:")

for key, value in batch.items():
    if hasattr(value, "shape"):
        print(f"  {key}: {tuple(value.shape)}")

print()

pixel_values = batch["pixel_values"]
labels = batch["labels"]

masked = (labels == -100).sum().item()
trainable = (labels != -100).sum().item()

print("Pixel values:", tuple(pixel_values.shape))
print("Masked label tokens:", masked)
print("Trainable label tokens:", trainable)

print()
print("=" * 60)

if trainable > 0 and pixel_values.shape[0] > 0:
    print("CONTROLLED RESOLUTION SMOKE TEST: PASS")
else:
    print("CONTROLLED RESOLUTION SMOKE TEST: FAILED")
    raise RuntimeError("Invalid batch")

print("=" * 60)
