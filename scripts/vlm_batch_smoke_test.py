import torch

from datasets import load_dataset, Image as HFImage
from transformers import AutoProcessor
from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET = r"data/processed/train_vlm_pc_portable.jsonl"

print("=" * 60)
print("MedVision - VLM Batch/Collator Smoke Test")
print("=" * 60)

print("Loading dataset...")

dataset = load_dataset(
    "json",
    data_files=DATASET,
    split="train[:2]"
)

dataset = dataset.cast_column(
    "image",
    HFImage()
)

print("Dataset rows:", len(dataset))
print("Columns:", dataset.column_names)

print()
print("Loading processor...")

processor = AutoProcessor.from_pretrained(
    MODEL_ID
)

print("Processor loaded.")

print()
print("Creating VLM data collator...")

collator = DataCollatorForVisionLanguageModeling(
    processor=processor,
    max_length=None,
    completion_only_loss=True
)

examples = [
    dataset[i]
    for i in range(len(dataset))
]

print("Examples prepared:", len(examples))

print()
print("Building batch...")

batch = collator(examples)

print()
print("Batch keys:")

for key, value in batch.items():

    if hasattr(value, "shape"):
        print(
            f"  {key}: shape={tuple(value.shape)}, "
            f"dtype={value.dtype}"
        )
    else:
        print(
            f"  {key}: {type(value).__name__}"
        )

print()

input_ids = batch["input_ids"]
labels = batch["labels"]
pixel_values = batch["pixel_values"]

print("Input tokens:", input_ids.shape)
print("Labels:", labels.shape)
print("Pixel values:", pixel_values.shape)

masked = (labels == -100).sum().item()
trainable = (labels != -100).sum().item()

print()
print("Masked label tokens:", masked)
print("Trainable label tokens:", trainable)

print()
print("=" * 60)

if (
    input_ids.shape[0] == 2
    and labels.shape[0] == 2
    and pixel_values.shape[0] > 0
    and trainable > 0
    and masked > 0
):
    print("VLM BATCH SMOKE TEST: PASS")
else:
    print("VLM BATCH SMOKE TEST: FAILED")
    raise RuntimeError("Unexpected VLM batch structure")

print("=" * 60)
