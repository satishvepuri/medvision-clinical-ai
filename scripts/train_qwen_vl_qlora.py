import argparse
import os
from pathlib import Path

import torch


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

TRAIN_FILE = "data/processed/train_vlm_pc_portable.jsonl"
VALID_FILE = "data/processed/valid_vlm_pc_portable.jsonl"

OUTPUT_DIR = "artifacts/qwen2_5_vl_3b_qlora"

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28

IMAGE_ROOT = os.environ.get(
    "MEDVISION_IMAGE_ROOT",
    r"D:\\medvision_data\\images"
)

def resolve_images(dataset):
    def resolve(example):
        path = example["image"]
        if not os.path.isabs(path):
            path = os.path.join(
                IMAGE_ROOT,
                path.replace("/", os.sep)
            )
        return {"image": path}

    return dataset.map(resolve)


def check_only():
    from datasets import load_dataset, Image as HFImage
    from transformers import AutoProcessor
    from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling

    print("=" * 60)
    print("MedVision - QLoRA Trainer Configuration Check")
    print("=" * 60)

    print("CUDA available:", torch.cuda.is_available())
    print("Model:", MODEL_ID)
    print("Train:", TRAIN_FILE)
    print("Validation:", VALID_FILE)

    if not os.path.exists(TRAIN_FILE):
        raise RuntimeError("Training dataset not found.")

    if not os.path.exists(VALID_FILE):
        raise RuntimeError("Validation dataset not found.")

    print()
    print("Loading processor...")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    dataset = load_dataset(
        "json",
        data_files=TRAIN_FILE,
        split="train[:2]",
    )

    dataset = resolve_images(dataset)

    dataset = dataset.cast_column(
        "image",
        HFImage(),
    )

    collator = DataCollatorForVisionLanguageModeling(
        processor=processor,
        max_length=None,
        completion_only_loss=True,
    )

    batch = collator(
        [dataset[i] for i in range(2)]
    )

    print("Processor: OK")
    print("Dataset: OK")
    print("Vision collator: OK")
    print("Input IDs:", tuple(batch["input_ids"].shape))
    print("Pixel values:", tuple(batch["pixel_values"].shape))
    print("Labels:", tuple(batch["labels"].shape))

    from peft import LoraConfig

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    print()
    print("LoRA configuration: OK")
    print("  r =", lora_config.r)
    print("  alpha =", lora_config.lora_alpha)
    print("  dropout =", lora_config.lora_dropout)
    print("  targets =", lora_config.target_modules)

    print()
    print("=" * 60)
    print("CONFIGURATION CHECK: PASS")
    print("=" * 60)


def train(max_steps):

    # Hard safety check: don't accidentally train on the AMD/CPU laptop.
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for QLoRA training. "
            "This machine has no CUDA GPU, so training was stopped "
            "before downloading/loading the 3B model."
        )

    try:
        import bitsandbytes
    except ImportError:
        raise RuntimeError(
            "bitsandbytes is missing. Install it on the CUDA machine "
            "with: python -m pip install -U bitsandbytes"
        )

    from datasets import load_dataset, Image as HFImage
    from peft import (
        LoraConfig,
        prepare_model_for_kbit_training,
    )
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
    )
    from trl import SFTConfig, SFTTrainer
    from trl.trainer.sft_trainer import (
        DataCollatorForVisionLanguageModeling,
    )

    print("=" * 60)
    print("MedVision - Qwen2.5-VL QLoRA Training")
    print("=" * 60)

    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM GB:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            2,
        ),
    )

    use_bf16 = torch.cuda.is_bf16_supported()

    if use_bf16:
        compute_dtype = torch.bfloat16
        print("Precision: BF16")
    else:
        compute_dtype = torch.float16
        print("Precision: FP16")

    print()
    print("Loading datasets...")

    train_dataset = load_dataset(
        "json",
        data_files=TRAIN_FILE,
        split="train",
    )

    valid_dataset = load_dataset(
        "json",
        data_files=VALID_FILE,
        split="train",
    )

    train_dataset = resolve_images(train_dataset)

    train_dataset = train_dataset.cast_column(
        "image",
        HFImage(),
    )

    valid_dataset = resolve_images(valid_dataset)

    valid_dataset = valid_dataset.cast_column(
        "image",
        HFImage(),
    )

    print("Training rows:", len(train_dataset))
    print("Validation rows:", len(valid_dataset))

    print()
    print("Loading processor...")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    print("Processor loaded.")

    print()
    print("Creating 4-bit NF4 configuration...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    print("Loading quantized Qwen2.5-VL model...")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
        device_map="auto",
    )

    model.config.use_cache = False

    print("Model loaded.")

    print()
    print("Preparing model for k-bit training...")

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    print("Model prepared.")

    print()
    print("Creating LoRA configuration...")

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "v_proj",
        ],
        task_type="CAUSAL_LM",
    )

    print("LoRA configuration created.")

    collator = DataCollatorForVisionLanguageModeling(
        processor=processor,
        max_length=None,
        completion_only_loss=True,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    max_steps_value = max_steps if max_steps > 0 else -1

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,

        num_train_epochs=1,
        max_steps=max_steps_value,

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,

        gradient_accumulation_steps=8,

        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=50,

        optim="paged_adamw_8bit",

        weight_decay=0.01,

        bf16=use_bf16,
        fp16=not use_bf16,

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },

        max_length=None,
        completion_only_loss=True,

        remove_unused_columns=False,

        dataset_kwargs={
            "skip_prepare_dataset": True
        },

        logging_strategy="steps",
        logging_steps=10,
        logging_first_step=True,

        eval_strategy="steps",
        eval_steps=250,

        save_strategy="steps",
        save_steps=250,
        save_total_limit=2,

        report_to=["mlflow"],
        run_name="Qwen2.5-VL-3B-QLoRA-MedVision",

        seed=42,
        data_seed=42,

        dataloader_num_workers=2,
        dataloader_pin_memory=True,

        use_cache=False,
    )

    print()
    print("Training configuration created.")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=processor,
        data_collator=collator,
        peft_config=peft_config,
    )

    print()
    print("Trainable parameter summary:")
    trainer.model.print_trainable_parameters()

    print()
    print("=" * 60)

    if max_steps > 0:
        print("TRAINING TEST MODE")
        print("Maximum optimizer steps:", max_steps)
    else:
        print("FULL TRAINING MODE")
        print("Epochs: 1")

    print("=" * 60)

    trainer.train()

    print()
    print("Saving adapter...")

    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print("Output:", OUTPUT_DIR)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate configuration without loading model weights.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Limit optimizer steps. Use a small value for a GPU smoke test.",
    )

    args = parser.parse_args()

    if args.check_only:
        check_only()
    else:
        train(args.max_steps)


if __name__ == "__main__":
    main()
