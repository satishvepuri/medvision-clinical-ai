from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlflow
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

from src.medvision.data import load_manifest, build_user_prompt, build_training_target
from src.medvision.image_utils import load_medical_image

MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"


class ManifestDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx].to_dict()


class VLMDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        texts = []
        images = []
        for row in features:
            image = load_medical_image(row["image_path"])
            user_prompt = build_user_prompt(row["clinical_text"])
            target = build_training_target(row)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": target}],
                },
            ]
            text = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=False,
            )
            texts.append(text)
            images.append(image)

        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        labels = batch["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        batch["labels"] = labels
        return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output", default="artifacts/medvision-lora")
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()

    df = load_manifest(args.manifest)
    train = df[df["split"] == "train"].copy()
    if args.limit:
        train = train.head(args.limit)
    if train.empty:
        raise SystemExit("No training rows found.")

    print(f"Training rows: {len(train)}")
    print("CUDA available:", torch.cuda.is_available())

    processor = AutoProcessor.from_pretrained(args.model)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output / "trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ManifestDataset(train),
        data_collator=VLMDataCollator(processor),
    )

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("MedVision")
    with mlflow.start_run(run_name="smolvlm-lora"):
        mlflow.log_params({
            "base_model": args.model,
            "training_rows": len(train),
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "lora_r": 8,
            "lora_alpha": 16,
        })
        result = trainer.train()
        mlflow.log_metric("train_loss", float(result.training_loss))

    model.save_pretrained(output)
    processor.save_pretrained(output)
    print(f"\nLoRA adapter saved to: {output}")


if __name__ == "__main__":
    main()
