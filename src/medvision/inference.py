from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM
from peft import PeftModel

from .data import build_user_prompt
from .image_utils import load_medical_image


DEFAULT_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"


class MedVisionEngine:
    def __init__(
        self,
        model_name: Optional[str] = None,
        adapter_path: Optional[str] = None,
    ):
        self.model_name = model_name or os.getenv("MEDVISION_MODEL", DEFAULT_MODEL)
        self.adapter_path = adapter_path or os.getenv("MEDVISION_ADAPTER")
        self.processor = None
        self.model = None

    def load(self):
        if self.model is not None:
            return self

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if self.adapter_path and Path(self.adapter_path).exists():
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
        self.model.eval()
        return self

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.inference_mode()
    def analyze(self, image_path: str, clinical_text: str = "", max_new_tokens: int = 180) -> str:
        self.load()
        image = load_medical_image(image_path)
        prompt = build_user_prompt(clinical_text)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }]

        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        prompt_len = inputs["input_ids"].shape[-1]
        generated = output[:, prompt_len:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
