from __future__ import annotations

import json
import os
from pathlib import Path
from transformers import pipeline


DEFAULT_LLAMA = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def run_llama_check(
    model_name: str | None = None,
    output_path: str = "artifacts/llama_check.json",
) -> dict:
    """Run one small Llama-family generation to prove the optional stage works.

    Some Meta Llama checkpoints require accepting the model license on
    Hugging Face and logging in with `huggingface-cli login`.
    """
    model_name = model_name or os.getenv("MEDVISION_LLAMA_MODEL", DEFAULT_LLAMA)
    generator = pipeline(
        "text-generation",
        model=model_name,
        device_map="auto",
    )
    prompt = (
        "Rewrite this research-model output in plain language without giving "
        "medical advice: 'Possible pleural effusion. Clinical review required.'"
    )
    result = generator(prompt, max_new_tokens=60, do_sample=False)
    text = result[0]["generated_text"]
    record = {"model": model_name, "output": text}

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record
