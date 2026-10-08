from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException

from src.medvision.inference import MedVisionEngine

app = FastAPI(
    title="MedVision Research API",
    version="0.1.0",
    description="Educational research demo only. Not for clinical use.",
)

engine = MedVisionEngine()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": os.getenv("MEDVISION_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct"),
        "clinical_use": False,
    }


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    clinical_text: str = Form(""),
):
    suffix = Path(image.filename or "image.png").suffix or ".png"
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image upload.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        output = engine.analyze(tmp_path, clinical_text)
        return {
            "output": output,
            "warning": (
                "Research/educational output only. Not validated for diagnosis, "
                "treatment, triage, or other clinical decisions."
            ),
        }
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
