from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

checks = []

manifest = Path("data/processed/manifest.csv")
if manifest.exists():
    try:
        rows = len(pd.read_csv(manifest))
    except Exception:
        rows = 0
else:
    rows = 0
checks.append(("15K+ real prepared image-text rows", rows >= 15000, f"{rows} rows"))

adapter = Path("artifacts/qwen2_5_vl_3b_qlora/adapter_config.json")
checks.append(("LoRA/PEFT adapter saved", adapter.exists(), str(adapter)))

evaluation = Path("artifacts/evaluation.json")
eval_ok = False
eval_detail = "missing"
if evaluation.exists():
    try:
        data = json.loads(evaluation.read_text(encoding="utf-8"))
        eval_ok = all(k in data for k in ["base", "adapted"])
        eval_detail = "base/adapted results present" if eval_ok else "invalid structure"
    except Exception as exc:
        eval_detail = str(exc)
checks.append(("Base vs adapted evaluation saved", eval_ok, eval_detail))

mlflow_ok = Path("mlflow.db").exists()
checks.append(("MLflow experiment data exists", mlflow_ok, "mlflow.db" if mlflow_ok else "missing"))

api_ok = Path("app/main.py").exists()
checks.append(("FastAPI application exists", api_ok, "app/main.py"))

docker_ok = Path("Dockerfile").exists()
checks.append(("Docker packaging exists", docker_ok, "Dockerfile"))

llama_ok = Path("artifacts/llama_check.json").exists()
checks.append(("Llama-family stage executed", llama_ok, "artifacts/llama_check.json"))

print("MedVision resume-claim verification")
print("----------------------------------")
for label, ok, detail in checks:
    print(f"[{'PASS' if ok else 'NOT YET'}] {label} — {detail}")

core = checks[:6]
if all(ok for _, ok, _ in core):
    print("\nCORE RESUME CLAIMS VERIFIED.")
else:
    print("\nCORE RESUME CLAIMS ARE NOT FULLY VERIFIED YET.")

if not llama_ok:
    print("Do not list Llama as an executed component until scripts/check_llama.py succeeds.")
