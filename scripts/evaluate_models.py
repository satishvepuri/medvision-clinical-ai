from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import pandas as pd

from src.medvision.data import load_manifest, row_labels
from src.medvision.inference import MedVisionEngine
from src.medvision.metrics import (
    labels_to_vector,
    extract_predicted_labels,
    multilabel_scores,
    rouge_l,
)


def evaluate(engine: MedVisionEngine, df: pd.DataFrame) -> dict:
    y_true, y_pred, rouge_scores = [], [], []

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        output = engine.analyze(row["image_path"], row["clinical_text"])
        truth = row_labels(row)
        predicted = extract_predicted_labels(output)
        y_true.append(labels_to_vector(truth))
        y_pred.append(labels_to_vector(predicted))
        rouge_scores.append(rouge_l(str(row["reference_text"]), output))
        print(f"Evaluated {i}/{len(df)}")

    result = multilabel_scores(y_true, y_pred)
    result["response_rougeL"] = float(sum(rouge_scores) / max(len(rouge_scores), 1))
    result["samples"] = int(len(df))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--adapter", default="artifacts/medvision-lora")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", default="artifacts/evaluation.json")
    args = parser.parse_args()

    df = load_manifest(args.manifest)
    test = df[df["split"].isin(["val", "test"])].head(args.limit)
    if test.empty:
        raise SystemExit("No validation/test rows found.")

    print("\nEvaluating BASE model...")
    base = evaluate(MedVisionEngine(), test)

    print("\nEvaluating LoRA-ADAPTED model...")
    adapted = evaluate(MedVisionEngine(adapter_path=args.adapter), test)

    result = {
        "base": base,
        "adapted": adapted,
        "comparison": {
            "micro_f1_delta": adapted["micro_f1"] - base["micro_f1"],
            "rougeL_delta": adapted["response_rougeL"] - base["response_rougeL"],
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("MedVision")
    with mlflow.start_run(run_name="base-vs-lora-evaluation"):
        for prefix, block in [("base", base), ("adapted", adapted)]:
            for key, value in block.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"{prefix}_{key}", float(value))

    print("\nEvaluation saved to:", out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
