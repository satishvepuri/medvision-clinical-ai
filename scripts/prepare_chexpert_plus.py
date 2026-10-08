from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.medvision.constants import CHEXPERT_LABELS


def patient_from_path(path: str) -> str:
    match = re.search(r"patient(\d+)", str(path), flags=re.I)
    return match.group(1) if match else str(path).split("/")[0]


def split_for_patient(patient_id: str) -> str:
    # Stable patient-level split without sklearn/random state dependence.
    value = sum((i + 1) * ord(ch) for i, ch in enumerate(str(patient_id))) % 100
    if value < 80:
        return "train"
    if value < 90:
        return "val"
    return "test"


def first_existing(row, names, default=""):
    for name in names:
        if name in row and pd.notna(row[name]):
            text = str(row[name]).strip()
            if text:
                return text
    return default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--limit", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/processed/manifest.csv")
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    labels_path = Path(args.labels)
    images_root = Path(args.images_root)

    if not metadata_path.exists():
        raise SystemExit(f"Metadata file not found: {metadata_path}")
    if not labels_path.exists():
        raise SystemExit(f"Label file not found: {labels_path}")
    if not images_root.exists():
        raise SystemExit(f"Images root not found: {images_root}")

    meta = pd.read_csv(metadata_path)
    labels = pd.read_json(labels_path, lines=True)

    if "path_to_image" not in meta.columns or "path_to_image" not in labels.columns:
        raise SystemExit("Both metadata and labels must contain path_to_image.")

    df = pd.merge(labels, meta, on="path_to_image", how="inner")

    # Prefer rows with genuine report text.
    if "section_findings" in df.columns:
        df = df[df["section_findings"].notna()]
    if len(df) < args.limit:
        raise SystemExit(
            f"Only {len(df)} usable metadata rows found, fewer than requested {args.limit}."
        )

    # Build actual image path and keep only files that truly exist locally.
    def resolve_image(rel):
        rel = str(rel).replace("\\", "/")
        candidates = [
            images_root / rel,
            images_root / Path(rel).name,
        ]
        for p in candidates:
            if p.exists():
                return str(p.resolve())
        return ""

    df["image_path"] = df["path_to_image"].map(resolve_image)
    df = df[df["image_path"] != ""].copy()

    if len(df) < args.limit:
        raise SystemExit(
            f"Only {len(df)} image files could be resolved. "
            f"Need {args.limit}. Check --images-root."
        )

    df = df.sample(n=args.limit, random_state=args.seed).reset_index(drop=True)

    df["clinical_text"] = df.apply(
        lambda r: first_existing(
            r,
            ["section_indication", "section_history", "section_clinical_history"],
            "No clinical history provided.",
        ),
        axis=1,
    )
    df["reference_text"] = df.apply(
        lambda r: " ".join(
            x for x in [
                first_existing(r, ["section_findings"], ""),
                first_existing(r, ["section_impression"], ""),
            ] if x
        ).strip(),
        axis=1,
    )

    df["patient_id"] = df["path_to_image"].map(patient_from_path)
    df["split"] = df["patient_id"].map(split_for_patient)

    for label in CHEXPERT_LABELS:
        if label not in df.columns:
            df[label] = 0
        # Educational baseline: positive=1; negative/uncertain/missing=0.
        df[label] = (pd.to_numeric(df[label], errors="coerce") == 1).astype(int)

    out_cols = [
        "image_path", "clinical_text", "reference_text",
        "split", "patient_id"
    ] + CHEXPERT_LABELS

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df[out_cols].to_csv(output, index=False)

    summary = {
        "requested_limit": args.limit,
        "saved_rows": int(len(df)),
        "split_counts": {k: int(v) for k, v in df["split"].value_counts().to_dict().items()},
        "unique_patients": int(df["patient_id"].nunique()),
        "source": "CheXpert Plus local user download",
    }
    summary_path = output.parent / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if len(df) >= 15000:
        print("\n15K DATASET MILESTONE VERIFIED.")
    else:
        print("\n15K milestone not reached.")


if __name__ == "__main__":
    main()
