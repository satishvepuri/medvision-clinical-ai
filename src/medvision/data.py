from pathlib import Path
import json
import pandas as pd

from .constants import CHEXPERT_LABELS


REQUIRED_COLUMNS = ["image_path", "clinical_text", "reference_text", "split"]


def load_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")
    return df


def row_labels(row) -> list[str]:
    return [
        label for label in CHEXPERT_LABELS
        if int(float(row.get(label, 0) or 0)) == 1
    ]


def build_user_prompt(clinical_text: str) -> str:
    if clinical_text is None:
        clinical_text = ""
    elif pd.isna(clinical_text):
        clinical_text = ""
    else:
        clinical_text = str(clinical_text)

    clinical_text = clinical_text.strip() or "No clinical history provided."
    return (
        "You are analyzing a chest X-ray for an educational research benchmark. "
        "Use the image and clinical text. Return concise structured findings and "
        "an explanation. Do not claim certainty and do not give treatment advice.\n\n"
        f"Clinical text: {clinical_text}"
    )


def build_training_target(row) -> str:
    labels = row_labels(row)
    payload = {
        "findings": labels,
        "explanation": str(row.get("reference_text", "") or "").strip(),
    }
    return json.dumps(payload, ensure_ascii=False)
