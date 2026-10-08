from pathlib import Path
import csv
import math
import random

import cv2
import numpy as np

from src.medvision.constants import CHEXPERT_LABELS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "demo"
IMG_DIR = OUT / "images"
MANIFEST = OUT / "manifest.csv"

random.seed(42)
IMG_DIR.mkdir(parents=True, exist_ok=True)

rows = []
for i in range(64):
    img = np.zeros((256, 256), dtype=np.uint8)
    yy, xx = np.mgrid[:256, :256]
    chest = 35 + 75 * np.exp(-(((xx - 128) / 92) ** 2 + ((yy - 130) / 120) ** 2))
    lungs = (
        65 * np.exp(-(((xx - 88) / 44) ** 2 + ((yy - 125) / 75) ** 2))
        + 65 * np.exp(-(((xx - 168) / 44) ** 2 + ((yy - 125) / 75) ** 2))
    )
    img = np.clip(chest + lungs + np.random.default_rng(i).normal(0, 5, (256, 256)), 0, 255).astype(np.uint8)

    positive = []
    if i % 3 == 0:
        cv2.circle(img, (175, 170), 28, 185, -1)
        positive.append("Pleural Effusion")
    if i % 5 == 0:
        cv2.ellipse(img, (128, 175), (45, 28), 0, 0, 360, 170, -1)
        positive.append("Cardiomegaly")
    if not positive:
        positive.append("No Finding")

    path = IMG_DIR / f"demo_{i:04d}.png"
    cv2.imwrite(str(path), img)

    row = {
        "image_path": str(path),
        "clinical_text": "Synthetic educational example. No real patient data.",
        "reference_text": "Synthetic reference: " + ", ".join(positive),
        "split": "train" if i < 48 else ("val" if i < 56 else "test"),
    }
    for label in CHEXPERT_LABELS:
        row[label] = 1 if label in positive else 0
    rows.append(row)

fieldnames = ["image_path", "clinical_text", "reference_text", "split"] + CHEXPERT_LABELS
with MANIFEST.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Created {len(rows)} SYNTHETIC demo rows.")
print(f"Manifest: {MANIFEST}")
print("These rows are for software testing only and do NOT count toward the 15K medical-data milestone.")
