import os
import pandas as pd

ORIGINAL = r"data\processed\manifest.csv"
RAW_METADATA = r"data\raw\df_chexpert_plus_240401.csv"
LABEL_FILE = r"data\raw\chexbert_labels\findings_fixed.json"

IMAGE_ROOT = r"D:\medvision_data\images"

TRAIN_OUT = r"data\processed\train_multimodal_manifest.csv"
VALID_OUT = r"data\processed\valid_multimodal_manifest.csv"

REPLACEMENT = "train/patient03617/study2/view1_frontal.jpg"


def clean(value):
    if pd.isna(value):
        return ""
    return str(value).replace("\\n", " ").strip()


def image_path(path):
    return os.path.join(
        IMAGE_ROOT,
        str(path).replace("/", os.sep).replace(".jpg", ".png")
    )


print("=" * 60)
print("MedVision - Finalize VLM Manifests")
print("=" * 60)

original = pd.read_csv(ORIGINAL)
original["section_impression"] = original["section_impression"].apply(clean)

train = original[
    (original["split"] == "train") &
    (original["section_impression"].str.len() > 0)
].copy()

valid = original[
    (original["split"] == "valid") &
    (original["section_impression"].str.len() > 0)
].copy()

print("Existing clean training rows:", len(train))
print("Validation rows:", len(valid))

if len(train) != 14999:
    raise RuntimeError(f"Expected 14999 clean training rows, got {len(train)}")

if len(valid) != 234:
    raise RuntimeError(f"Expected 234 validation rows, got {len(valid)}")


metadata = pd.read_csv(RAW_METADATA)
labels = pd.read_json(LABEL_FILE, lines=True)

full = metadata.merge(
    labels,
    on="path_to_image",
    how="inner"
)

full["section_impression"] = full["section_impression"].apply(clean)

matches = full[
    full["path_to_image"] == REPLACEMENT
]

if len(matches) != 1:
    raise RuntimeError(
        f"Expected exactly one replacement record, found {len(matches)}"
    )

candidate = matches.iloc[0]

print("Replacement:", candidate["path_to_image"])

replacement = {}

for column in original.columns:
    replacement[column] = candidate[column] if column in candidate.index else ""

replacement = pd.DataFrame(
    [replacement],
    columns=original.columns
)

replacement["split"] = "train"
replacement["section_impression"] = replacement["section_impression"].apply(clean)

replacement_file = image_path(REPLACEMENT)

print("Replacement image:", replacement_file)

if not os.path.exists(replacement_file):
    raise RuntimeError(
        "Replacement image does not exist: " + replacement_file
    )

train_final = pd.concat(
    [train, replacement],
    ignore_index=True
)

print("Final training rows:", len(train_final))
print("Final validation rows:", len(valid))

train_missing = [
    p for p in train_final["path_to_image"]
    if not os.path.exists(image_path(p))
]

valid_missing = [
    p for p in valid["path_to_image"]
    if not os.path.exists(image_path(p))
]

empty_train = (
    train_final["section_impression"]
    .fillna("")
    .astype(str)
    .str.strip()
    .eq("")
    .sum()
)

empty_valid = (
    valid["section_impression"]
    .fillna("")
    .astype(str)
    .str.strip()
    .eq("")
    .sum()
)

print("Missing training images:", len(train_missing))
print("Missing validation images:", len(valid_missing))
print("Empty training impressions:", empty_train)
print("Empty validation impressions:", empty_valid)

if len(train_final) != 15000:
    raise RuntimeError("Training count is not 15000")

if len(valid) != 234:
    raise RuntimeError("Validation count is not 234")

if train_missing:
    raise RuntimeError("Training image verification failed")

if valid_missing:
    raise RuntimeError("Validation image verification failed")

if empty_train or empty_valid:
    raise RuntimeError("Empty impressions remain")

train_final.to_csv(TRAIN_OUT, index=False)
valid.to_csv(VALID_OUT, index=False)

print()
print("=" * 60)
print("FINAL MANIFEST VERIFICATION PASSED")
print("=" * 60)
print("Training rows:", len(train_final))
print("Validation rows:", len(valid))
print("Missing training images:", len(train_missing))
print("Missing validation images:", len(valid_missing))
print("Empty training impressions:", empty_train)
print("Empty validation impressions:", empty_valid)
print("Saved:", TRAIN_OUT)
print("Saved:", VALID_OUT)
