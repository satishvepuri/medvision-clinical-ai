import os
import pandas as pd
import redivis


ORIGINAL_MANIFEST = r"data\processed\manifest.csv"
RAW_METADATA = r"data\raw\df_chexpert_plus_240401.csv"
LABEL_FILE = r"data\raw\chexbert_labels\findings_fixed.json"

IMAGE_ROOT = r"D:\medvision_data\images"

TRAIN_OUTPUT = r"data\processed\train_multimodal_manifest.csv"
VALID_OUTPUT = r"data\processed\valid_multimodal_manifest.csv"

TABLE_REF = "aimi.chexpert_plus:5yyj:v1_0.png_train:s6cj"


LABELS = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding",
]


def clean_impression(value):
    if pd.isna(value):
        return ""

    value = str(value)

    # Remove literal \n text.
    value = value.replace("\\n", " ")

    return value.strip()


def local_image_path(dataset_path):
    path = str(dataset_path).replace("\\", "/")

    if path.startswith("train/"):
        path = path[len("train/"):]

    if path.startswith("valid/"):
        path = path[len("valid/"):]

    path = path.replace(".jpg", ".png")

    return os.path.join(
        IMAGE_ROOT,
        path
    )

print("=" * 60)
print("MedVision - Repair Training Manifest")
print("=" * 60)

print("Loading original manifest...")
original = pd.read_csv(ORIGINAL_MANIFEST)

print("Original rows:", len(original))

# IMPORTANT:
# These are the exact records whose images were already downloaded.
original_paths = set(
    original["path_to_image"].astype(str)
)

# Clean impressions in the original downloaded set.
original["section_impression"] = (
    original["section_impression"]
    .apply(clean_impression)
)

# Keep the already-downloaded training records
# that have a real impression.
train = original[
    (original["split"] == "train") &
    (original["section_impression"].str.len() > 0)
].copy()

valid = original[
    (original["split"] == "valid") &
    (original["section_impression"].str.len() > 0)
].copy()

print()
print("Existing clean training rows:", len(train))
print("Existing validation rows:", len(valid))

if len(train) != 14999:
    raise RuntimeError(
        f"Expected 14999 clean existing training rows, "
        f"but found {len(train)}"
    )

if len(valid) != 234:
    raise RuntimeError(
        f"Expected 234 validation rows, "
        f"but found {len(valid)}"
    )


# ---------------------------------------------------------
# Find ONE replacement training record.
# ---------------------------------------------------------

print()
print("Finding one replacement training record...")

metadata = pd.read_csv(RAW_METADATA)

labels = pd.read_json(
    LABEL_FILE,
    lines=True
)

full = metadata.merge(
    labels,
    on="path_to_image",
    how="inner"
)

full["section_impression"] = (
    full["section_impression"]
    .apply(clean_impression)
)

candidates = full[
    (full["split"] == "train") &
    (full["section_impression"].str.len() > 0) &
    (~full["path_to_image"].isin(original_paths))
].copy()

if len(candidates) == 0:
    raise RuntimeError(
        "No replacement training candidate found."
    )

candidate = candidates.sample(
    n=1,
    random_state=20261001
).iloc[0]

candidate_path = str(
    candidate["path_to_image"]
)

print()
print("Replacement selected:")
print(candidate_path)


# ---------------------------------------------------------
# Download ONLY the replacement image.
# ---------------------------------------------------------

table = redivis.table(TABLE_REF)

download_path = (
    candidate_path
    .replace("train/", "", 1)
    .replace(".jpg", ".png")
    .replace("/", "\\")
)

output_image = os.path.join(
    IMAGE_ROOT,
    download_path
)

os.makedirs(
    os.path.dirname(output_image),
    exist_ok=True
)

print()
print("Downloading ONE replacement image...")
print("Destination:", output_image)

table.file(download_path).download(
    output_image
)

if not os.path.exists(output_image):
    raise RuntimeError(
        "Replacement image download failed."
    )

print("Replacement image downloaded successfully.")


# ---------------------------------------------------------
# Build replacement row using the SAME columns
# as the original manifest.
# ---------------------------------------------------------

replacement = {}

for column in original.columns:

    if column in candidate.index:
        replacement[column] = candidate[column]
    else:
        replacement[column] = ""

replacement = pd.DataFrame(
    [replacement],
    columns=original.columns
)

replacement["section_impression"] = (
    replacement["section_impression"]
    .apply(clean_impression)
)

replacement["split"] = "train"


# Add the one replacement.
train_final = pd.concat(
    [train, replacement],
    ignore_index=True
)


# ---------------------------------------------------------
# Final verification.
# ---------------------------------------------------------

print()
print("Final training rows:", len(train_final))
print("Final validation rows:", len(valid))

if len(train_final) != 15000:
    raise RuntimeError(
        "Final training set is not exactly 15,000 rows."
    )

if len(valid) != 234:
    raise RuntimeError(
        "Final validation set is not exactly 234 rows."
    )


train_missing = [
    path
    for path in train_final["path_to_image"]
    if not os.path.exists(
        local_image_path(path)
    )
]

valid_missing = [
    path
    for path in valid["path_to_image"]
    if not os.path.exists(
        local_image_path(path)
    )
]


print()
print("Missing training images:", len(train_missing))
print("Missing validation images:", len(valid_missing))

if train_missing:
    print("First missing training image:")
    print(train_missing[0])
    raise RuntimeError(
        "Training image verification failed."
    )

if valid_missing:
    print("First missing validation image:")
    print(valid_missing[0])
    raise RuntimeError(
        "Validation image verification failed."
    )


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

print("Empty training impressions:", empty_train)
print("Empty validation impressions:", empty_valid)

if empty_train != 0 or empty_valid != 0:
    raise RuntimeError(
        "Empty impressions remain."
    )


# Save.
train_final.to_csv(
    TRAIN_OUTPUT,
    index=False
)

valid.to_csv(
    VALID_OUTPUT,
    index=False
)


print()
print("=" * 60)
print("REPAIR COMPLETE")
print("=" * 60)
print("Training rows:", len(train_final))
print("Validation rows:", len(valid))
print("Training missing images:", len(train_missing))
print("Validation missing images:", len(valid_missing))
print("Empty training impressions:", empty_train)
print("Empty validation impressions:", empty_valid)
print()
print("Train manifest:", TRAIN_OUTPUT)
print("Validation manifest:", VALID_OUTPUT)
print()
print("Only ONE new X-ray was downloaded.")