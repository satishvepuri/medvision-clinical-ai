import os
import pandas as pd

RAW_METADATA = r"data\raw\df_chexpert_plus_240401.csv"
LABEL_FILE = r"data\raw\chexbert_labels\findings_fixed.json"
OUTPUT_DIR = r"data\processed"

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

TEXT_FIELDS = [
    "section_narrative",
    "section_clinical_history",
    "section_history",
    "section_comparison",
    "section_technique",
    "section_procedure_comments",
    "section_findings",
    "section_impression",
    "section_end_of_impression",
    "section_summary",
]

BASE_FIELDS = [
    "path_to_image",
    "report",
    "split",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading metadata...")
metadata = pd.read_csv(RAW_METADATA)

print("Loading labels...")
labels = pd.read_json(LABEL_FILE, lines=True)

print("Merging metadata and labels...")
df = metadata.merge(
    labels,
    on="path_to_image",
    how="inner"
)

print("Merged rows:", len(df))


# ---------------------------------------------------------
# Clean impression text.
# Treat blank text and literal "\n" as missing.
# ---------------------------------------------------------

def clean_text(value):
    if pd.isna(value):
        return ""

    value = str(value)

    # Convert literal backslash-n into an empty line.
    value = value.replace("\\n", " ")

    return value.strip()


df["_clean_impression"] = df["section_impression"].apply(
    clean_text
)

df["section_impression"] = df["_clean_impression"]

df = df.drop(
    columns=["_clean_impression"]
)


# ---------------------------------------------------------
# Create the exact training set.
# ---------------------------------------------------------

train_candidates = df[
    (df["split"] == "train") &
    (df["section_impression"].str.len() > 0)
]

valid_candidates = df[
    (df["split"] == "valid") &
    (df["section_impression"].str.len() > 0)
]

print("Valid train candidates:", len(train_candidates))
print("Valid validation candidates:", len(valid_candidates))


# Reproduce our 15K training design,
# but now exclude empty impressions.
train = train_candidates.sample(
    n=15000,
    random_state=42
)

valid = valid_candidates.copy()


columns = BASE_FIELDS + TEXT_FIELDS + LABELS

columns = [
    c for c in columns
    if c in df.columns
]

train_out = train[columns].copy()
valid_out = valid[columns].copy()


train_path = os.path.join(
    OUTPUT_DIR,
    "train_multimodal_manifest.csv"
)

valid_path = os.path.join(
    OUTPUT_DIR,
    "valid_multimodal_manifest.csv"
)


train_out.to_csv(
    train_path,
    index=False
)

valid_out.to_csv(
    valid_path,
    index=False
)


print()
print("=" * 60)
print("MULTIMODAL MANIFESTS REBUILT")
print("=" * 60)

print("Train rows:", len(train_out))
print("Validation rows:", len(valid_out))

print()
print("Train:", train_path)
print("Validation:", valid_path)

print()
print("Empty train impressions:",
      train_out["section_impression"]
      .fillna("")
      .astype(str)
      .str.strip()
      .eq("")
      .sum())

print("Empty validation impressions:",
      valid_out["section_impression"]
      .fillna("")
      .astype(str)
      .str.strip()
      .eq("")
      .sum())