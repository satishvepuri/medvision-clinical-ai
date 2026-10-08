import pandas as pd

TRAIN = r"data\processed\train_multimodal_manifest.csv"
VALID = r"data\processed\valid_multimodal_manifest.csv"
RAW = r"data\raw\df_chexpert_plus_240401.csv"

print("=" * 60)
print("MedVision - Patient Leakage Audit")
print("=" * 60)

train = pd.read_csv(TRAIN)
valid = pd.read_csv(VALID)
raw = pd.read_csv(RAW)

print("Training rows:", len(train))
print("Validation rows:", len(valid))

# Build a direct image-path -> patient-ID mapping.
patient_map = (
    raw[["path_to_image", "deid_patient_id"]]
    .drop_duplicates("path_to_image")
    .set_index("path_to_image")["deid_patient_id"]
)

train["patient_id"] = train["path_to_image"].map(patient_map)
valid["patient_id"] = valid["path_to_image"].map(patient_map)

missing_train_ids = train["patient_id"].isna().sum()
missing_valid_ids = valid["patient_id"].isna().sum()

train_patients = set(
    train["patient_id"].dropna().astype(str)
)

valid_patients = set(
    valid["patient_id"].dropna().astype(str)
)

overlap = train_patients & valid_patients

train_duplicates = train["path_to_image"].duplicated().sum()
valid_duplicates = valid["path_to_image"].duplicated().sum()

path_overlap = (
    set(train["path_to_image"]) &
    set(valid["path_to_image"])
)

print()
print("Training patients:", len(train_patients))
print("Validation patients:", len(valid_patients))
print("Patients in BOTH splits:", len(overlap))

print()
print("Missing training patient IDs:", missing_train_ids)
print("Missing validation patient IDs:", missing_valid_ids)

print()
print("Duplicate training image paths:", train_duplicates)
print("Duplicate validation image paths:", valid_duplicates)
print("Train/validation image-path overlap:", len(path_overlap))

print()
print("Target length statistics:")

lengths = (
    train["section_impression"]
    .fillna("")
    .astype(str)
    .str.len()
)

print("Minimum:", lengths.min())
print("Median:", lengths.median())
print("95th percentile:", lengths.quantile(0.95))
print("Maximum:", lengths.max())

print()

if len(overlap) == 0:
    print("PATIENT SPLIT STATUS: PASS")
else:
    print("PATIENT SPLIT STATUS: WARNING")
    print()
    print("Number of overlapping patients:", len(overlap))
    print("First overlapping patient IDs:")

    for patient in sorted(overlap)[:20]:
        print(" ", patient)