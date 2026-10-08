import os
import time
import pandas as pd
import redivis

MANIFEST = r"data\processed\manifest.csv"
OUTPUT_ROOT = r"D:\medvision_data\images"

TRAIN_TABLE = "aimi.chexpert_plus:5yyj:v1_0.png_train:s6cj"
VALID_TABLE = "aimi.chexpert_plus:5yyj:v1_0.png_valid:41v9"

MAX_RETRIES = 3

os.makedirs(OUTPUT_ROOT, exist_ok=True)

df = pd.read_csv(MANIFEST)

train_table = redivis.table(TRAIN_TABLE)
valid_table = redivis.table(VALID_TABLE)

total = len(df)
downloaded = 0
skipped = 0
failed = 0

print("=" * 60)
print("MedVision 15K+ CheXpert Plus Downloader")
print("=" * 60)
print(f"Total images: {total}")
print(f"Saving to: {OUTPUT_ROOT}")
print()

for number, row in enumerate(df.itertuples(index=False), start=1):

    original_path = str(row.path_to_image)
    split = str(row.split)

    remote_path = (
        original_path
        .replace("train/", "", 1)
        .replace("valid/", "", 1)
        .replace(".jpg", ".png")
        .replace("/", "\\")
    )

    local_relative = original_path.replace(".jpg", ".png")
    local_path = os.path.join(OUTPUT_ROOT, local_relative)

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Resume support
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        skipped += 1
        if number % 10 == 0 or number == total:
            print(
                f"[{number}/{total}] "
                f"Downloaded={downloaded} "
                f"Skipped={skipped} "
                f"Failed={failed}"
            )
        continue

    if split == "train":
        table = train_table
    else:
        table = valid_table

    success = False

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            table.file(remote_path).download(local_path)

            if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                downloaded += 1
                success = True
                break

        except Exception as exc:
            print(
                f"[{number}/{total}] "
                f"Attempt {attempt}/{MAX_RETRIES} failed"
            )
            print(f"Path: {remote_path}")
            print(f"Error: {exc}")

            if attempt < MAX_RETRIES:
                time.sleep(2 * attempt)

    if not success:
        failed += 1
        print(f"FAILED: {remote_path}")

    if number % 10 == 0 or number == total:
        print(
            f"[{number}/{total}] "
            f"Downloaded={downloaded} "
            f"Skipped={skipped} "
            f"Failed={failed}"
        )

print()
print("=" * 60)
print("DOWNLOAD COMPLETE")
print("=" * 60)
print(f"Manifest rows : {total}")
print(f"Downloaded    : {downloaded}")
print(f"Skipped       : {skipped}")
print(f"Failed        : {failed}")
print(f"Output        : {OUTPUT_ROOT}")
print("=" * 60)