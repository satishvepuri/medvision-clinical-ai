import json
import os

import pandas as pd


TRAIN_MANIFEST = r"data\processed\train_multimodal_manifest.csv"
VALID_MANIFEST = r"data\processed\valid_multimodal_manifest.csv"

IMAGE_ROOT = r"D:\medvision_data\images"
OUTPUT_DIR = r"data\processed"

TRAIN_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "train_vlm.jsonl"
)

VALID_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "valid_vlm.jsonl"
)

PROMPT = (
    "Analyze this chest X-ray and provide the radiology impression. "
    "Use concise clinical language. Do not invent findings that are "
    "not supported by the image."
)


def make_image_path(path):
    return os.path.join(
        IMAGE_ROOT,
        str(path).replace(".jpg", ".png")
    )


def build_dataset(input_file, output_file):

    df = pd.read_csv(input_file)

    written = 0
    skipped = 0

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        for _, row in df.iterrows():

            image_path = make_image_path(
                row["path_to_image"]
            )

            impression = row["section_impression"]

            if not os.path.exists(image_path):
                skipped += 1
                continue

            if pd.isna(impression):
                skipped += 1
                continue

            impression = str(impression).strip()

            if not impression:
                skipped += 1
                continue

            record = {
                "image": image_path,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image"
                            },
                            {
                                "type": "text",
                                "text": PROMPT
                            }
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": impression
                            }
                        ]
                    }
                ]
            }

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

            written += 1

    return written, skipped


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("=" * 60)
    print("MedVision VLM Dataset Builder")
    print("=" * 60)

    train_written, train_skipped = build_dataset(
        TRAIN_MANIFEST,
        TRAIN_OUTPUT
    )

    valid_written, valid_skipped = build_dataset(
        VALID_MANIFEST,
        VALID_OUTPUT
    )

    print()
    print("TRAIN")
    print("Written:", train_written)
    print("Skipped:", train_skipped)

    print()
    print("VALIDATION")
    print("Written:", valid_written)
    print("Skipped:", valid_skipped)

    print()
    print("Train file:", TRAIN_OUTPUT)
    print("Validation file:", VALID_OUTPUT)

    print()
    print("IMPORTANT:")
    print("The model input contains the image only.")
    print("The impression is the prediction target.")
    print("Clinical history, findings, summary, and comparison")
    print("are intentionally excluded to reduce leakage.")


if __name__ == "__main__":
    main()