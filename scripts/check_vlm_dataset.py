import json
import os

files = [
    (r"data\processed\train_vlm.jsonl", 15000),
    (r"data\processed\valid_vlm.jsonl", 234),
]

total = 0
bad = 0
missing = 0
unique_images = set()
target_lengths = []

for filename, expected in files:
    rows = 0

    print("Checking:", filename)

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            rows += 1

            try:
                record = json.loads(line)
            except Exception:
                bad += 1
                continue

            if "image" not in record or "messages" not in record:
                bad += 1
                continue

            image_path = record["image"]
            unique_images.add(image_path)

            if not os.path.exists(image_path):
                missing += 1

            messages = record["messages"]

            if len(messages) != 2:
                bad += 1
                continue

            if messages[0]["role"] != "user":
                bad += 1

            if messages[1]["role"] != "assistant":
                bad += 1

            try:
                target = messages[1]["content"][0]["text"]
                target_lengths.append(len(target.strip()))
            except Exception:
                bad += 1

    print("Rows:", rows)
    print("Expected:", expected)

    if rows != expected:
        bad += 1

    total += rows

print()
print("=" * 60)
print("VLM DATASET INTEGRITY CHECK")
print("=" * 60)
print("TOTAL ROWS:", total)
print("UNIQUE IMAGES:", len(unique_images))
print("MISSING IMAGE FILES:", missing)
print("BAD RECORDS:", bad)
print("EMPTY TARGETS:", sum(x == 0 for x in target_lengths))
print("MIN TARGET CHARS:", min(target_lengths))
print("MAX TARGET CHARS:", max(target_lengths))

if (
    total == 15234
    and len(unique_images) == 15234
    and missing == 0
    and bad == 0
    and sum(x == 0 for x in target_lengths) == 0
):
    print()
    print("STATUS: PASS")
else:
    print()
    print("STATUS: CHECK FAILED")
