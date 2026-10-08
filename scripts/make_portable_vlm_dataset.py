import json

INPUTS = [
    (
        r"data\processed\train_vlm_pc.jsonl",
        r"data\processed\train_vlm_pc_portable.jsonl",
    ),
    (
        r"data\processed\valid_vlm_pc.jsonl",
        r"data\processed\valid_vlm_pc_portable.jsonl",
    ),
]

for source, destination in INPUTS:
    count = 0

    with open(source, "r", encoding="utf-8") as src, \
         open(destination, "w", encoding="utf-8") as dst:

        for line in src:
            record = json.loads(line)

            path = record["image"]
            path = path.replace("\\", "/")

            marker = "/images/"
            if marker in path:
                path = path.split(marker, 1)[1]

            record["image"] = path

            dst.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

            count += 1

    print("Created:", destination)
    print("Rows:", count)

print()
print("=" * 60)
print("PORTABLE VLM DATASETS CREATED")
print("=" * 60)
