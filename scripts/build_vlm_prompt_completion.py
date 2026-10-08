import json
import os

INPUTS = [
    (
        r"data\processed\train_vlm.jsonl",
        r"data\processed\train_vlm_pc.jsonl"
    ),
    (
        r"data\processed\valid_vlm.jsonl",
        r"data\processed\valid_vlm_pc.jsonl"
    ),
]


for input_file, output_file in INPUTS:

    written = 0
    skipped = 0

    with open(input_file, "r", encoding="utf-8") as src, \
         open(output_file, "w", encoding="utf-8") as dst:

        for line in src:

            record = json.loads(line)

            image = record["image"]
            messages = record["messages"]

            if not os.path.exists(image):
                skipped += 1
                continue

            user_message = messages[0]
            assistant_message = messages[1]

            prompt = [
                {
                    "role": "user",
                    "content": user_message["content"]
                }
            ]

            completion = [
                {
                    "role": "assistant",
                    "content": assistant_message["content"]
                }
            ]

            new_record = {
                "image": image,
                "prompt": prompt,
                "completion": completion
            }

            dst.write(
                json.dumps(
                    new_record,
                    ensure_ascii=False
                ) + "\n"
            )

            written += 1

    print()
    print("Input:", input_file)
    print("Output:", output_file)
    print("Written:", written)
    print("Skipped:", skipped)

print()
print("=" * 60)
print("PROMPT-COMPLETION VLM DATASET CREATED")
print("=" * 60)
