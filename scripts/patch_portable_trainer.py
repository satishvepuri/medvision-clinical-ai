from pathlib import Path

p = Path(r"scripts\train_qwen_vl_qlora.py")
text = p.read_text(encoding="utf-8")

text = text.replace(
    'TRAIN_FILE = r"data\\processed\\train_vlm_pc.jsonl"',
    'TRAIN_FILE = r"data\\processed\\train_vlm_pc_portable.jsonl"'
)

text = text.replace(
    'VALID_FILE = r"data\\processed\\valid_vlm_pc.jsonl"',
    'VALID_FILE = r"data\\processed\\valid_vlm_pc_portable.jsonl"'
)

text = text.replace(
    'MAX_PIXELS = 1280 * 28 * 28\n',
    'MAX_PIXELS = 1280 * 28 * 28\n\n'
    'IMAGE_ROOT = os.environ.get(\n'
    '    "MEDVISION_IMAGE_ROOT",\n'
    '    r"D:\\\\medvision_data\\\\images"\n'
    ')\n\n'
    'def resolve_images(dataset):\n'
    '    def resolve(example):\n'
    '        path = example["image"]\n'
    '        if not os.path.isabs(path):\n'
    '            path = os.path.join(\n'
    '                IMAGE_ROOT,\n'
    '                path.replace("/", os.sep)\n'
    '            )\n'
    '        return {"image": path}\n\n'
    '    return dataset.map(resolve)\n'
)

text = text.replace(
    '    dataset = dataset.cast_column(\n        "image",\n        HFImage(),\n    )',
    '    dataset = resolve_images(dataset)\n\n'
    '    dataset = dataset.cast_column(\n'
    '        "image",\n'
    '        HFImage(),\n'
    '    )'
)

text = text.replace(
    '    train_dataset = train_dataset.cast_column(\n        "image",\n        HFImage(),\n    )',
    '    train_dataset = resolve_images(train_dataset)\n'
    '\n'
    '    train_dataset = train_dataset.cast_column(\n'
    '        "image",\n'
    '        HFImage(),\n'
    '    )'
)

text = text.replace(
    '    valid_dataset = valid_dataset.cast_column(\n        "image",\n        HFImage(),\n    )',
    '    valid_dataset = resolve_images(valid_dataset)\n'
    '\n'
    '    valid_dataset = valid_dataset.cast_column(\n'
    '        "image",\n'
    '        HFImage(),\n'
    '    )'
)

p.write_text(text, encoding="utf-8")

print("Portable trainer patch applied.")
print("Train:", "data\\processed\\train_vlm_pc_portable.jsonl")
print("Valid:", "data\\processed\\valid_vlm_pc_portable.jsonl")
print("Image root:", r"D:\medvision_data\images")
