import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torchxrayvision as xrv
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


IMAGE_ROOT = r"D:\medvision_data\images"
VALID_MANIFEST = r"data\processed\valid_manifest.csv"

LABEL_MAP = {
    "Atelectasis": "Atelectasis",
    "Consolidation": "Consolidation",
    "Pneumothorax": "Pneumothorax",
    "Edema": "Edema",
    "Pleural Effusion": "Effusion",
    "Pneumonia": "Pneumonia",
    "Cardiomegaly": "Cardiomegaly",
    "Lung Lesion": "Lung Lesion",
    "Fracture": "Fracture",
    "Lung Opacity": "Lung Opacity",
    "Enlarged Cardiomediastinum": "Enlarged Cardiomediastinum",
}


def image_path(metadata_path):
    return os.path.join(
        IMAGE_ROOT,
        str(metadata_path).replace(".jpg", ".png")
    )


def main(limit):
    os.makedirs("artifacts", exist_ok=True)

    df = pd.read_csv(VALID_MANIFEST)

    if limit > 0:
        df = df.head(limit)

    print("=" * 60)
    print("MedVision - Pretrained CheXpert Baseline")
    print("=" * 60)
    print(f"Validation rows: {len(df)}")

    model = xrv.models.DenseNet(
        weights="densenet121-res224-chex",
        apply_sigmoid=True,
    )

    model.eval()

    print("Baseline model loaded.")
    print("Evaluating images...")

    model_index = {
        name: i
        for i, name in enumerate(model.pathologies)
        if name
    }

    usable_labels = {
        chexpert: xrv_name
        for chexpert, xrv_name in LABEL_MAP.items()
        if xrv_name in model_index
    }

    print("Usable labels:")
    for chexpert, xrv_name in usable_labels.items():
        print(f"  {chexpert} -> {xrv_name}")

    predictions = []
    valid_rows = []

    transform_crop = xrv.datasets.XRayCenterCrop()
    transform_resize = xrv.datasets.XRayResizer(224)

    for count, (_, row) in enumerate(df.iterrows(), start=1):

        path = image_path(row["path_to_image"])

        if not os.path.exists(path):
            print(f"WARNING: missing image: {path}")
            continue

        try:
            img = xrv.utils.load_image(path)

            img = transform_crop(img)
            img = transform_resize(img)

            tensor = torch.from_numpy(img).float().unsqueeze(0)

            with torch.no_grad():
                output = model(tensor)[0].cpu().numpy()

            predictions.append(output)
            valid_rows.append(row)

        except Exception as exc:
            print(f"ERROR processing {path}: {exc}")

        if count % 25 == 0 or count == len(df):
            print(f"Processed {count}/{len(df)}")

    predictions = np.asarray(predictions)

    if len(predictions) == 0:
        raise RuntimeError("No images were successfully evaluated.")

    result_rows = []

    for row_index, row in enumerate(valid_rows):
        result = {
            "path_to_image": row.path_to_image
        }

        for chexpert_name, xrv_name in usable_labels.items():
            result[f"pred_{chexpert_name}"] = float(
                predictions[row_index][model_index[xrv_name]]
            )

        result_rows.append(result)

    pred_df = pd.DataFrame(result_rows)

    pred_df.to_csv(
        "artifacts/baseline_predictions.csv",
        index=False
    )

    metrics = {
        "model": "TorchXRayVision DenseNet121 CheXpert",
        "weights": "densenet121-res224-chex",
        "validation_rows_requested": len(df),
        "validation_rows_evaluated": len(valid_rows),
        "labels_evaluated": {},
    }

    all_true = []
    all_pred = []

    for chexpert_name, xrv_name in usable_labels.items():

        y_true = []
        y_score = []

        for row_index, row in enumerate(valid_rows):

            value = row[chexpert_name]

            if pd.isna(value):
                continue

            if float(value) not in (0.0, 1.0):
                continue

            y_true.append(int(value))
            y_score.append(
                float(
                    predictions[row_index][
                        model_index[xrv_name]
                    ]
                )
            )

        if len(y_true) == 0:
            continue

        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        y_pred = (y_score >= 0.5).astype(int)

        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        if len(np.unique(y_true)) == 2:
            auc = roc_auc_score(y_true, y_score)
        else:
            auc = None

        metrics["labels_evaluated"][chexpert_name] = {
            "n": int(len(y_true)),
            "accuracy": float(accuracy),
            "f1": float(f1),
            "roc_auc": None if auc is None else float(auc),
        }

        all_true.extend(y_true.tolist())
        all_pred.extend(y_pred.tolist())

        print(
            f"{chexpert_name}: "
            f"n={len(y_true)} "
            f"accuracy={accuracy:.4f} "
            f"F1={f1:.4f} "
            f"AUC={auc if auc is not None else 'N/A'}"
        )

    if all_true:
        metrics["overall_label_accuracy"] = float(
            accuracy_score(all_true, all_pred)
        )

        metrics["overall_macro_f1"] = float(
            f1_score(
                all_true,
                all_pred,
                average="macro",
                zero_division=0
            )
        )

    with open(
        "artifacts/baseline_metrics.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metrics, f, indent=2)

    print()
    print("=" * 60)
    print("BASELINE EVALUATION COMPLETE")
    print("=" * 60)
    print("Predictions: artifacts/baseline_predictions.csv")
    print("Metrics:     artifacts/baseline_metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of validation images. 0 = all."
    )

    args = parser.parse_args()

    main(args.limit)