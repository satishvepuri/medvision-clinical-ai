import json
import os
import mlflow

TRACKING_URI = "http://127.0.0.1:5000"
METRICS_FILE = r"artifacts\baseline_metrics.json"
PREDICTIONS_FILE = r"artifacts\baseline_predictions.csv"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("MedVision")

with open(METRICS_FILE, "r", encoding="utf-8") as f:
    metrics = json.load(f)

with mlflow.start_run(run_name="CheXpert_DenseNet121_Baseline"):

    mlflow.log_param(
        "model",
        "TorchXRayVision DenseNet121"
    )

    mlflow.log_param(
        "weights",
        "densenet121-res224-chex"
    )

    mlflow.log_param(
        "validation_rows",
        metrics["validation_rows_evaluated"]
    )

    mlflow.log_param(
        "dataset",
        "CheXpert Plus"
    )

    for label, values in metrics["labels_evaluated"].items():

        safe_label = (
            label.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        mlflow.log_metric(
            f"{safe_label}_accuracy",
            values["accuracy"]
        )

        mlflow.log_metric(
            f"{safe_label}_f1",
            values["f1"]
        )

        if values["roc_auc"] is not None:
            mlflow.log_metric(
                f"{safe_label}_auc",
                values["roc_auc"]
            )

    if "overall_label_accuracy" in metrics:
        mlflow.log_metric(
            "overall_label_accuracy",
            metrics["overall_label_accuracy"]
        )

    if "overall_macro_f1" in metrics:
        mlflow.log_metric(
            "overall_macro_f1",
            metrics["overall_macro_f1"]
        )

    mlflow.log_artifact(METRICS_FILE)

    if os.path.exists(PREDICTIONS_FILE):
        mlflow.log_artifact(PREDICTIONS_FILE)

    print("BASELINE LOGGED TO MLFLOW")
    print("Experiment: MedVision")
    print("Run: CheXpert_DenseNet121_Baseline")