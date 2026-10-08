import re
from typing import Iterable
from sklearn.metrics import f1_score, accuracy_score
from rouge_score import rouge_scorer

from .constants import CHEXPERT_LABELS


def labels_to_vector(labels: Iterable[str]) -> list[int]:
    selected = {str(x).strip().lower() for x in labels}
    return [1 if label.lower() in selected else 0 for label in CHEXPERT_LABELS]


def extract_predicted_labels(text: str) -> list[str]:
    low = (text or "").lower()
    found = []
    for label in CHEXPERT_LABELS:
        if label.lower() in low:
            found.append(label)
    return found


def multilabel_scores(y_true: list[list[int]], y_pred: list[list[int]]) -> dict:
    if not y_true:
        return {"subset_accuracy": 0.0, "micro_f1": 0.0, "macro_f1": 0.0}
    return {
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def rouge_l(reference: str, prediction: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return float(scorer.score(reference or "", prediction or "")["rougeL"].fmeasure)
