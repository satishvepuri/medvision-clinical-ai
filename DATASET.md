# Dataset Notes

## Primary dataset: CheXpert Plus

MedVision is designed around CheXpert Plus because it provides paired chest X-ray images and radiology-report text at large scale.

Official dataset page:

https://aimi.stanford.edu/datasets/chexpert-plus

Official project repository:

https://github.com/Stanford-AIMI/chexpert-plus

The MedVision repository does not redistribute the dataset.

## Expected local files

The preparation script expects paths similar to:

```text
data/raw/df_chexpert_plus_240401.csv
data/raw/chexbert_labels/findings_fixed.json
data/raw/images/train/patient.../study.../view...jpg
```

If your downloaded folder names differ, pass the correct paths to the script.

## Data split

The preparation script attempts a patient-level split when a patient identifier can be found, reducing patient leakage across train/validation/test sets.

## Label policy

Positive label (`1`) is treated as present.

Negative, uncertain, or missing labels are mapped to `0` in this educational baseline. This is a simplifying modeling choice, not a clinical truth.
