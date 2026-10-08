# MedVision — Multimodal Clinical AI Assistant

MedVision is an educational/research portfolio project that combines chest X-ray images and clinical text to produce structured findings and plain-language explanations.

**Important:** MedVision is **not a medical device**, is **not validated for diagnosis**, and must not be used to make clinical decisions.

## What this project demonstrates

- PyTorch + Hugging Face Transformers
- Multimodal vision-language modeling
- LoRA / PEFT parameter-efficient fine-tuning
- OpenCV image preprocessing
- CheXpert Plus image-report data preparation
- Base-vs-adapted model evaluation
- Classification accuracy and F1 metrics
- Response-quality scoring with ROUGE-L
- MLflow experiment tracking
- FastAPI inference API
- Docker packaging
- Optional Llama-family text explanation stage

## Architecture

```text
Chest X-ray + clinical text
          |
          v
  OpenCV preprocessing
          |
          v
SmolVLM-256M-Instruct
  + LoRA / PEFT adapter
          |
          +----> structured findings
          |
          +----> generated explanation
          |
          v
Optional Llama text rewriter
          |
          v
      FastAPI
          |
          v
   Docker container

Training + evaluation metrics ---> MLflow
```

The default vision-language model is `HuggingFaceTB/SmolVLM-256M-Instruct`, a small multimodal Transformers model. The project includes an optional Llama-family explanation stage for the resume technology stack.

## Dataset

The full project is designed for **CheXpert Plus**, which pairs chest X-rays with radiology reports and pathology annotations.

You download the dataset yourself from Stanford AIMI / Redivis and place the files under `data/raw/`. The repository does **not** redistribute medical images or reports.

The full resume milestone uses a patient-separated manifest of **15,000 image-text samples**.

## Beginner path

Do these in order:

1. `setup_windows.bat`
2. `create_demo_data.bat`
3. `run_checks.bat`
4. Download CheXpert Plus and prepare 15,000 real image-text pairs
5. Run a small local training test
6. Run full LoRA training on a GPU machine / Colab
7. Evaluate base vs adapted models
8. Run MLflow
9. Run FastAPI
10. Run the resume-claim verifier
11. Push to GitHub

Read `BEGINNER_GUIDE.md` for exact instructions.

## Quick setup on Windows

Open this project folder in File Explorer. Click the address bar, type `cmd`, and press Enter.

Then run:

```cmd
setup_windows.bat
```

Create harmless synthetic demo data:

```cmd
create_demo_data.bat
```

Check the project:

```cmd
run_checks.bat
```

## Prepare 15,000 CheXpert Plus samples

After downloading CheXpert Plus, place:

```text
data/raw/df_chexpert_plus_240401.csv
data/raw/chexbert_labels/findings_fixed.json
data/raw/images/...
```

Then run:

```cmd
.venv\Scripts\python scripts\prepare_chexpert_plus.py ^
  --metadata data\raw\df_chexpert_plus_240401.csv ^
  --labels data\raw\chexbert_labels\findings_fixed.json ^
  --images-root data\raw\images ^
  --limit 15000
```

This creates:

```text
data/processed/manifest.csv
data/processed/dataset_summary.json
```

The script refuses to claim 15,000 samples unless it can actually resolve 15,000 image files.

## Small LoRA training test

Use a small number first:

```cmd
.venv\Scripts\python scripts\train_vlm_lora.py --manifest data\processed\manifest.csv --limit 32 --epochs 1
```

For a real 15K training run, use a CUDA GPU and omit the small limit:

```cmd
.venv\Scripts\python scripts\train_vlm_lora.py --manifest data\processed\manifest.csv --limit 15000 --epochs 1
```

The adapter is saved under:

```text
artifacts/medvision-lora
```

## Evaluate base vs adapted model

Start small:

```cmd
.venv\Scripts\python scripts\evaluate_models.py --manifest data\processed\manifest.csv --limit 20
```

Then run a larger evaluation when your GPU budget allows it.

Results are written to:

```text
artifacts/evaluation.json
```

and logged to MLflow.

## MLflow

Start the local experiment dashboard:

```cmd
start_mlflow.bat
```

Then open:

```text
http://127.0.0.1:5000
```

## FastAPI

Start the API:

```cmd
start_api.bat
```

Open:

```text
http://127.0.0.1:8000/docs
```

The `/analyze` endpoint accepts:

- an image file
- optional clinical text

## Docker

Build:

```cmd
docker build -t medvision .
```

Run:

```cmd
docker run --rm -p 8000:8000 medvision
```

## Resume-claim verification

Run:

```cmd
.venv\Scripts\python scripts\verify_resume_claims.py
```

Only use the full resume bullets after the verifier confirms the required evidence.

## Resume wording after verification

**MedVision — Multimodal Clinical AI Assistant | PyTorch, Hugging Face Transformers, Llama, LoRA, PEFT, OpenCV, MLflow, FastAPI**

- Built a multimodal AI application combining medical text and chest X-ray images to classify findings, extract information, and generate explanations from 15K+ image-text samples using vision-language and transformer architectures.
- Fine-tuned a vision-language model using LoRA and PEFT, comparing pretrained and adapted models using classification accuracy, F1 score, and response-quality metrics; tracked experiments with MLflow and served inference through FastAPI and Docker.

Use the `15K+` wording only after `verify_resume_claims.py` confirms the dataset-size milestone.

## Repository layout

```text
medvision-clinical-ai/
├── app/
│   └── main.py
├── artifacts/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── scripts/
│   ├── check_environment.py
│   ├── check_llama.py
│   ├── create_demo_data.py
│   ├── evaluate_models.py
│   ├── prepare_chexpert_plus.py
│   ├── train_vlm_lora.py
│   └── verify_resume_claims.py
├── src/medvision/
│   ├── constants.py
│   ├── data.py
│   ├── image_utils.py
│   ├── inference.py
│   ├── llama_explainer.py
│   └── metrics.py
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```

## Safety and limitations

This project is for learning and portfolio demonstration only. Generated findings can be wrong, incomplete, or misleading. Do not use MedVision for diagnosis, treatment, triage, or any other clinical decision.
