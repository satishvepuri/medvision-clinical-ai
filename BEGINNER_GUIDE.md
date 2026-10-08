# MedVision Beginner Guide

This guide assumes you are new to laptops, Python, Git, and AI projects.

## First: understand the four things you will use

**File Explorer** is where you see folders and files.

**Command Prompt** is the black window where you type commands.

**Python** runs the project code.

**GitHub** stores a copy of your code online.

You do not need to understand every line of code on day one.

## Step 1 — Extract the project

Download `medvision-clinical-ai.zip`.

Right-click the ZIP → **Extract All**.

Put the extracted folder in:

```text
C:\Users\sathi\Downloads\medvision-clinical-ai
```

## Step 2 — Open Command Prompt inside the folder

Open the `medvision-clinical-ai` folder in File Explorer.

Click the address bar at the top.

Type:

```text
cmd
```

Press Enter.

A black window opens.

You should see a line ending in:

```text
...\medvision-clinical-ai>
```

## Step 3 — Install the project

Type:

```cmd
setup_windows.bat
```

This creates a private Python environment named `.venv` and installs the packages.

This can take several minutes.

## Step 4 — Create demo data

Run:

```cmd
create_demo_data.bat
```

The demo images are synthetic. They are only for checking that files, OpenCV, and the data pipeline work.

They do NOT count toward the 15K medical-data resume claim.

## Step 5 — Check the project

Run:

```cmd
run_checks.bat
```

You want to see the project checks pass.

## Step 6 — Get the real dataset

Use Stanford's CheXpert Plus dataset.

You must accept the dataset's terms yourself. Do not upload the dataset to GitHub.

After downloading, place metadata, labels, and images under:

```text
data\raw
```

Then we will run the preparation script together.

## Step 7 — Why we start with tiny training

A vision-language model is much bigger than normal Python code.

We first train on 32 samples:

```cmd
.venv\Scripts\python scripts\train_vlm_lora.py --manifest data\processed\manifest.csv --limit 32 --epochs 1
```

If that works, we move to GPU training.

## Step 8 — What LoRA means

Imagine a huge textbook with millions of words.

Instead of rewriting the entire textbook, LoRA adds a small set of notes that teach the model your new task.

That is why LoRA is cheaper than full model training.

PEFT is the Hugging Face library that helps us add and train those small adapters.

## Step 9 — What MLflow means

MLflow is a notebook for experiments.

It records things such as:

- model name
- learning rate
- number of samples
- accuracy
- F1 score
- response-quality score

Start it with:

```cmd
start_mlflow.bat
```

## Step 10 — What FastAPI means

FastAPI turns your Python model into a small web service.

Start it with:

```cmd
start_api.bat
```

Then open:

```text
http://127.0.0.1:8000/docs
```

You can upload an image and send clinical text from that page.

## Step 11 — What Docker means

Docker puts the application and its dependencies into a repeatable container.

Build:

```cmd
docker build -t medvision .
```

Run:

```cmd
docker run --rm -p 8000:8000 medvision
```

## Step 12 — Before adding the resume claims

Run:

```cmd
.venv\Scripts\python scripts\verify_resume_claims.py
```

Do not claim work the verifier says has not been completed.
