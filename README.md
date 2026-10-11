MedVision — Multimodal Clinical AI Assistant
MedVision explores how vision-language models can work with chest X-ray images and clinical text to produce structured findings and readable explanations.
I built this project to work through the practical challenges of a multimodal machine learning system: preparing image-report data, adapting a pretrained model, evaluating its outputs, and making inference available through an API.
Research use only: MedVision is not a medical device, has not been clinically validated, and must not be used for diagnosis, treatment, triage, or other clinical decisions.

Overview
The pipeline combines chest X-ray images with optional clinical text. After image preprocessing, a vision-language model generates findings and an explanation. An optional Llama-family stage can rewrite the explanation. The application exposes inference through FastAPI and can be packaged with Docker.
The project includes scripts for LoRA fine-tuning, evaluation against a base model, and tracking experiments with MLflow.
Architecture
Chest X-ray + optional clinical text
                 |
                 v
         Image preprocessing
             (OpenCV)
                 |
                 v
        Qwen2.5-VL-3B-Instruct
           + LoRA adapter
                 |
          +------+------+
          |             |
          v             v
   Structured       Generated
    findings        explanation
          |             |
          +------+------+
                 |
                 v
       Optional Llama rewriter
                 |
                 v
           FastAPI /analyze
                 |
                 v
          Docker packaging

Training and evaluation results ---> MLflow
The model integration uses Qwen2.5-VL-3B-Instruct with Hugging Face Transformers. Check the model identifier in your local configuration before running training or inference; the exact repository ID should match the weights you use.
Technology stack
Area	Tools
Model development	Python, PyTorch, Hugging Face Transformers
Multimodal modeling	Qwen2.5-VL-3B-Instruct
Fine-tuning	LoRA, PEFT
Image processing	OpenCV
Data	CheXpert Plus image-report pairs
Evaluation	Accuracy, F1, ROUGE-L
Experiment tracking	MLflow
Inference	FastAPI
Packaging	Docker
Optional text refinement	Llama-family model


Dataset
MedVision is designed to work with CheXpert Plus, a chest X-ray dataset with associated radiology reports and annotations.
The preparation script builds a manifest linking image files with the corresponding text and labels. It supports patient-separated data preparation to reduce leakage between training and evaluation.
The workflow can prepare up to 15,000 image-text pairs using the command below. This is a supported preparation target, not a published claim about completed training or evaluation results.
The original medical images and reports are not distributed in this repository. Obtain access through the dataset provider and place the required files under data/raw/.
Expected input structure:
data/raw/
├── df_chexpert_plus_240401.csv
├── chexbert_labels/
│   └── findings_fixed.json
└── images/
    └── ...
Prepare the manifest on Windows:
.venv\Scripts\python scripts\prepare_chexpert_plus.py ^
  --metadata data\raw\df_chexpert_plus_240401.csv ^
  --labels data\raw\chexbert_labels\findings_fixed.json ^
  --images-root data\raw\images ^
  --limit 15000
Generated files:
data/processed/manifest.csv
data/processed/dataset_summary.json
The preparation script checks that referenced image files can be resolved before including them in the manifest.
Getting started
These commands use Windows Command Prompt. Open a terminal in the repository root.
Set up the environment:
setup_windows.bat
Generate synthetic data for a quick pipeline check:
create_demo_data.bat
Run the project checks:
run_checks.bat
For a more detailed walkthrough, see [BEGINNER_GUIDE.md](BEGINNER_GUIDE.md). Synthetic demo data is for checking the software workflow, not for evaluating medical performance.
Fine-tuning
The training pipeline uses LoRA and PEFT to adapt the vision-language model while updating a smaller set of parameters than full-model fine-tuning.
Start with a small test run to check the data and training workflow:
.venv\Scripts\python scripts\train_vlm_lora.py --manifest data\processed\manifest.csv --limit 32 --epochs 1
For a larger run on a suitable CUDA GPU:
.venv\Scripts\python scripts\train_vlm_lora.py --manifest data\processed\manifest.csv --limit 15000 --epochs 1
The configured adapter output directory is:
artifacts/medvision-lora
The larger command is an example configuration; successful execution depends on available data, GPU memory, dependencies, and model settings.
Evaluation
The evaluation pipeline compares outputs from the pretrained model and the adapted model. It includes classification accuracy and F1 for structured predictions, as well as ROUGE-L for generated text.
Run a small evaluation:
.venv\Scripts\python scripts\evaluate_models.py --manifest data\processed\manifest.csv --limit 20
The evaluation output is written to:
artifacts/evaluation.json
Experiment metrics can also be logged to MLflow. The metrics should be interpreted in the context of the evaluation dataset and methodology; they are not evidence of clinical validity.
Experiment tracking
Start MLflow:
start_mlflow.bat
Open http://127.0.0.1:5000 to inspect logged runs and metrics.
Inference API
FastAPI provides an /analyze endpoint that accepts an image and optional clinical text.
Start the application:
start_api.bat
Interactive API documentation is available at:
http://127.0.0.1:8000/docs
The endpoint uses the configured inference pipeline. Model weights and adapters may need to be downloaded or made available locally before model-backed inference will work.
Docker
Build the application image:
docker build -t medvision .
Run it:
docker run --rm -p 8000:8000 medvision
Docker packages the application environment; access to model weights, adapters, and hardware resources still needs to be configured for the intended inference setup.
Repository structure
medvision-clinical-ai/
├── app/                  # FastAPI application
├── artifacts/            # Generated model and evaluation outputs
├── data/                 # Raw and processed dataset locations
├── notebooks/            # Exploration and experiments
├── scripts/              # Preparation, training, evaluation, and checks
├── src/medvision/        # Core preprocessing and inference modules
├── tests/                # Project tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── BEGINNER_GUIDE.md
├── DATASET.md
└── README.md
Limitations and next steps
Vision-language models can generate incorrect, incomplete, or misleading findings, particularly on inputs that differ from the data used during development. Text-generation metrics also do not establish medical correctness.
Areas worth improving include more extensive held-out evaluation, error analysis, uncertainty handling, inference reliability, and reproducibility across environments. Before making any performance claims, results should be supported by saved evaluation artifacts and clearly described test conditions.
Project scope
MedVision is an engineering and research project focused on multimodal model development and deployment. It is not intended to replace a radiologist or support real-world clinical decisions.
