# Business Issue Allocation

A text classification system that maps a natural language description of a business data problem to the most likely data engineering solution category.

---

## What It Does

A business user describes a problem in plain English. The system classifies it into one of nine solution categories:

| Label | Description |
|-------|-------------|
| `stream_processing` | Real-time event-driven processing |
| `etl_pipeline` | Scheduled data consolidation across systems |
| `data_warehouse` | Centralised analytical reporting store |
| `data_lake` | Raw data storage for future use |
| `api_integration` | System-to-system connectivity |
| `ml_feature_store` | Shared feature management for ML models |
| `data_caching` | Performance optimisation through stored results |
| `data_governance` | Data ownership, lineage and access policy |
| `data_quality` | Data correctness, completeness and trust |

**Example input:**
> "We receive data from 12 regional offices every Monday and need one consolidated report for the board."

**Predicted label:** `etl_pipeline`

---

## Live Demo

[Try the demo on Hugging Face Spaces](https://huggingface.co/spaces/dianamikova/business-issue-allocation-demo)

---

## Dataset and Model

| Resource | Link |
|----------|------|
| Dataset | [Hugging Face Dataset](https://huggingface.co/datasets/dianamikova/business-issue-allocation) |
| Model | [Hugging Face Model](https://huggingface.co/dianamikova/business-issue-allocation-classifier) |
| Demo | [Hugging Face Space](https://huggingface.co/spaces/dianamikova/business-issue-allocation-demo) |

---

## How It Works

1. A pretrained sentence embedding model (`all-mpnet-base-v2`) converts the input text into a dense vector
2. A trained SVM classifier predicts the solution category from that vector
3. The system returns the predicted label, confidence score and top predictions

The embedding model is not trained — it is used as a frozen feature extractor. Only the classifier is trained on the custom dataset.

---

## Project Structure

```
business-issue-allocation/
├── app.py                          # Gradio demo
├── requirements.txt                # Python dependencies
├── README.md
├── data/
│   └── raw/
│       └── data.csv                # Labeled dataset
├── artifacts/                      # Saved model files after training
│   ├── model.joblib
│   ├── label_encoder.joblib
│   ├── metrics.json
│   ├── all_results.json
│   └── task_config.json
├── scripts/
│   ├── train.py                    # Train classifiers
│   ├── evaluate.py                 # Print evaluation results
│   ├── predict.py                  # Single prediction from terminal
│   └── create_sample_dataset.py   # Copy dataset into project
├── src/
│   ├── config.py                   # Paths and model settings
│   ├── embeddings.py               # Text to vector encoding
│   ├── inference.py                # Prediction pipeline
│   ├── model_utils.py              # Save and load model
│   └── data_utils.py               # Load and preprocess data
└── report/
    └── REPORT_TEMPLATE.md
```

---

## Installation

**Requirements:** Python 3.10 or higher

```bash
git clone https://github.com/YOUR_USERNAME/business-issue-allocation.git
cd business-issue-allocation

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Reproduce Training

### Step 1 — Add the dataset

Download `data.csv` from the [Hugging Face dataset page](https://huggingface.co/datasets/dianamikova/business-issue-allocation) and place it at:

```
data/raw/data.csv
```

Or copy your own CSV using the helper script:

```bash
python3 scripts/create_sample_dataset.py --source "path/to/your/data.csv"
```

### Step 2 — Train the classifiers

```bash
python3 scripts/train.py --classifier logreg --target-column label_it
python3 scripts/train.py --classifier svm --target-column label_it
python3 scripts/train.py --classifier random_forest --target-column label_it
```

### Step 3 — Evaluate results

```bash
python3 scripts/evaluate.py
```

### Step 4 — Test a single prediction

```bash
python3 scripts/predict.py "We need to detect fraudulent transactions the moment they happen."
```

---

## Run the Demo Locally

```bash
python3 app.py
```

Open the local Gradio URL shown in the terminal.

---

## Results

All classifiers were evaluated on a held-out test set (20% of the dataset, stratified split).

Embedding model: `sentence-transformers/all-mpnet-base-v2`

| Classifier | Accuracy | Macro F1 |
|-----------|----------|----------|
| SVM | **88.2%** | **88.4%** |
| Logistic Regression | 83.9% | 84.1% |
| Random Forest | 80.6% | 79.9% |

The SVM classifier was selected as the final model.

Embedding model comparison on SVM:

| Embedding Model | Accuracy | Macro F1 |
|----------------|----------|----------|
| all-mpnet-base-v2 | **88.2%** | **88.4%** |
| all-MiniLM-L6-v2 | 87.1% | 86.4% |

---

## Dataset

The dataset was created specifically for this project. It contains 465 labeled examples of business problem descriptions across 9 solution categories.

- Labels: `label_it` (technical) and `label_bus` (business-friendly)
- Format: `query_id`, `query`, `label_it`, `label_bus`
- Sources: AI-generated examples reviewed and validated by a domain expert with professional experience in banking and data engineering

---

## Ethical Considerations

This tool is designed to assist technical routing decisions, not to replace human judgement. Predictions should be reviewed by a qualified data engineer before architectural decisions are made. The dataset was synthetically generated and may not cover all edge cases.

---

## AI Contributions

This project was developed with assistance from Claude (Anthropic) for dataset generation, code scaffolding and documentation. All examples were reviewed and validated by the author. The classifier training, evaluation and final model selection were performed and verified by the author.

---

## Author

Diana Mikova
Uppsala University — Information Retrieval, 2026
