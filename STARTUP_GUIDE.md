# 🚀 Getting Started - Complete Startup Guide

Two ways to use AutoLLM: **CLI** (for production/scripting) or **Streamlit** (for interactive demos).

---

## ⚡ 5-Minute Quick Start

### 1. Install Dependencies
```bash
cd c:\Users\Hp\Desktop\autoencoder
pip install -r requirements.txt
```

### 2. Train on your data (CLI)
```bash
python train.py --file data/sample_dataset.csv --label label
```

### 3. Run inference
```bash
python predict.py --experiment experiments/<timestamp> --text "Your text here"
```

---

## 📚 Setup by OS

### 🪟 Windows
```bash
# 1. Navigate to project
cd c:\Users\Hp\Desktop\autoencoder

# 2. Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train
python train.py --file data/sample_dataset.csv --label label

# 5. Predict
python predict.py --experiment experiments/<timestamp> --text "Great product!"
```

### 🍎 macOS / 🐧 Linux
```bash
# 1. Navigate to project
cd ~/Desktop/autoencoder

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train
python train.py --file data/sample_dataset.csv --label label

# 5. Predict
python predict.py --experiment experiments/<timestamp> --text "Great product!"
```

---

## 💻 CLI — train.py

The primary way to use AutoLLM. Accepts CSV or XLSX, auto-selects models based
on dataset size, and prints a full performance + latency comparison at the end.

### Minimal
```bash
python train.py --file data.csv --label sentiment
```

### Explicit text column
```bash
python train.py --file emails.xlsx --label category --text body
```

### Merge multiple text columns
```bash
python train.py --file data.csv --label label --text-columns title description
```

### Pin specific models
```bash
python train.py --file data.csv --label label \
  --models prajjwal1/bert-tiny distilbert-base-uncased
```

### Full options with Optuna + CV
```bash
python train.py --file data.csv --label label \
  --optuna --optuna-trials 15 \
  --cv --cv-folds 5 \
  --output-dir experiments --name my_run
```

### All flags
| Flag | Description | Default |
|------|-------------|---------|
| `--file` | CSV or XLSX path | required |
| `--label` | label column name | required |
| `--text` | text column name | auto-detect |
| `--text-columns` | multiple columns to merge | — |
| `--models` | specific HuggingFace IDs | auto-select |
| `--optuna` | Bayesian hyperparameter search | off |
| `--optuna-trials N` | Optuna trials per model | 10 |
| `--cv` | K-fold cross-validation | off |
| `--cv-folds K` | number of folds | 5 |
| `--output-dir DIR` | experiment save directory | experiments/ |
| `--name NAME` | experiment subfolder name | timestamp |

### What train.py prints
```
======================================================================
  TRAINING COMPLETE
======================================================================
  Best Model : distilbert-base-uncased
  F1 Score   : 0.9312
  Accuracy   : 0.9280
  Latency    : 18.40 ms  (single-sample, tokenization included)

  Model path      : models/20260422_123456_distilbert-base-uncased
  Experiment dir  : experiments/20260422_123456
  Report          : experiments/20260422_123456/best_model_report.txt

----------------------------------------------------------------------
  ALL MODELS COMPARED
  Model                          F1      Accuracy  Latency(ms)
  prajjwal1/bert-tiny            0.8901  0.8783    4.20
  prajjwal1/bert-mini            0.9056  0.8990    6.80
  distilbert-base-uncased        0.9312  0.9280    18.40

----------------------------------------------------------------------
  NEXT STEP — run inference:
    python predict.py --experiment experiments/20260422_123456 --text "..."
    python predict.py --experiment experiments/20260422_123456 --file new_data.csv
```

---

## 🔍 CLI — predict.py

Loads the best model and label encoder from a completed experiment and
runs inference on a single text or a whole CSV/XLSX file.

### Single text
```bash
python predict.py --experiment experiments/20260422_123456 --text "Amazing product!"
```

### Batch file — print to terminal
```bash
python predict.py --experiment experiments/20260422_123456 --file new_reviews.csv
```

### Batch file — save to CSV
```bash
python predict.py --experiment experiments/20260422_123456 \
  --file new_reviews.xlsx --text-column review_body \
  --output predictions.csv
```

### Show top-3 candidates with confidence
```bash
python predict.py --experiment experiments/20260422_123456 \
  --text "Not entirely sure about this." --top-k 3
```

### Override model directly
```bash
python predict.py --model models/20260422_123456_distilbert-base-uncased \
  --experiment experiments/20260422_123456 --file data.csv
```

### All flags
| Flag | Description | Default |
|------|-------------|---------|
| `--experiment` | experiment folder | recommended |
| `--model` | specific model folder override | from config.pkl |
| `--text` | single text to classify | — |
| `--file` | CSV or XLSX for batch | — |
| `--text-column` | text column in input file | auto-detect |
| `--output` | save predictions CSV | print to terminal |
| `--batch-size N` | inference batch size | 32 |
| `--top-k K` | show top-K classes | 1 |

---

## 🎨 Streamlit — Interactive Demo UI

For exploring results and demoing interactively in a browser.
Not intended for production use or scripting.

```bash
# Windows
run_streamlit.bat

# macOS / Linux
bash run_streamlit.sh

# Manual
streamlit run app_streamlit.py
```

Opens at: **http://localhost:8501**

### Features
- Upload CSV or XLSX, select label + text columns
- Model selector with info table (Step 2.5)
- Advanced options: Optuna, cross-validation, GPU toggle
- Results: metrics table, latency bar chart, model comparison
- Explainability: token importance, confidence distribution
- Full report viewer

### Tips
- Prepare CSV with at least 100 rows
- Balanced classes give better results
- Use `data/sample_dataset.csv` to test first

---

## 🧪 Testing with Your Own Data

### Data format (CSV)
```csv
text,label
"This movie is amazing!",positive
"Terrible experience.",negative
"Would not recommend.",negative
"Absolutely loved it!",positive
```

### Data format (XLSX)
Same structure — just save as `.xlsx`. `openpyxl` is installed with the requirements.

### Programmatic use (Python API)
```python
from automl import AutoLLMPipeline

pipeline = AutoLLMPipeline(output_dir="experiments")
result = pipeline.run(
    csv_path="data/your_data.csv",
    label_column="label",
    text_column="text",
    use_optuna=True,
)

print(result["best_model_name"])
print(result["best_model_metrics"])
```

---

## 📊 Checking Results

```
experiments/
└── 20260422_123456/
    ├── best_model_report.txt      ← full evaluation report
    ├── decisions_log.json         ← every automated decision + reason
    ├── data_quality_report.txt    ← input data health check
    ├── baseline_comparison.txt    ← model vs dummy classifier
    ├── explainability_report.txt  ← token importance + confidence
    ├── pipeline_state.json        ← step-by-step run tracker
    └── config.pkl                 ← label encoder + config (used by predict.py)

models/
└── 20260422_123456_distilbert-base-uncased/
    ├── config.json
    ├── model.safetensors
    └── tokenizer files
```

```bash
# View report — Windows
type experiments\20260422_123456\best_model_report.txt

# View report — macOS/Linux
cat experiments/20260422_123456/best_model_report.txt
```

---

## 🐛 Troubleshooting

### "Module not found"
```bash
pip install -r requirements.txt
```

### "Port already in use" (Streamlit)
```bash
streamlit run app_streamlit.py --server.port=8502
```

### "CUDA out of memory"
Pin smaller models with:
```bash
python train.py --file data.csv --label label --models prajjwal1/bert-tiny
```

### "Label column not found"
Check column names in your file:
```python
import pandas as pd
df = pd.read_csv("your_file.csv")
print(df.columns.tolist())
```

### "Only 1 unique class found"
Your label column must have at least 2 distinct class values.

### "Class has only 1 sample"
Each class needs at least 2 samples for stratified splitting.
Remove or merge classes with fewer than 2 samples.

---

## 🎯 Quick Decision Guide

| Goal | Tool | Command |
|------|------|---------|
| Train on new data | CLI | `python train.py --file data.csv --label label` |
| Classify new texts | CLI | `python predict.py --experiment experiments/<run> --file new.csv` |
| Interactive demo | Streamlit | `run_streamlit.bat` |
| Programmatic pipeline | Python | `AutoLLMPipeline().run(...)` |

---

## ✅ Verification Checklist

- ✅ Python 3.8+ installed
- ✅ In project directory
- ✅ Virtual environment activated
- ✅ Dependencies installed (`pip install -r requirements.txt`)
- ✅ Dataset has at least 100 rows, ≥ 2 classes, ≥ 2 samples per class

---

## 🚀 You're Ready!

```bash
python train.py --file data/sample_dataset.csv --label label
