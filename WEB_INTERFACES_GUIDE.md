# Interfaces Guide

Two ways to use AutoLLM: **CLI** (for training and inference) and **Streamlit** (for interactive demos).

---

## CLI â€” train.py

The primary interface. Accepts any CSV or XLSX file.

### Quick examples

```bash
# Minimal â€” auto-detect text column
python train.py --file data.csv --label sentiment

# Explicit text column
python train.py --file emails.xlsx --label category --text body

# Merge multiple text columns
python train.py --file data.csv --label label --text-columns title body description

# Pin specific models
python train.py --file data.csv --label label \
  --models prajjwal1/bert-tiny distilbert-base-uncased

# Full options: Optuna + cross-validation
python train.py --file data.csv --label label \
  --optuna --optuna-trials 15 \
  --cv --cv-folds 5
```

### All flags

| Flag | Description | Default |
|------|-------------|---------|
| `--file` | CSV or XLSX path | **required** |
| `--label` | label column name | **required** |
| `--text` | single text column | auto-detect |
| `--text-columns` | multiple columns to merge | â€” |
| `--models` | HuggingFace model IDs | auto-select |
| `--optuna` | Bayesian hyperparameter search | off |
| `--optuna-trials N` | trials per model (3â€“20) | 10 |
| `--cv` | K-fold cross-validation on best model | off |
| `--cv-folds K` | number of folds | 5 |
| `--output-dir DIR` | experiment save directory | experiments/ |
| `--name NAME` | custom experiment subfolder name | timestamp |

### What gets printed at the end

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
  NEXT STEP â€” run inference:
    python predict.py --experiment experiments/20260422_123456 --text "..."
    python predict.py --experiment experiments/20260422_123456 --file new.csv
```

---

## CLI â€” predict.py

Loads the best model and label encoder from a completed experiment and runs
inference on a single string or a whole CSV/XLSX file.

### Quick examples

```bash
# Single text
python predict.py --experiment experiments/20260422_123456 --text "Amazing product!"

# Batch file â€” auto-detect text column, print preview
python predict.py --experiment experiments/20260422_123456 --file new_reviews.csv

# Batch file â€” explicit column, save to CSV
python predict.py --experiment experiments/20260422_123456 \
  --file new_reviews.xlsx --text-column review_body --output predictions.csv

# Show top-3 label candidates with confidence
python predict.py --experiment experiments/20260422_123456 \
  --text "Not sure about this." --top-k 3
```

### All flags

| Flag | Description | Default |
|------|-------------|---------|
| `--experiment` | experiment folder (loads config.pkl) | recommended |
| `--model` | specific model folder override | from config |
| `--text` | single text string | â€” |
| `--file` | CSV or XLSX for batch prediction | â€” |
| `--text-column` | text column in input file | auto-detect |
| `--output` | save predictions to this CSV | print to terminal |
| `--batch-size N` | inference batch size | 32 |
| `--top-k K` | show top-K classes with confidence | 1 |

### Sample output â€” single text

```
  Input text    : Amazing product!
  Prediction    : positive
  Confidence    : 97.4%

  Inference time: 5.80 ms  (tokenization + model forward)
```

### Sample output â€” batch file with --output

```
  Running inference on 1200 texts  (batch_size=32) ...

  Done. Avg latency: 3.20 ms/sample

  Predictions saved to: predictions.csv

  Label distribution in predictions:
    positive                       834  (69.5%)
    negative                       366  (30.5%)
```

---

## Streamlit â€” Interactive Demo UI

For exploring results and demoing in a browser. Not intended for production use.

```bash
# Windows
run_streamlit.bat

# macOS / Linux
bash run_streamlit.sh

# Manual
streamlit run app_streamlit.py
```

Opens at **http://localhost:8501**

### Features

- Upload CSV or XLSX, pick label + text columns with a dropdown
- Model selector with parameter count and speed rating (Step 2.5)
- Advanced options: Optuna trials, cross-validation folds, GPU toggle
- Results dashboard: F1, accuracy, per-class breakdown
- Latency bar chart â€” all models compared side by side
- Explainability: token importance heatmap, confidence distribution
- Full text report viewer (best_model_report.txt)

### Tips

- Use at least 100 rows for meaningful results
- Try `data/sample_dataset.csv` first to verify setup
- Balanced classes give better model comparisons

---

## Which Interface?

| Goal | Use |
|------|-----|
| Train on new data | `python train.py --file data.csv --label label` |
| Run inference / get predictions | `python predict.py --experiment experiments/<run> --file new.csv` |
| Interactive visual demo | Streamlit â€” `run_streamlit.bat` |
| Scripting / automation | `AutoLLMPipeline().run(...)` |

---

## Troubleshooting

### "Module not found"
```bash
pip install -r requirements.txt
```

### Streamlit port already in use
```bash
streamlit run app_streamlit.py --server.port=8502
```

### "CUDA out of memory"
Pin a smaller model:
```bash
python train.py --file data.csv --label label --models prajjwal1/bert-tiny
```

### "Label column not found"
```python
import pandas as pd
print(pd.read_csv("your_file.csv").columns.tolist())
```

### "Only 1 unique class found"
Your label column must have at least 2 distinct class values.

### "Class has only 1 sample"
Each class needs at least 2 samples.
Remove or merge classes with fewer than 2 samples before running.
