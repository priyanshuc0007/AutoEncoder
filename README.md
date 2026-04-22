<div align="center">

# 🚀 AutoLLM — Automated Fine-Tuning Pipeline

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Transformers-FFD21E)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-grade AutoLLM system that automatically handles the complete machine learning pipeline for text classification tasks.

</div>

---

## 📑 Table of Contents

- [✨ Features](#-features)
- [📋 Architecture](#-architecture)
- [🏗️ Project Structure](#️-project-structure)
- [🚀 Quick Start](#-quick-start)
- [📊 Pipeline Steps Explained](#-pipeline-steps-explained)
- [📈 Output Files](#-output-files)
- [🔧 Configuration & Customization](#-configuration--customization)
- [💡 Tips for Best Results](#-tips-for-best-results)
- [🎯 Expected Performance](#-expected-performance)
- [🐛 Troubleshooting](#-troubleshooting)
- [📚 Next Steps](#-next-steps-advanced-features)
- [📝 License](#-license)
- [🤝 Contributing](#-contributing)

---

## ✨ Features

### 🧠 Intelligent Data Analysis
- **Task Detection**: Automatically detects binary vs multiclass tasks
- **Imbalance Detection**: Identifies class imbalance and recommends correction strategies
- **Text Length Analysis**: Intelligently sets max sequence length (95th percentile)
- **Data Quality Checks**: Handles missing values, duplicates, and validates data

### 🤖 Smart Model Selection
- **Dataset-Aware Selection**: Picks optimal models based on dataset size
  - `< 2K samples`: MiniLM (fastest, smallest)
  - `2K-10K samples`: DistilBERT (balanced)
  - `> 10K samples`: BERT, RoBERTa (most accurate)

### ⚙️ Automatic Hyperparameter Tuning
- **Dynamic Epochs**: Adapts training duration based on dataset size
- **Adaptive Batch Size**: Optimizes batch size for GPU memory
- **Learning Rate Scaling**: Automatically scales learning rates
- **Early Stopping**: Prevents overfitting with configurable patience

### 📊 Comprehensive Evaluation
- **Multi-Metric Evaluation**: F1, Accuracy, Precision, Recall
- **Latency Tracking**: Measures inference speed per sample
- **Intelligent Scoring**: Balances accuracy (70%) and speed (30%)
- **Detailed Reporting**: Classification reports, confusion matrices

---

## 📋 Architecture

```
CSV Input
    ↓
┌─────────────────────────────┐
│   Data Validation Layer     │ ← Validates & loads CSV
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│  Data Intelligence Engine   │ ← Analyzes dataset
├─────────────────────────────┤
│ • Task Detection            │
│ • Imbalance Detection       │
│ • Text Length Analysis      │
│ • Model Selection           │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│   Model Training Layer      │ ← Fine-tunes transformers
├─────────────────────────────┤
│ • Multi-model training      │
│ • Hyperparameter tuning     │
│ • Early stopping            │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│  Evaluation & Selection     │ ← Compares & selects best
├─────────────────────────────┤
│ • Individual evaluation     │
│ • Multi-model comparison    │
│ • Best model selection      │
└─────────────────────────────┘
    ↓
Best Model + Report
```

---

## 🏗️ Project Structure

```
autoencoder/
├── automl/
│   ├── __init__.py                 # Package init
│   ├── data_validator.py           # Data loading & validation
│   ├── data_intelligence.py        # Dataset analysis engine
│   ├── model_trainer.py            # Model training with tuning
│   ├── evaluator.py                # Model evaluation & comparison
│   └── pipeline.py                 # Main orchestrator
├── data/                           # Input CSV files go here
├── models/                         # Trained models saved here
├── experiments/                    # Experiment results & reports
├── example.py                      # Usage example
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

---

## 📦 Requirements

| Package | Version |
|---------|---------|
| Python | 3.8+ |
| PyTorch | 2.0+ |
| Transformers | 4.30+ |
| scikit-learn | 1.0+ |
| pandas | 1.5+ |

Install all dependencies at once:
```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to project
cd autoencoder

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Data

Create a CSV file with at least:
- One column with **text data** (messages, reviews, descriptions, etc.)
- One column with **labels** (categories, classes)

Example `data/sample.csv`:
```csv
text,label
"This movie was fantastic!",positive
"Terrible experience.",negative
"Really enjoyed this.",positive
"Would not recommend.",negative
```

### 3. Run the Pipeline

```python
from automl import AutoLLMPipeline

# Initialize
pipeline = AutoLLMPipeline(output_dir="experiments")

# Run
result = pipeline.run(
    csv_path="data/sample.csv",
    label_column="label",
    text_column="text",  # or None for auto-detection
    experiment_name="my_first_run"
)

# Check results
print(f"Best model: {result['best_model_name']}")
print(f"F1 Score: {result['best_model_metrics']['f1_score']:.4f}")
print(f"Model saved to: {result['best_model_path']}")
```

---

## 📊 Pipeline Steps Explained

### Step 1: Data Validation
```
✓ Loads CSV file
✓ Validates presence of required columns
✓ Removes missing values
✓ Removes duplicates
✓ Auto-detects text column if not specified
```

### Step 2: Data Intelligence
```
✓ Detects task type (binary/multiclass)
✓ Calculates class imbalance ratio
✓ Analyzes text length distribution
✓ Selects appropriate models
✓ Generates training configuration
```

**Example Output:**
```
⚖️  Imbalance Ratio: 2.45
   Use Class Weights: ✓ YES
   Use Focal Loss: ✗ NO

🤖 Selected Models:
   - microsoft/MiniLM-L6-H384-uncased

⚙️  Training Config:
   Epochs: [2, 3]
   Batch Size: 16
   Learning Rate: 2e-05
```

### Step 3: Data Preparation
```
✓ Encodes labels
✓ Splits into train/validation sets (80/20)
✓ Creates PyTorch datasets
✓ Prepares tokenizers
```

### Step 4: Model Training
```
✓ Trains each selected model
✓ Tries different hyperparameter combinations
✓ Uses early stopping to prevent overfitting
✓ Tracks training time
✓ Saves best checkpoints
```

### Step 5: Model Evaluation
```
✓ Evaluates on validation set
✓ Calculates F1, Accuracy, Precision, Recall
✓ Measures inference latency
✓ Generates classification reports
```

### Step 6: Best Model Selection
```
✓ Compares all trained models
✓ Uses composite score: 70% F1 + 30% (1/latency)
✓ Selects best performer
✓ Generates final report
```

---

## 📈 Output Files

After running, check `experiments/[timestamp]/`:

```
experiments/
└── 20240101_120000/
    ├── best_model_report.txt       # Detailed evaluation report
    ├── config.pkl                  # Saved configuration
    └── models/
        └── [model_checkpoints]     # Trained model files
```

### Report Example:
```
======================================================================
BEST MODEL EVALUATION REPORT
======================================================================

Model: microsoft/MiniLM-L6-H384-uncased

METRICS:
  F1 Score: 0.9234
  Accuracy: 0.9145
  Precision: 0.9201
  Recall: 0.9268
  Avg Latency: 15.23ms per sample

CLASSIFICATION REPORT:
                precision    recall  f1-score   support
       positive       0.94      0.92      0.93       500
       negative       0.91      0.93      0.92       500
```

---

## 🔧 Configuration & Customization

### Modify Training Parameters

Edit `data_intelligence.py` `_get_training_config()`:

```python
# For your use case
if dataset_size < 3000:
    num_epochs_range = [2, 3, 4]  # Try more epochs
    batch_size = 8  # Smaller batches
    learning_rate = 3e-5  # Higher LR
```

### Add More Models

Edit `data_intelligence.py` `_select_models()`:

```python
if dataset_size < 10000:
    models = [
        "microsoft/MiniLM-L6-H384-uncased",
        "distilbert-base-uncased",
        # Add more!
        "sentence-transformers/distilroberta-base"
    ]
```

### Adjust Imbalance Thresholds

Edit `data_intelligence.py` `_detect_imbalance()`:

```python
info = {
    'use_class_weights': imbalance_ratio > 1.5,  # Lower threshold
    'use_focal_loss': imbalance_ratio > 3.0,     # Different threshold
}
```

---

## 💡 Tips for Best Results

### ✅ Do's
- Use **at least 500 samples** per class for best results
- Ensure **balanced classes** (ideally 1:1 ratio)
- Use **clear, meaningful text** data
- Include **sufficient examples** of each class

### ❌ Don'ts
- Use very short text (< 5 characters)
- Have extreme class imbalance (> 10:1)
- Use <100 total samples
- Include duplicates or noise

---

## 🎯 Expected Performance

### MVP Performance on Standard Datasets
- **Accuracy**: 85-95% (depends on data quality)
- **F1 Score**: 0.84-0.94
- **Inference Speed**: 10-30ms per sample on GPU
- **Training Time**: 5-30 minutes (depends on model & data size)

## 🐛 Troubleshooting

### Q: "CUDA out of memory"
**A:** Reduce batch size in training config or use a smaller model

```python
# Modify in data_intelligence.py
batch_size = 8  # Instead of 16/32
```

### Q: "Model trained but F1 is very low"
**A:** Your data might be imbalanced or have low signal. Try:
- Increase training data
- Check data quality
- Use class weights (pipeline enables automatically)

### Q: "Training is too slow"
**A:** Switch to faster model:
```python
# Instead of BERT, use MiniLM
models = ["microsoft/MiniLM-L6-H384-uncased"]
```

## 📚 Next Steps (Advanced Features)

After MVP works, add:

1. **Ensemble Methods**: Combine multiple models
2. **SHAP Explainability**: Understand model decisions
3. **Threshold Optimization**: For binary classification
4. **Active Learning**: Suggest which samples to label
5. **Real-time Serving**: REST API for predictions
6. **Multi-GPU Training**: Parallel model training

## 📝 License

MIT License

## 🤝 Contributing

Feel free to extend and improve!

---

**Questions?** Check the example files or create an issue.
