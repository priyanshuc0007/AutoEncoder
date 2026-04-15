# 🚀 Getting Started - Complete Startup Guide

Choose your preferred way to test the AutoML system!

---

## ⚡ 5-Minute Quick Start

### 1. Install Dependencies (2 minutes)
```bash
cd c:\Users\Hp\Desktop\autoencoder
pip install -r requirements.txt
```

### 2. Choose Your Interface

#### Option A: Streamlit (Easiest - Recommended) 🎨
```bash
run_streamlit.bat
```
Opens automatically at: http://localhost:8501

#### Option B: Flask API 🔌
```bash
run_flask.bat
```
Available at: http://localhost:5000

#### Option C: Command Line (Python) 💻
```bash
python quickstart.py
```

---

## 📚 Detailed Setup by OS

### 🪟 Windows Setup

#### Prerequisites Check
```bash
# Check Python installed
python --version

# Should show Python 3.8+
```

#### Installation Steps
```bash
# 1. Navigate to project
cd c:\Users\Hp\Desktop\autoencoder

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run one of:
# Option A - Streamlit
run_streamlit.bat

# Option B - Flask
run_flask.bat

# Option C - Command Line
python quickstart.py
```

### 🍎 macOS Setup

```bash
# 1. Navigate to project
cd ~/Desktop/autoencoder

# 2. Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run one of:
# Option A - Streamlit
bash run_streamlit.sh

# Option B - Flask
bash run_flask.sh

# Option C - Command Line
python quickstart.py
```

### 🐧 Linux Setup

```bash
# 1. Navigate to project
cd ~/autoencoder

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run one of:
# Option A - Streamlit
bash run_streamlit.sh

# Option B - Flask
bash run_flask.sh

# Option C - Command Line
python quickstart.py
```

---

## 🎨 Streamlit - Interactive Web UI

### Start Streamlit

```bash
# Windows
run_streamlit.bat

# macOS/Linux
bash run_streamlit.sh

# Manual
streamlit run app_streamlit.py
```

### Access UI
- **URL**: http://localhost:8501
- Opens automatically in your browser

### Using Streamlit

1. **Tab 1: Run Pipeline**
   - Upload CSV file
   - Select label column
   - Select text column
   - Click "Start Pipeline"

2. **Tab 2: Results**
   - View best model metrics
   - See model comparison
   - Check data analysis

3. **Tab 3: History**
   - See all past experiments
   - Compare performance
   - Re-view old results

4. **Tab 4: Help**
   - Data format guide
   - Tips & tricks
   - Troubleshooting

### Example CSV
```csv
text,label
"Amazing product!",positive
"Terrible quality.",negative
"I love it!",positive
```

### Tips
- Start with sample data: `data/sample_dataset.csv`
- Upload CSV with at least 100 samples
- Keep text column < 500 characters
- Balanced classes work better

---

## 🔌 Flask API - Production-Ready

### Start Flask

```bash
# Windows
run_flask.bat

# macOS/Linux
bash run_flask.sh

# Manual
python app_flask.py
```

### Access API
- **Base URL**: http://localhost:5000
- **Documentation**: http://localhost:5000/api/info

### Health Check
```bash
curl http://localhost:5000/health
```

### Run Pipeline (cURL)
```bash
curl -X POST http://localhost:5000/api/pipeline/run \
  -F "file=@data/sample_dataset.csv" \
  -F "label_column=label" \
  -F "text_column=text"
```

### Python Client (Easier)
```python
from api_client import AutoMLClient

client = AutoMLClient('http://localhost:5000')

# Run pipeline
exp_id = client.run_pipeline(
    csv_path='data/sample_dataset.csv',
    label_column='label',
    text_column='text'
)

# Wait for results
results = client.wait_for_completion(exp_id)

print(f"F1 Score: {results['best_model']['metrics']['f1_score']}")
```

### Using Python Client
```bash
python api_client.py
```

This will:
- Check API health
- Run pipeline
- Download results
- Make predictions
- List experiments

---

## 💻 Command Line - Simple Python Script

### Run Quickstart
```bash
python quickstart.py
```

### What It Does
- Uses sample data
- Trains models
- Generates report
- Saves results to `experiments/`

### Expected Output
```
======================================================================
🚀 AutoML Pipeline - Quick Start Test
======================================================================

✓ Found sample data at data/sample_dataset.csv

📌 Step 1: Initializing AutoML Pipeline...
✓ Pipeline initialized

📌 Step 2: Running AutoML Pipeline...
⏳ This may take a few minutes...

======================================================================
📊 RESULTS
======================================================================

Best Model: microsoft/MiniLM-L6-H384-uncased
F1 Score: 0.9234
Accuracy: 0.9145
Latency: 15.23ms
```

---

## 🧪 Testing with Your Own Data

### Step 1: Prepare CSV
Create CSV with 2 columns:
```csv
text,label
"Your text here","class_name"
...
```

### Step 2: Choose Interface

#### Streamlit
- Upload CSV via UI
- Select columns
- Run pipeline

#### Flask API
```python
client = AutoMLClient()
exp_id = client.run_pipeline(
    csv_path='your_file.csv',
    label_column='label',
    text_column='text'
)
```

#### Command Line
```python
from automl import AutoMLPipeline

pipeline = AutoMLPipeline()
result = pipeline.run(
    csv_path='your_file.csv',
    label_column='label',
    text_column='text'
)
```

---

## 📊 Checking Results

### Results Location
```
experiments/
└── [timestamp]/
    ├── best_model_report.txt     # Detailed report
    ├── config.pkl               # Configuration
    └── [models]/                # Trained models
```

### View Report
```bash
# Windows
type experiments\[timestamp]\best_model_report.txt

# macOS/Linux
cat experiments/[timestamp]/best_model_report.txt
```

---

## 🐛 Troubleshooting

### "Port already in use"

**Streamlit:**
```bash
streamlit run app_streamlit.py --server.port=8502
```

**Flask:**
Edit `app_flask.py`, change:
```python
app.run(port=5001)  # Change port
```

### "Module not found"

```bash
pip install -r requirements.txt
```

### "CUDA out of memory"

```bash
# Use smaller model
# Edit data_intelligence.py
models = ["microsoft/MiniLM-L6-H384-uncased"]
```

### "API connection refused"

Check Flask is running:
```bash
curl http://localhost:5000/health
```

If not, start Flask:
```bash
python app_flask.py
```

### "CSV parsing error"

Ensure CSV format:
```csv
text,label
"row1 text","class1"
"row2 text","class2"
```

---

## 🎯 Quick Decision Guide

| Goal | Tool | How |
|------|------|-----|
| Quick test | CLI | `python quickstart.py` |
| Interactive testing | Streamlit | `run_streamlit.bat` |
| Production API | Flask | `run_flask.bat` |
| Programmatic access | Python Client | `python api_client.py` |

---

## ✅ Verification Checklist

Before starting, verify:

- ✅ Python 3.8+ installed
- ✅ In project directory
- ✅ Dependencies installed (`pip install -r requirements.txt`)
- ✅ Sample data exists (`data/sample_dataset.csv`)
- ✅ Ports available (8501 for Streamlit, 5000 for Flask)

---

## 🚀 You're Ready!

Choose your interface and start testing:

1. **Quick Test**: `python quickstart.py`
2. **Visual Testing**: `run_streamlit.bat`
3. **API Testing**: `run_flask.bat` + `python api_client.py`

---

## 📞 Need Help?

Check these files:
- [README.md](README.md) - Project overview
- [WEB_INTERFACES_GUIDE.md](WEB_INTERFACES_GUIDE.md) - Detailed interface guide
- [example.py](example.py) - Code examples

---

**Happy testing! 🚀**
