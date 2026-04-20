# 🌐 Web Interfaces Guide

This guide shows you how to use both **Streamlit** and **Flask** interfaces to test the AutoLLM system.

## 📋 Overview

| Interface | Purpose | Use Case |
|-----------|---------|----------|
| **Streamlit** | Interactive Web UI | Quick testing, visualization, easy to use |
| **Flask API** | REST API | Production, programmatic access, integration |

---

## 🎨 Option 1: Streamlit App (Recommended for Testing)

### What is Streamlit?
- Interactive web UI
- No coding required
- Real-time visualizations
- Perfect for testing and exploring

### ⚡ Quick Start

#### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 2: Run Streamlit App

**Windows:**
```bash
run_streamlit.bat
```

**macOS/Linux:**
```bash
bash run_streamlit.sh
```

**Or manually:**
```bash
streamlit run app_streamlit.py
```

#### Step 3: Open in Browser
Automatically opens at: **http://localhost:8501**

### 📱 Using the Streamlit App

#### Tab 1: Run Pipeline
1. Upload your CSV file
2. Select label column
3. Select text column (or auto-detect)
4. Click "🚀 Start Pipeline"
5. Wait for results

#### Tab 2: Results & Analysis
- View best model performance
- See model comparison table
- View data intelligence report
- Check training configuration

#### Tab 3: Experiment History
- See all past experiments
- Compare different runs
- Re-view previous results

#### Tab 4: Help
- Quick start guide
- Data format examples
- Tips and troubleshooting

### 🎯 Example CSV Format
```csv
text,label
"This is awesome!",positive
"Terrible experience.",negative
"I love it!",positive
"Would not recommend.",negative
```

---

## 🔌 Option 2: Flask REST API (For Production)

### What is Flask API?
- REST API endpoints
- JSON requests/responses
- Background job processing
- Production-grade
- Can be integrated with other apps

### ⚡ Quick Start

#### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 2: Run Flask App

**Windows:**
```bash
run_flask.bat
```

**macOS/Linux:**
```bash
bash run_flask.sh
```

**Or manually:**
```bash
python app_flask.py
```

#### Step 3: API Available
- Base URL: **http://localhost:5000**
- Documentation: **http://localhost:5000/api/info**

### 🔗 API Endpoints

#### 1. Health Check
```bash
GET /health

Response:
{
    "status": "healthy",
    "timestamp": "2024-01-01T12:00:00"
}
```

#### 2. Get API Info
```bash
GET /api/info

Response:
{
    "name": "AutoLLM Text Classification API",
    "version": "1.0.0",
    "endpoints": { ... }
}
```

#### 3. Run Pipeline (Main Endpoint)
```bash
POST /api/pipeline/run

Form Data:
- file: CSV file
- label_column: "label"
- text_column: "text" (optional)
- experiment_name: "my_experiment" (optional)

Response:
{
    "experiment_id": "uuid-here",
    "status": "success",
    "data": {
        "best_model_name": "distilbert-base-uncased",
        "metrics": {
            "f1_score": 0.92,
            "accuracy": 0.91,
            "latency_ms": 15.23
        }
    }
}
```

#### 4. Get Results
```bash
GET /api/pipeline/{experiment_id}/results

Response:
{
    "experiment_id": "...",
    "status": "success",
    "best_model": { ... },
    "model_comparison": [ ... ]
}
```

#### 5. List All Experiments
```bash
GET /api/experiments

Response:
{
    "total": 5,
    "experiments": [
        {
            "experiment_id": "...",
            "timestamp": "...",
            "status": "success",
            "best_model": "...",
            "f1_score": 0.92
        }
    ]
}
```

#### 6. Download Model
```bash
GET /api/pipeline/{experiment_id}/download-model

Returns: ZIP file with model
```

#### 7. Get Report
```bash
GET /api/pipeline/{experiment_id}/report

Response:
{
    "report": "Full text report..."
}
```

#### 8. Make Predictions
```bash
POST /api/predict/{experiment_id}

Request Body:
{
    "texts": ["This is great!", "Terrible!"]
}

Response:
{
    "predictions": [
        {
            "text": "This is great!",
            "prediction": 1,
            "confidence": 0.95
        }
    ]
}
```

### 🐍 Python Example
```python
import requests

# Run pipeline
response = requests.post(
    'http://localhost:5000/api/pipeline/run',
    files={'file': open('data.csv', 'rb')},
    data={
        'label_column': 'label',
        'text_column': 'text'
    }
)

experiment_id = response.json()['experiment_id']

# Get results
results = requests.get(
    f'http://localhost:5000/api/pipeline/{experiment_id}/results'
).json()

print(f"F1 Score: {results['best_model']['metrics']['f1_score']}")
```

### 🔧 cURL Examples

#### Run Pipeline
```bash
curl -X POST http://localhost:5000/api/pipeline/run \
  -F "file=@data.csv" \
  -F "label_column=label" \
  -F "text_column=text"
```

#### Get Results
```bash
curl http://localhost:5000/api/pipeline/{experiment_id}/results
```

#### Make Predictions
```bash
curl -X POST http://localhost:5000/api/predict/{experiment_id} \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great!", "Bad"]}'
```

---

## 🚀 Run Both Simultaneously

Want to use both? Open 2 terminals:

**Terminal 1 - Streamlit:**
```bash
streamlit run app_streamlit.py
```

**Terminal 2 - Flask:**
```bash
python app_flask.py
```

Then access:
- Streamlit UI: http://localhost:8501
- Flask API: http://localhost:5000

---

## 🎯 Which One to Use?

### Use Streamlit if:
- ✅ Quick testing
- ✅ Non-technical users
- ✅ Want visualizations
- ✅ Need simple interface
- ✅ Exploring data

### Use Flask if:
- ✅ Production deployment
- ✅ Integrate with other apps
- ✅ Programmatic access
- ✅ Background job processing
- ✅ Need REST API
- ✅ Build custom frontend

---

## 🐛 Troubleshooting

### Streamlit Issues

**Port already in use:**
```bash
streamlit run app_streamlit.py --server.port=8502
```

**Module not found:**
```bash
pip install -r requirements.txt
```

### Flask Issues

**Port already in use:**
Edit `app_flask.py` and change port:
```python
app.run(host='0.0.0.0', port=5001)  # Changed to 5001
```

**CORS errors:**
- CORS is already enabled in Flask app
- Make sure requests include proper headers

---

## 📊 Example Workflow

### 1. Test with Streamlit
```
Open http://localhost:8501
Upload CSV → See results → Download report
```

### 2. Deploy with Flask
```
Integrate API into your app
POST /api/pipeline/run → GET /api/pipeline/{id}/results
```

### 3. Use Predictions
```
POST /api/predict/{id}
Get predictions for new data
```

---

## 🔒 Security Notes

For production deployment:

1. **Use environment variables:**
```python
# app_flask.py
flask_env = os.getenv('FLASK_ENV', 'development')
```

2. **Add authentication:**
```python
from flask_httpauth import HTTPBasicAuth
auth = HTTPBasicAuth()
```

3. **Use HTTPS:**
```bash
# Use gunicorn with SSL
gunicorn --certfile=cert.pem --keyfile=key.pem app_flask:app
```

4. **Rate limiting:**
```python
from flask_limiter import Limiter
limiter = Limiter(app)
```

---

## 📚 Next Steps

1. **Test both interfaces** with sample data
2. **Integrate Flask API** into your application
3. **Deploy to production** (Streamlit Cloud, Heroku, AWS, etc.)
4. **Add authentication** for security

---

## 💡 Pro Tips

- Save experiment results for comparison
- Use Flask API for batch processing
- Monitor latency for production use
- Keep data uploads organized
- Review reports for insights

---

**Ready to start? Pick one and begin testing!** 🚀
