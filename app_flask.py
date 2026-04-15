"""
Flask REST API for AutoML Pipeline
Production-ready API endpoints for running and managing AutoML
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import logging
from pathlib import Path
import os
from datetime import datetime
import uuid
import pickle
import io

from automl import AutoMLPipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload
UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

# In-memory experiment tracker
experiments = {}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def allowed_file(filename):
    """Check if file is allowed"""
    return filename.endswith('.csv')


def save_experiment(experiment_id, result):
    """Save experiment to dictionary"""
    experiments[experiment_id] = {
        'timestamp': datetime.now().isoformat(),
        'result': result,
        'status': result.get('status', 'unknown')
    }


# ============================================================================
# HEALTH & STATUS ENDPOINTS
# ============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'AutoML API'
    }), 200


@app.route('/api/info', methods=['GET'])
def info():
    """Get API information"""
    return jsonify({
        'name': 'AutoML Text Classification API',
        'version': '1.0.0',
        'description': 'Automated machine learning for text classification',
        'endpoints': {
            'health': 'GET /health',
            'info': 'GET /api/info',
            'run_pipeline': 'POST /api/pipeline/run',
            'get_status': 'GET /api/pipeline/<experiment_id>/status',
            'get_results': 'GET /api/pipeline/<experiment_id>/results',
            'list_experiments': 'GET /api/experiments',
            'download_model': 'GET /api/pipeline/<experiment_id>/download-model',
        }
    }), 200


# ============================================================================
# PIPELINE ENDPOINTS
# ============================================================================

@app.route('/api/pipeline/run', methods=['POST'])
def run_pipeline():
    """
    Run AutoML pipeline
    
    Form data:
    - file: CSV file
    - label_column: Label column name
    - text_column: Text column name (optional, auto-detected)
    - experiment_name: Experiment name (optional)
    """
    try:
        # Validate input
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not allowed_file(file.filename):
            return jsonify({'error': 'File must be CSV'}), 400
        
        # Get parameters
        label_column = request.form.get('label_column')
        text_column = request.form.get('text_column', None)
        experiment_name = request.form.get('experiment_name')
        
        if not label_column:
            return jsonify({'error': 'label_column is required'}), 400
        
        # Generate experiment ID
        experiment_id = str(uuid.uuid4())
        
        # Save uploaded file
        filename = f"{experiment_id}_{file.filename}"
        filepath = UPLOAD_FOLDER / filename
        file.save(filepath)
        
        logger.info(f"Starting experiment {experiment_id}")
        
        try:
            # Run pipeline
            pipeline = AutoMLPipeline(output_dir="experiments")
            
            result = pipeline.run(
                csv_path=str(filepath),
                label_column=label_column,
                text_column=text_column,
                experiment_name=experiment_name or experiment_id
            )
            
            # Save experiment
            save_experiment(experiment_id, result)
            
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
            
            # Prepare response
            response = {
                'experiment_id': experiment_id,
                'status': result.get('status'),
                'timestamp': datetime.now().isoformat(),
            }
            
            if result.get('status') == 'success':
                response['data'] = {
                    'best_model_name': result.get('best_model_name'),
                    'best_model_path': result.get('best_model_path'),
                    'metrics': result.get('best_model_metrics'),
                    'experiment_dir': result.get('experiment_dir'),
                }
                return jsonify(response), 200
            else:
                response['error'] = result.get('error')
                return jsonify(response), 422
            
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}", exc_info=True)
            return jsonify({
                'experiment_id': experiment_id,
                'error': str(e),
                'status': 'failed'
            }), 500
        
    except Exception as e:
        logger.error(f"Request error: {str(e)}")
        return jsonify({'error': str(e)}), 400


@app.route('/api/pipeline/<experiment_id>/status', methods=['GET'])
def get_pipeline_status(experiment_id):
    """Get pipeline status for experiment"""
    if experiment_id not in experiments:
        return jsonify({'error': 'Experiment not found'}), 404
    
    exp = experiments[experiment_id]
    return jsonify({
        'experiment_id': experiment_id,
        'status': exp['status'],
        'timestamp': exp['timestamp']
    }), 200


@app.route('/api/pipeline/<experiment_id>/results', methods=['GET'])
def get_pipeline_results(experiment_id):
    """Get full results for experiment"""
    if experiment_id not in experiments:
        return jsonify({'error': 'Experiment not found'}), 404
    
    exp = experiments[experiment_id]
    result = exp['result']
    
    if result.get('status') != 'success':
        return jsonify({
            'experiment_id': experiment_id,
            'status': result.get('status'),
            'error': result.get('error')
        }), 422
    
    # Prepare results
    response = {
        'experiment_id': experiment_id,
        'status': result.get('status'),
        'timestamp': exp['timestamp'],
        'best_model': {
            'name': result.get('best_model_name'),
            'path': result.get('best_model_path'),
            'metrics': result.get('best_model_metrics'),
        },
        'data_analysis': result.get('data_analysis'),
        'experiment_dir': result.get('experiment_dir'),
    }
    
    # Add comparison dataframe as dict
    if 'comparison_df' in result and result['comparison_df'] is not None:
        response['model_comparison'] = result['comparison_df'].to_dict(orient='records')
    
    return jsonify(response), 200


# ============================================================================
# EXPERIMENT MANAGEMENT
# ============================================================================

@app.route('/api/experiments', methods=['GET'])
def list_experiments():
    """List all experiments"""
    exp_list = []
    
    for exp_id, exp_data in experiments.items():
        result = exp_data['result']
        exp_list.append({
            'experiment_id': exp_id,
            'timestamp': exp_data['timestamp'],
            'status': exp_data['status'],
            'best_model': result.get('best_model_name') if result.get('status') == 'success' else None,
            'f1_score': result.get('best_model_metrics', {}).get('f1_score') if result.get('status') == 'success' else None,
        })
    
    return jsonify({
        'total': len(exp_list),
        'experiments': sorted(exp_list, key=lambda x: x['timestamp'], reverse=True)
    }), 200


@app.route('/api/experiments/<experiment_id>', methods=['DELETE'])
def delete_experiment(experiment_id):
    """Delete experiment"""
    if experiment_id in experiments:
        del experiments[experiment_id]
        return jsonify({'message': 'Experiment deleted'}), 200
    return jsonify({'error': 'Experiment not found'}), 404


# ============================================================================
# FILE OPERATIONS
# ============================================================================

@app.route('/api/pipeline/<experiment_id>/download-model', methods=['GET'])
def download_model(experiment_id):
    """Download trained model"""
    if experiment_id not in experiments:
        return jsonify({'error': 'Experiment not found'}), 404
    
    result = experiments[experiment_id]['result']
    if result.get('status') != 'success':
        return jsonify({'error': 'No model available'}), 422
    
    model_path = result.get('best_model_path')
    if not os.path.exists(model_path):
        return jsonify({'error': 'Model files not found'}), 404
    
    # Create zip file of model
    try:
        import shutil
        zip_path = f"/tmp/model_{experiment_id}.zip"
        shutil.make_archive(zip_path.replace('.zip', ''), 'zip', model_path)
        
        return send_file(zip_path, as_attachment=True, download_name=f"model_{experiment_id}.zip")
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/pipeline/<experiment_id>/report', methods=['GET'])
def get_report(experiment_id):
    """Get experiment report"""
    if experiment_id not in experiments:
        return jsonify({'error': 'Experiment not found'}), 404
    
    result = experiments[experiment_id]['result']
    if result.get('status') != 'success':
        return jsonify({'error': 'No report available'}), 422
    
    report_path = Path(result.get('experiment_dir')) / "best_model_report.txt"
    if not report_path.exists():
        return jsonify({'error': 'Report not found'}), 404
    
    with open(report_path, 'r') as f:
        report_content = f.read()
    
    return jsonify({'report': report_content}), 200


# ============================================================================
# PREDICTION ENDPOINTS
# ============================================================================

@app.route('/api/predict/<experiment_id>', methods=['POST'])
def predict(experiment_id):
    """
    Make predictions using trained model
    
    JSON body:
    {
        "texts": ["text1", "text2", ...] or "single text"
    }
    """
    if experiment_id not in experiments:
        return jsonify({'error': 'Experiment not found'}), 404
    
    result = experiments[experiment_id]['result']
    if result.get('status') != 'success':
        return jsonify({'error': 'No model available'}), 422
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts:
            return jsonify({'error': 'No texts provided'}), 400
        
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
        
        # Load model
        model_path = result.get('best_model_path')
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Get label encoder
        exp_data = experiments[experiment_id]['result']
        # This would need label_encoder from training - simplified for now
        
        predictions = []
        for text in texts:
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                pred_class = torch.argmax(logits, dim=1).item()
                confidence = torch.softmax(logits, dim=1)[0][pred_class].item()
            
            predictions.append({
                'text': text,
                'prediction': int(pred_class),
                'confidence': float(confidence)
            })
        
        return jsonify({'predictions': predictions}), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Development
    app.run(debug=True, host='0.0.0.0', port=5000)
"""
Flask REST API for AutoML Pipeline
Production-grade API for AutoML Text Classification
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import threading
from typing import Dict, Any
import uuid

from automl import AutoMLPipeline

# Configure Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'experiments'

# Create folders
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['RESULTS_FOLDER']).mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory job tracking
JOBS = {}


class Job:
    """Track pipeline job status"""
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.status = "pending"  # pending, running, completed, failed
        self.progress = 0
        self.result = None
        self.error = None
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
    
    def to_dict(self) -> Dict:
        return {
            'job_id': self.job_id,
            'status': self.status,
            'progress': self.progress,
            'result': self.result,
            'error': self.error,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
        }


# Routes

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/api/v1/pipeline/info', methods=['GET'])
def pipeline_info():
    """Get pipeline information"""
    return jsonify({
        'name': 'AutoML Text Classification',
        'version': '0.1.0',
        'endpoints': {
            'health': 'GET /health',
            'pipeline_info': 'GET /api/v1/pipeline/info',
            'pipeline_run': 'POST /api/v1/pipeline/run',
            'job_status': 'GET /api/v1/jobs/{job_id}',
            'job_results': 'GET /api/v1/jobs/{job_id}/results',
            'list_jobs': 'GET /api/v1/jobs',
        }
    }), 200


@app.route('/api/v1/pipeline/run', methods=['POST'])
def pipeline_run():
    """
    Start AutoML pipeline
    
    Expected form data:
    - file: CSV file
    - label_column: Name of label column
    - text_column: Name of text column (optional)
    - experiment_name: Experiment name (optional)
    
    Returns: job_id for tracking
    """
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'File must be CSV format'}), 400
        
        # Get parameters
        label_column = request.form.get('label_column')
        text_column = request.form.get('text_column')
        experiment_name = request.form.get('experiment_name', 
                                          datetime.now().strftime("%Y%m%d_%H%M%S"))
        
        if not label_column:
            return jsonify({'error': 'label_column parameter required'}), 400
        
        # Save uploaded file
        job_id = str(uuid.uuid4())
        filename = secure_filename(f"{job_id}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Create job
        job = Job(job_id)
        JOBS[job_id] = job
        
        # Run pipeline in background
        thread = threading.Thread(
            target=_run_pipeline_job,
            args=(job_id, filepath, label_column, text_column, experiment_name)
        )
        thread.daemon = True
        thread.start()
        
        logger.info(f"Started job {job_id}")
        
        return jsonify({
            'job_id': job_id,
            'status': 'pending',
            'message': 'Pipeline job started',
            'check_status_url': f'/api/v1/jobs/{job_id}',
        }), 202
    
    except Exception as e:
        logger.error(f"Error in pipeline_run: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


def _run_pipeline_job(job_id: str, filepath: str, label_column: str, 
                      text_column: str, experiment_name: str):
    """Run pipeline job in background"""
    job = JOBS[job_id]
    
    try:
        job.status = "running"
        job.started_at = datetime.now()
        job.progress = 10
        
        # Initialize pipeline
        pipeline = AutoMLPipeline(output_dir=app.config['RESULTS_FOLDER'])
        
        job.progress = 20
        
        # Run pipeline
        result = pipeline.run(
            csv_path=filepath,
            label_column=label_column,
            text_column=text_column,
            experiment_name=experiment_name
        )
        
        job.progress = 90
        
        # Handle result
        if result.get('status') == 'success':
            # Convert non-serializable objects
            result_copy = result.copy()
            if 'comparison_df' in result_copy:
                result_copy['comparison_df'] = result_copy['comparison_df'].to_dict()
            if 'data_analysis' in result_copy:
                # Simplify analysis for JSON
                analysis = result_copy['data_analysis']
                for key in ['class_weights']:
                    if key in analysis.get('task_info', {}):
                        analysis['task_info'][key] = str(analysis['task_info'][key])
            
            job.result = result_copy
            job.status = "completed"
            job.progress = 100
            logger.info(f"Job {job_id} completed successfully")
        else:
            job.error = result.get('error', 'Unknown error')
            job.status = "failed"
            logger.error(f"Job {job_id} failed: {job.error}")
        
    except Exception as e:
        job.error = str(e)
        job.status = "failed"
        logger.error(f"Job {job_id} error: {str(e)}", exc_info=True)
    
    finally:
        job.completed_at = datetime.now()
        
        # Cleanup uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route('/api/v1/jobs/<job_id>', methods=['GET'])
def get_job_status(job_id: str):
    """Get job status"""
    if job_id not in JOBS:
        return jsonify({'error': 'Job not found'}), 404
    
    job = JOBS[job_id]
    return jsonify(job.to_dict()), 200


@app.route('/api/v1/jobs/<job_id>/results', methods=['GET'])
def get_job_results(job_id: str):
    """Get job results (when completed)"""
    if job_id not in JOBS:
        return jsonify({'error': 'Job not found'}), 404
    
    job = JOBS[job_id]
    
    if job.status != 'completed':
        return jsonify({
            'error': f'Job not completed. Current status: {job.status}',
            'progress': job.progress
        }), 202  # 202 Accepted
    
    if job.error:
        return jsonify({'error': job.error}), 500
    
    return jsonify({
        'job_id': job_id,
        'status': job.status,
        'result': job.result,
    }), 200


@app.route('/api/v1/jobs', methods=['GET'])
def list_jobs():
    """List all jobs"""
    jobs_list = [job.to_dict() for job in JOBS.values()]
    
    return jsonify({
        'total_jobs': len(jobs_list),
        'jobs': jobs_list,
    }), 200


@app.route('/api/v1/jobs/<job_id>/report', methods=['GET'])
def download_report(job_id: str):
    """Download report for completed job"""
    if job_id not in JOBS:
        return jsonify({'error': 'Job not found'}), 404
    
    job = JOBS[job_id]
    
    if job.status != 'completed' or not job.result:
        return jsonify({'error': 'Job not completed or no results'}), 400
    
    try:
        experiment_dir = job.result.get('experiment_dir')
        report_path = os.path.join(experiment_dir, 'best_model_report.txt')
        
        if not os.path.exists(report_path):
            return jsonify({'error': 'Report not found'}), 404
        
        return send_file(report_path, as_attachment=True, 
                        download_name=f"report_{job_id}.txt")
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/data-preview', methods=['POST'])
def data_preview():
    """Preview CSV file before running pipeline"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Read CSV
        df = pd.read_csv(file)
        
        return jsonify({
            'columns': df.columns.tolist(),
            'shape': {'rows': len(df), 'cols': len(df.columns)},
            'preview': df.head(5).to_dict('records'),
            'dtypes': df.dtypes.astype(str).to_dict(),
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# Root endpoint
@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API documentation"""
    return jsonify({
        'app': 'AutoML Text Classification API',
        'version': '0.1.0',
        'documentation': 'See /api/v1/pipeline/info for available endpoints',
        'endpoints': {
            'health': 'GET /health',
            'info': 'GET /api/v1/pipeline/info',
            'run_pipeline': 'POST /api/v1/pipeline/run',
            'job_status': 'GET /api/v1/jobs/{job_id}',
            'job_results': 'GET /api/v1/jobs/{job_id}/results',
            'list_jobs': 'GET /api/v1/jobs',
            'download_report': 'GET /api/v1/jobs/{job_id}/report',
            'preview_data': 'POST /api/v1/data-preview',
        }
    }), 200


if __name__ == '__main__':
    logger.info("Starting AutoML Flask API...")
    app.run(host='0.0.0.0', port=5000, debug=False)
