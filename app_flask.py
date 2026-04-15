
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
        try:
            df = pd.read_csv(file, encoding='utf-8')
        except UnicodeDecodeError:
            file.seek(0)
            df = pd.read_csv(file, encoding='latin-1')
        
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
