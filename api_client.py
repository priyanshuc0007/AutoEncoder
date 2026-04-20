"""
Python Client for AutoLLM Flask API
Example of how to use the API programmatically
"""

import requests
import pandas as pd
import time
import json
from pathlib import Path


class AutoLLMClient:
    """Client for interacting with AutoLLM Flask API"""
    
    def __init__(self, base_url='http://localhost:5000'):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self):
        """Check API health"""
        try:
            response = self.session.get(f'{self.base_url}/health')
            return response.json()
        except Exception as e:
            print(f"❌ API not available: {str(e)}")
            return None
    
    def get_info(self):
        """Get API information"""
        response = self.session.get(f'{self.base_url}/api/v1/pipeline/info')
        return response.json()
    
    def run_pipeline(self, csv_path, label_column, text_column=None, 
                     experiment_name=None):
        """
        Run AutoLLM pipeline
        
        Args:
            csv_path: Path to CSV file
            label_column: Label column name
            text_column: Text column name (optional)
            experiment_name: Experiment name (optional)
            
        Returns:
            Experiment ID
        """
        with open(csv_path, 'rb') as f:
            files = {'file': f}
            data = {
                'label_column': label_column,
                'text_column': text_column or '',
                'experiment_name': experiment_name or '',
            }
            
            response = self.session.post(
                f'{self.base_url}/api/v1/pipeline/run',
                files=files,
                data=data
            )
        
        if response.status_code in [200, 202]:
            result = response.json()
            job_id = result.get('job_id')
            print(f"✓ Pipeline started: {job_id}")
            return job_id
        else:
            print(f"❌ Failed to start pipeline: {response.text}")
            return None
    
    def get_status(self, job_id):
        """Get pipeline status"""
        response = self.session.get(
            f'{self.base_url}/api/v1/jobs/{job_id}'
        )
        return response.json()
    
    def get_results(self, job_id):
        """Get pipeline results"""
        response = self.session.get(
            f'{self.base_url}/api/v1/jobs/{job_id}/results'
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {'error': response.json().get('error')}
    
    def wait_for_completion(self, job_id, max_wait=3600, poll_interval=30):
        """
        Wait for pipeline to complete

        Args:
            job_id: Job ID returned by run_pipeline()
            max_wait: Maximum wait time in seconds
            poll_interval: Poll interval in seconds
        """
        start_time = time.time()

        while True:
            status = self.get_status(job_id)
            current = status.get('status')
            print(f"Status: {current}")

            # Server sets status to 'completed' (not 'success') on success
            if current == 'completed':
                print("✓ Pipeline completed successfully!")
                return self.get_results(job_id)
            elif current == 'failed':
                print("❌ Pipeline failed!")
                return status

            elapsed = time.time() - start_time
            if elapsed > max_wait:
                print(f"⏱️ Timeout after {elapsed:.0f}s")
                return None

            print(f"⏳ Waiting... ({elapsed:.0f}s elapsed)")
            time.sleep(poll_interval)
    
    def list_experiments(self):
        """List all experiments"""
        response = self.session.get(f'{self.base_url}/api/v1/jobs')
        return response.json()
    
    def get_report(self, job_id):
        """Get experiment report"""
        response = self.session.get(
            f'{self.base_url}/api/v1/jobs/{job_id}/report'
        )
        
        if response.status_code == 200:
            return response.json()['report']
        else:
            return None
    
    def download_model(self, experiment_id, output_path='model.zip'):
        """Download trained model"""
        response = self.session.get(
            f'{self.base_url}/api/pipeline/{experiment_id}/download-model'
        )
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"✓ Model saved to {output_path}")
            return True
        else:
            print(f"❌ Failed to download model: {response.text}")
            return False
    
    def predict(self, experiment_id, texts):
        """
        Make predictions
        
        Args:
            experiment_id: Experiment ID
            texts: Single text (str) or list of texts
        """
        if isinstance(texts, str):
            texts = [texts]
        
        payload = {'texts': texts}
        
        response = self.session.post(
            f'{self.base_url}/api/predict/{experiment_id}',
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()['predictions']
        else:
            print(f"❌ Prediction failed: {response.text}")
            return None


def main():
    """Example usage of AutoLLM client"""
    
    print("\n" + "="*70)
    print("🚀 AutoLLM API Client Example")
    print("="*70 + "\n")
    
    # Initialize client
    client = AutoLLMClient(base_url='http://localhost:5000')
    
    # Check health
    print("1️⃣  Checking API health...")
    health = client.health_check()
    if health:
        print(f"✓ API is healthy: {health['status']}\n")
    else:
        print("❌ API is not available. Make sure Flask app is running!")
        return
    
    # Get info
    print("2️⃣  Getting API information...")
    info = client.get_info()
    print(f"API Name: {info['name']}")
    print(f"Version: {info['version']}\n")
    
    # Check if sample data exists
    sample_csv = 'data/sample_dataset.csv'
    if not Path(sample_csv).exists():
        print(f"❌ Sample CSV not found at {sample_csv}")
        print("Please run from project root directory")
        return
    
    # Run pipeline
    print("3️⃣  Running AutoLLM pipeline...")
    print(f"Using data from: {sample_csv}")
    
    exp_id = client.run_pipeline(
        csv_path=sample_csv,
        label_column='label',
        text_column='text',
        experiment_name='client_example'
    )
    
    if not exp_id:
        print("Failed to start pipeline")
        return
    
    print(f"Experiment ID: {exp_id}\n")
    
    # Wait for completion
    print("4️⃣  Waiting for pipeline to complete...")
    print("⏳ This may take several minutes...")
    results = client.wait_for_completion(exp_id, max_wait=1800, poll_interval=30)
    
    if results:
        # Display results
        print("\n" + "="*70)
        print("📊 RESULTS")
        print("="*70 + "\n")
        
        best_model = results.get('best_model', {})
        metrics = best_model.get('metrics', {}) or {}
        
        f1 = metrics.get('f1_score')
        acc = metrics.get('accuracy')
        lat = metrics.get('latency_ms')

        print(f"Best Model: {best_model.get('name')}")
        print(f"F1 Score: {f1:.4f}" if f1 is not None else "F1 Score: N/A")
        print(f"Accuracy: {acc:.4f}" if acc is not None else "Accuracy: N/A")
        print(f"Latency: {lat:.2f}ms\n" if lat is not None else "Latency: N/A\n")
        
        # Get report
        print("5️⃣  Downloading experiment report...")
        report = client.get_report(exp_id)
        if report:
            print("✓ Report retrieved")
            # Save report
            report_path = f"experiments/report_{exp_id}.txt"
            Path(report_path).parent.mkdir(exist_ok=True)
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"Saved to: {report_path}\n")
        
        # Download model
        print("6️⃣  Downloading trained model...")
        model_path = f"models/model_{exp_id}.zip"
        Path(model_path).parent.mkdir(exist_ok=True)
        client.download_model(exp_id, model_path)
        print()
        
        # Make predictions
        print("7️⃣  Making predictions...")
        test_texts = [
            "This is absolutely amazing!",
            "Terrible experience, would not recommend.",
            "Love it!",
        ]
        
        predictions = client.predict(exp_id, test_texts)
        if predictions:
            print("\nPredictions:")
            for pred in predictions:
                print(f"  Text: {pred['text'][:40]}...")
                print(f"  Class: {pred['prediction']}, Confidence: {pred['confidence']:.4f}\n")
        
        # List all experiments
        print("8️⃣  Listing all experiments...")
        exps = client.list_experiments()
        print(f"Total experiments: {exps['total']}")
        for exp in exps['experiments'][:3]:  # Show first 3
            print(f"  - {exp['experiment_id']}: {exp['status']} "
                  f"(F1: {exp['f1_score']:.4f if exp['f1_score'] else 'N/A'})")
        
        print("\n" + "="*70)
        print("✅ Example completed successfully!")
        print("="*70 + "\n")


if __name__ == '__main__':
    main()
