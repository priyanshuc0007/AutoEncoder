"""
Evaluator Module
Evaluates trained models and selects the best one
"""

import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
import logging
import time
from typing import Dict, List, Tuple
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """PyTorch Dataset for text classification"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class ModelEvaluator:
    """Evaluates and compares trained models"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def evaluate_model(
        self,
        model_path: str,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int,
        batch_size: int = 32,
        label_encoder = None,
    ) -> Dict:
        """
        Evaluate a single model
        
        Args:
            model_path: Path to saved model
            texts: List of text samples
            labels: List of labels
            tokenizer: Tokenizer
            max_length: Max sequence length
            batch_size: Batch size for evaluation
            label_encoder: Label encoder for decoding
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating model from {model_path}")
        
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        model.eval()
        
        # Create dataset and loader
        dataset = TextDataset(texts, labels, tokenizer, max_length)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Inference
        all_preds = []
        all_labels = []
        inference_times = []
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                batch_labels = batch['labels'].to(self.device)
                
                # Time inference
                start_time = time.time()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                inference_time = (time.time() - start_time) / len(input_ids)  # per sample
                inference_times.append(inference_time)
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(batch_labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        avg_inference_time = np.mean(inference_times)
        
        results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'avg_inference_time_ms': avg_inference_time * 1000,  # Convert to ms
            'predictions': all_preds,
            'true_labels': all_labels,
            'confusion_matrix': confusion_matrix(all_labels, all_preds),
        }
        
        logger.info(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        logger.info(f"Avg inference time: {avg_inference_time*1000:.2f}ms per sample")
        
        return results
    
    def compare_models(self, results_list: List[Dict]) -> Dict:
        """
        Compare multiple model results and select best
        
        Args:
            results_list: List of evaluation results for each model
            
        Returns:
            Dictionary with comparison and best model info
        """
        logger.info(f"\n{'='*70}")
        logger.info("MODEL COMPARISON")
        logger.info(f"{'='*70}\n")
        
        # Handle empty results
        if not results_list:
            logger.warning("⚠️ No models were successfully trained for comparison")
            return {
                'best_model': None,
                'all_results_sorted': [],
                'comparison_df': pd.DataFrame(),
            }
        
        # Calculate composite score
        for result in results_list:
            # Weighted score: 70% F1, 30% (inverse of latency)
            # Higher latency → lower score
            f1_component = result['f1_score'] * 0.7
            latency_component = (1.0 / (1.0 + result['avg_inference_time_ms'] / 100)) * 0.3
            result['composite_score'] = f1_component + latency_component
        
        # Sort by composite score
        sorted_results = sorted(results_list, key=lambda x: x['composite_score'], reverse=True)
        
        # Print comparison table
        print(f"{'Model':<30} {'F1':<8} {'Accuracy':<10} {'Latency(ms)':<12} {'Score':<8}")
        print("-" * 70)
        
        for i, result in enumerate(sorted_results):
            marker = "🏆 BEST" if i == 0 else ""
            model_name = result.get('model_name', 'Unknown')[:28]
            print(f"{model_name:<30} {result['f1_score']:<8.4f} {result['accuracy']:<10.4f} "
                  f"{result['avg_inference_time_ms']:<12.2f} {result['composite_score']:<8.4f} {marker}")
        
        best_model = sorted_results[0]
        
        comparison = {
            'best_model': best_model,
            'all_results_sorted': sorted_results,
            'comparison_df': self._create_comparison_df(sorted_results),
        }
        
        logger.info(f"\n✓ Best model selected: {best_model.get('model_name', 'Unknown')}")
        logger.info(f"  F1 Score: {best_model['f1_score']:.4f}")
        logger.info(f"  Composite Score: {best_model['composite_score']:.4f}")
        
        return comparison
    
    def _create_comparison_df(self, results_list: List[Dict]) -> pd.DataFrame:
        """Create comparison dataframe"""
        data = []
        for result in results_list:
            data.append({
                'Model': result.get('model_name', 'Unknown'),
                'F1 Score': result['f1_score'],
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'Latency (ms)': result['avg_inference_time_ms'],
                'Composite Score': result['composite_score'],
            })
        
        return pd.DataFrame(data)
    
    def generate_report(
        self,
        best_model_result: Dict,
        label_encoder,
        output_path: str = "experiments/best_model_report.txt"
    ) -> None:
        """
        Generate evaluation report
        
        Args:
            best_model_result: Best model evaluation result
            label_encoder: Label encoder for class names
            output_path: Where to save report
        """
        logger.info(f"Generating report to {output_path}")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("BEST MODEL EVALUATION REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Model: {best_model_result.get('model_name', 'Unknown')}\n\n")
            
            f.write("METRICS:\n")
            f.write(f"  F1 Score: {best_model_result['f1_score']:.4f}\n")
            f.write(f"  Accuracy: {best_model_result['accuracy']:.4f}\n")
            f.write(f"  Precision: {best_model_result['precision']:.4f}\n")
            f.write(f"  Recall: {best_model_result['recall']:.4f}\n")
            f.write(f"  Avg Latency: {best_model_result['avg_inference_time_ms']:.2f}ms per sample\n\n")
            
            # Classification report
            f.write("CLASSIFICATION REPORT:\n")
            class_report = classification_report(
                best_model_result['true_labels'],
                best_model_result['predictions'],
                target_names=label_encoder.classes_,
                zero_division=0
            )
            f.write(class_report + "\n\n")
            
            # Confusion matrix
            f.write("CONFUSION MATRIX:\n")
            cm = best_model_result['confusion_matrix']
            f.write(str(cm) + "\n")
        
        logger.info(f"Report saved to {output_path}")
