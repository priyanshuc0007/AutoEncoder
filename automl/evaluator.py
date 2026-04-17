"""
Evaluator Module
Evaluates trained models and selects the best one
"""

import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
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

from automl.dataset import TextDataset  # shared dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        split: str = 'val',
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

        # Guard against a missing or empty model directory (e.g. disk-full during training)
        model_dir = Path(model_path)
        if not model_dir.exists():
            raise ValueError(f"Model path does not exist: {model_path}")
        if not any(model_dir.iterdir()):
            raise ValueError(f"Model path is empty (training may have failed to save): {model_path}")

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
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)

        # Weighted mean: weight each batch by its actual sample count to avoid
        # the final (smaller) batch skewing the per-sample latency estimate
        total_samples = len(all_preds)
        # Recompute from loader length (batch records were appended per-batch)
        batch_cursor = 0
        weighted_latency_sum = 0.0
        for i, t in enumerate(inference_times):
            bs = batch_size if (batch_cursor + batch_size) <= total_samples else (total_samples - batch_cursor)
            weighted_latency_sum += t * bs
            batch_cursor += bs
        avg_inference_time = weighted_latency_sum / total_samples if total_samples > 0 else 0.0
        
        results = {
            'split': split,
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

        # Free model memory immediately — each call loads a fresh model instance and
        # keeping them alive causes GPU OOM when evaluating multiple models in sequence.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        output_path: str = "experiments/best_model_report.txt",
        train_result: Dict = None,
    ) -> None:
        """
        Generate evaluation report with optional train vs val comparison.

        Args:
            best_model_result: Validation evaluation result (the best model)
            label_encoder: Label encoder for class names
            output_path: Where to save report
            train_result: Training set evaluation result (optional, for overfitting analysis)
        """
        logger.info(f"Generating report to {output_path}")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("BEST MODEL EVALUATION REPORT\n")
            f.write("="*70 + "\n\n")

            f.write(f"Model: {best_model_result.get('model_name', 'Unknown')}\n\n")

            # ── Side-by-side train vs val metrics ─────────────────────────────
            if train_result is not None:
                f.write("="*70 + "\n")
                f.write("TRAIN vs VALIDATION METRICS (overfitting check)\n")
                f.write("="*70 + "\n")
                f.write(f"{'Metric':<20} {'Train':>10} {'Validation':>12} {'Gap':>8}\n")
                f.write("-"*54 + "\n")

                metrics = [
                    ('F1 Score',  train_result['f1_score'],  best_model_result['f1_score']),
                    ('Accuracy',  train_result['accuracy'],  best_model_result['accuracy']),
                    ('Precision', train_result['precision'], best_model_result['precision']),
                    ('Recall',    train_result['recall'],    best_model_result['recall']),
                ]
                for name, tr, vl in metrics:
                    gap = tr - vl
                    flag = '  ⚠️ overfit' if gap > 0.05 else ('  ✅' if abs(gap) <= 0.05 else '  📉 underfit')
                    f.write(f"{name:<20} {tr:>10.4f} {vl:>12.4f} {gap:>+8.4f}{flag}\n")

                f.write("\n")
                # Diagnosis
                avg_gap = train_result['f1_score'] - best_model_result['f1_score']
                if avg_gap > 0.10:
                    diagnosis = "HIGH OVERFITTING — model memorised training data. Consider more regularisation, dropout, or more data."
                elif avg_gap > 0.05:
                    diagnosis = "MILD OVERFITTING — slight gap. Monitor and consider early stopping or weight decay."
                elif avg_gap < -0.05:
                    diagnosis = "UNDERFITTING — val F1 > train F1. Model may not have converged. Try more epochs or a lower LR."
                else:
                    diagnosis = "GOOD FIT — train and val metrics are close. Model generalises well."
                f.write(f"Diagnosis: {diagnosis}\n\n")
            else:
                f.write("VALIDATION METRICS:\n")
                f.write(f"  F1 Score:  {best_model_result['f1_score']:.4f}\n")
                f.write(f"  Accuracy:  {best_model_result['accuracy']:.4f}\n")
                f.write(f"  Precision: {best_model_result['precision']:.4f}\n")
                f.write(f"  Recall:    {best_model_result['recall']:.4f}\n")
                f.write(f"  Avg Latency: {best_model_result['avg_inference_time_ms']:.2f}ms per sample\n\n")

            # ── Validation classification report ──────────────────────────────
            f.write("VALIDATION CLASSIFICATION REPORT:\n")
            val_class_report = classification_report(
                best_model_result['true_labels'],
                best_model_result['predictions'],
                target_names=label_encoder.classes_,
                zero_division=0
            )
            f.write(val_class_report + "\n")

            f.write("VALIDATION CONFUSION MATRIX:\n")
            f.write(str(best_model_result['confusion_matrix']) + "\n")

            # ── Training classification report (if available) ─────────────────
            if train_result is not None:
                f.write("\n" + "="*70 + "\n")
                f.write("TRAINING CLASSIFICATION REPORT:\n")
                train_class_report = classification_report(
                    train_result['true_labels'],
                    train_result['predictions'],
                    target_names=label_encoder.classes_,
                    zero_division=0
                )
                f.write(train_class_report + "\n")

                f.write("TRAINING CONFUSION MATRIX:\n")
                f.write(str(train_result['confusion_matrix']) + "\n")

        logger.info(f"Report saved to {output_path}")
