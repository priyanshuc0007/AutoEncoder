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
import statistics
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
                
                # Time inference — synchronize GPU before/after so we measure
                # actual compute time, not just async kernel dispatch time.
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.time()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                inference_time = (time.time() - start_time) / len(input_ids)  # per sample
                inference_times.append(inference_time)
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(batch_labels.cpu().numpy())
        
        # --- Single-sample latency: one request at a time, tokenization included ---
        # This is the metric that matters for deployment — not batch throughput ÷ N.
        # We run 1 warmup (discarded) + up to 10 timed calls, each on a single text.
        _n_timing = min(10, len(texts))
        _single_times: List[float] = []
        with torch.no_grad():
            # Warmup — discarded to exclude cold CUDA kernel compilation overhead
            _wenc = tokenizer(
                str(texts[0]),
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _ = model(
                input_ids=_wenc['input_ids'].to(self.device),
                attention_mask=_wenc['attention_mask'].to(self.device),
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            # Timed single-sample runs
            for _txt in texts[:_n_timing]:
                _enc = tokenizer(
                    str(_txt),
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt',
                )
                _ids  = _enc['input_ids'].to(self.device)
                _mask = _enc['attention_mask'].to(self.device)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _t0 = time.time()
                _ = model(input_ids=_ids, attention_mask=_mask)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _single_times.append(time.time() - _t0)
        single_sample_latency_ms = (
            statistics.mean(_single_times) * 1000 if _single_times else 0.0
        )

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
            'avg_inference_time_ms': avg_inference_time * 1000,  # batch throughput (ms/sample)
            'single_sample_latency_ms': single_sample_latency_ms,  # real single-request latency
            'predictions': all_preds,
            'true_labels': all_labels,
            'confusion_matrix': confusion_matrix(all_labels, all_preds),
        }
        
        logger.info(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        logger.info(
            f"Batch throughput: {avg_inference_time*1000:.2f}ms/sample  |  "
            f"Single-sample: {single_sample_latency_ms:.2f}ms"
        )

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
        # Latency reference is set to the MEDIAN latency across all models so
        # the bonus/penalty is relative — no single model architecture receives
        # a structural advantage just by being small/fast.
        # Use single_sample_latency_ms for composite score — it reflects real deployment
        # latency (one request + tokenization, CUDA sync'd). Fall back to batch throughput
        # if single-sample latency is absent (e.g. results built by older code).
        latency_key = (
            'single_sample_latency_ms'
            if all('single_sample_latency_ms' in r for r in results_list)
            else 'avg_inference_time_ms'
        )
        latencies = [r[latency_key] for r in results_list]
        # statistics.median() is correct for all N — sorted()[N//2] returns max for N=2
        median_latency = statistics.median(latencies) if latencies else 100.0
        median_latency = max(median_latency, 1.0)  # guard against zero
        for result in results_list:
            f1_component = result['f1_score'] * 0.7
            latency_component = (1.0 / (1.0 + result[latency_key] / median_latency)) * 0.3
            result['composite_score'] = f1_component + latency_component
        
        # Sort by composite score
        sorted_results = sorted(results_list, key=lambda x: x['composite_score'], reverse=True)
        
        # Print comparison table
        print(f"{'Model':<30} {'F1':<8} {'Accuracy':<10} {'Batch(ms)':<12} {'Single(ms)':<12} {'Score':<8}")
        print("-" * 88)

        for i, result in enumerate(sorted_results):
            marker = "🏆 BEST" if i == 0 else ""
            model_name = result.get('model_name', 'Unknown')[:28]
            single_lat = result.get('single_sample_latency_ms', result['avg_inference_time_ms'])
            print(f"{model_name:<30} {result['f1_score']:<8.4f} {result['accuracy']:<10.4f} "
                  f"{result['avg_inference_time_ms']:<12.2f} {single_lat:<12.2f} "
                  f"{result['composite_score']:<8.4f} {marker}")
        
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
                'Batch Latency (ms)': result['avg_inference_time_ms'],
                'Single-Sample Latency (ms)': result.get('single_sample_latency_ms', result['avg_inference_time_ms']),
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
