"""
Data Intelligence Engine
Analyzes dataset and makes intelligent decisions about:
- Task type (binary vs multiclass)
- Class imbalance
- Text length
- Model selection
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIntelligence:
    """Analyzes dataset and provides intelligence for AutoML decisions"""
    
    def __init__(self):
        pass
    
    def analyze(self, df: pd.DataFrame, label_column: str, text_column: str) -> Dict:
        """
        Comprehensive dataset analysis
        
        Args:
            df: DataFrame
            label_column: Label column name
            text_column: Text column name
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {}
        
        # 1. Task Detection
        analysis['task_info'] = self._detect_task(df, label_column)
        
        # 2. Class Imbalance Detection
        analysis['imbalance_info'] = self._detect_imbalance(df, label_column)
        
        # 3. Text Length Analysis
        analysis['text_info'] = self._analyze_text_length(df, text_column)
        
        # 4. Model Selection
        analysis['model_selection'] = self._select_models(len(df), analysis['imbalance_info'])
        
        # 5. Training Configuration
        analysis['training_config'] = self._get_training_config(
            len(df), 
            analysis['imbalance_info'],
            analysis['task_info']
        )
        
        return analysis
    
    def _detect_task(self, df: pd.DataFrame, label_column: str) -> Dict:
        """
        Detect if task is binary or multiclass
        
        Args:
            df: DataFrame
            label_column: Label column name
            
        Returns:
            Dictionary with task information
        """
        num_classes = df[label_column].nunique()
        task_type = "binary" if num_classes == 2 else "multiclass"
        
        class_dist = df[label_column].value_counts()
        
        info = {
            'task_type': task_type,
            'num_classes': num_classes,
            'class_distribution': class_dist.to_dict(),
            'class_weights': self._compute_class_weights(df[label_column]),
        }
        
        logger.info(f"Task detected: {task_type} ({num_classes} classes)")
        logger.info(f"Class distribution: {class_dist.to_dict()}")
        
        return info
    
    def _detect_imbalance(self, df: pd.DataFrame, label_column: str) -> Dict:
        """
        Detect class imbalance and recommend handling strategies
        
        Args:
            df: DataFrame
            label_column: Label column name
            
        Returns:
            Dictionary with imbalance information
        """
        class_counts = df[label_column].value_counts()
        max_count = class_counts.max()
        min_count = class_counts.min()
        imbalance_ratio = max_count / min_count
        
        info = {
            'imbalance_ratio': imbalance_ratio,
            'use_class_weights': imbalance_ratio > 2.0,
            'use_focal_loss': imbalance_ratio > 5.0,
            'minority_class_ratio': min_count / len(df),
        }
        
        logger.info(f"Imbalance ratio: {imbalance_ratio:.2f}")
        if info['use_class_weights']:
            logger.info("✓ Will use class weights for handling imbalance")
        if info['use_focal_loss']:
            logger.info("✓ Dataset is highly imbalanced (ratio > 5), focal loss recommended")
        
        return info
    
    def _analyze_text_length(self, df: pd.DataFrame, text_column: str) -> Dict:
        """
        Analyze text length distribution
        
        Args:
            df: DataFrame
            text_column: Text column name
            
        Returns:
            Dictionary with text length information
        """
        text_lengths = df[text_column].astype(str).str.len()
        
        # Use 95th percentile as max_length (avoids truncation)
        max_length = int(np.percentile(text_lengths, 95))
        avg_length = text_lengths.mean()
        
        info = {
            'avg_length': avg_length,
            'max_length': max_length,
            'min_length': text_lengths.min(),
            'p95_length': max_length,
            'p99_length': int(np.percentile(text_lengths, 99)),
        }
        
        logger.info(f"Text length - Min: {info['min_length']}, Avg: {avg_length:.0f}, Max: {info['max_length']}")
        
        return info
    
    def _select_models(self, dataset_size: int, imbalance_info: Dict) -> List[str]:
        """
        Select appropriate models based on dataset size
        Uses open-source models available on HuggingFace
        
        Args:
            dataset_size: Number of samples
            imbalance_info: Imbalance information
            
        Returns:
            List of model names to train
        """
        if dataset_size < 2000:
            # Lightweight open-source model
            models = ["distilbert-base-uncased"]
            logger.info(f"Small dataset ({dataset_size} samples) → DistilBERT (open source)")
        elif dataset_size < 10000:
            # Medium open-source model
            models = ["bert-base-uncased"]
            logger.info(f"Medium dataset ({dataset_size} samples) → BERT (open source)")
        else:
            # Larger open-source models
            models = ["bert-base-uncased", "distilbert-base-uncased"]
            logger.info(f"Large dataset ({dataset_size} samples) → BERT + DistilBERT (open source)")
        
        return models
    
    def _get_training_config(self, dataset_size: int, imbalance_info: Dict, task_info: Dict) -> Dict:
        """
        Get dynamic training configuration based on ACTUAL data metrics
        Uses simple rules (IF/ELSE) based on samples-per-class, not hardcoded thresholds
        
        Args:
            dataset_size: Total number of samples
            imbalance_info: Imbalance information
            task_info: Task information with class distribution
            
        Returns:
            Dictionary with training configuration
        """
        # Calculate KEY METRICS from actual data
        num_classes = task_info['num_classes']
        class_dist = task_info['class_distribution']
        
        samples_per_class = dataset_size / num_classes
        min_class_size = min(class_dist.values())
        max_class_size = max(class_dist.values())
        imbalance_ratio = imbalance_info['imbalance_ratio']
        
        # Calculate coefficient of variation (how uneven is the distribution)
        class_sizes = list(class_dist.values())
        mean_size = np.mean(class_sizes)
        std_size = np.std(class_sizes)
        cv = std_size / mean_size if mean_size > 0 else 0
        
        logger.info(f"\n{'='*70}")
        logger.info("📊 DATA METRICS ANALYSIS")
        logger.info(f"{'='*70}")
        logger.info(f"Total Samples: {dataset_size}")
        logger.info(f"Number of Classes: {num_classes}")
        logger.info(f"Samples per Class: {samples_per_class:.1f} (min: {min_class_size}, max: {max_class_size})")
        logger.info(f"Imbalance Ratio: {imbalance_ratio:.2f}x")
        logger.info(f"Distribution Variance: {cv:.2f}")
        
        # SIMPLE RULE-BASED LOGIC based on samples_per_class
        # This handles ANY dataset size and number of classes robustly
        
        if samples_per_class < 50:
            # CRITICAL: Very few samples per class
            strategy = "CRITICAL"
            num_epochs_range = [15, 20]
            batch_size = 4
            learning_rate = 5e-4
            gradient_accumulation_steps = 2
            warmup_steps = 10
            early_stopping_patience = 7
            weight_decay = 0.02  # Stronger regularization
            max_grad_norm = 0.5  # Tighter gradient clipping
            
            logger.info(f"\n⚠️  CRITICAL STRATEGY: Only {samples_per_class:.1f} samples/class")
            logger.info(f"    → Extreme overfitting risk")
            logger.info(f"    → Using aggressive regularization")
            logger.info(f"    → Action: Consider collecting more data or data augmentation")
            
        elif samples_per_class < 200:
            # SMALL: Limited data, needs regularization
            strategy = "SMALL"
            num_epochs_range = [10, 15]
            batch_size = 8
            learning_rate = 2e-4
            gradient_accumulation_steps = 2
            warmup_steps = 20
            early_stopping_patience = 6
            weight_decay = 0.015
            max_grad_norm = 0.75
            
            logger.info(f"\n🟡 SMALL DATA STRATEGY: {samples_per_class:.1f} samples/class")
            logger.info(f"    → Enhanced training with regularization")
            logger.info(f"    → More epochs to learn patterns effectively")
            
        elif samples_per_class < 500:
            # MODERATE: Manageable but still careful
            strategy = "MODERATE"
            num_epochs_range = [8, 12]
            batch_size = 12
            learning_rate = 1e-4
            gradient_accumulation_steps = 1
            warmup_steps = 50
            early_stopping_patience = 4
            weight_decay = 0.01
            max_grad_norm = 1.0
            
            logger.info(f"\n🟢 MODERATE DATA STRATEGY: {samples_per_class:.1f} samples/class")
            logger.info(f"    → Standard training with careful monitoring")
            
        else:
            # GOOD: Sufficient data
            strategy = "GOOD"
            num_epochs_range = [5, 8]
            batch_size = 16
            learning_rate = 1e-4
            gradient_accumulation_steps = 1
            warmup_steps = 100
            early_stopping_patience = 3
            weight_decay = 0.01
            max_grad_norm = 1.0
            
            logger.info(f"\n✅ GOOD DATA STRATEGY: {samples_per_class:.1f} samples/class")
            logger.info(f"    → Standard fine-tuning approach")
        
        # ADJUST for severe imbalance even if samples_per_class is decent
        if imbalance_ratio > 10:
            logger.info(f"\n⚠️  SEVERE IMBALANCE DETECTED: {imbalance_ratio:.1f}x")
            logger.info(f"    → Adjusting early stopping patience")
            early_stopping_patience = max(early_stopping_patience + 2, 5)
            weight_decay = weight_decay * 1.2
        
        config = {
            'strategy': strategy,
            'num_epochs_range': num_epochs_range,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'warmup_steps': warmup_steps,
            'weight_decay': weight_decay,
            'max_grad_norm': max_grad_norm,
            'early_stopping_patience': early_stopping_patience,
            'use_class_weights': imbalance_info['use_class_weights'],
            # Store metrics for reporting
            'samples_per_class': samples_per_class,
            'min_class_size': min_class_size,
            'imbalance_ratio': imbalance_ratio,
        }
        
        logger.info(f"\n{'='*70}")
        logger.info(f"📋 TRAINING CONFIG: {strategy} Strategy")
        logger.info(f"{'='*70}")
        logger.info(f"Epochs: {num_epochs_range} | Batch: {batch_size} | LR: {learning_rate}")
        logger.info(f"Grad Accum: {gradient_accumulation_steps} | Warmup: {warmup_steps}")
        logger.info(f"Weight Decay: {weight_decay} | Max Grad Norm: {max_grad_norm}")
        logger.info(f"Early Stopping Patience: {early_stopping_patience}")
        logger.info(f"{'='*70}\n")
        
        return config
    
    def _compute_class_weights(self, labels: pd.Series) -> Dict:
        """
        Compute class weights for handling imbalance
        
        Args:
            labels: Label series
            
        Returns:
            Dictionary mapping class to weight
        """
        class_counts = labels.value_counts()
        total = len(labels)
        weights = {}
        
        for class_label, count in class_counts.items():
            # Inverse frequency weighting
            weight = total / (len(class_counts) * count)
            weights[class_label] = weight
        
        # Normalize weights
        max_weight = max(weights.values())
        weights = {k: v/max_weight for k, v in weights.items()}
        
        return weights
    
    def print_summary(self, analysis: Dict) -> None:
        """
        Print analysis summary
        
        Args:
            analysis: Analysis dictionary
        """
        print("\n" + "="*60)
        print("📊 DATA INTELLIGENCE REPORT")
        print("="*60)
        
        # Task info
        task_info = analysis['task_info']
        print(f"\n🎯 Task: {task_info['task_type'].upper()}")
        print(f"   Classes: {task_info['num_classes']}")
        
        # Imbalance info
        imb = analysis['imbalance_info']
        print(f"\n⚖️  Imbalance Ratio: {imb['imbalance_ratio']:.2f}")
        print(f"   Use Class Weights: {'✓ YES' if imb['use_class_weights'] else '✗ NO'}")
        print(f"   Use Focal Loss: {'✓ YES' if imb['use_focal_loss'] else '✗ NO'}")
        
        # Text info
        text_info = analysis['text_info']
        print(f"\n📝 Text Analysis:")
        print(f"   Avg Length: {text_info['avg_length']:.0f}")
        print(f"   Max Length (p95): {text_info['p95_length']}")
        
        # Model selection
        print(f"\n🤖 Selected Models:")
        for model in analysis['model_selection']:
            print(f"   - {model}")
        
        # Training config
        config = analysis['training_config']
        print(f"\n⚙️  Training Config:")
        print(f"   Epochs: {config['num_epochs_range']}")
        print(f"   Batch Size: {config['batch_size']}")
        print(f"   Learning Rate: {config['learning_rate']}")
        
        print("="*60 + "\n")
