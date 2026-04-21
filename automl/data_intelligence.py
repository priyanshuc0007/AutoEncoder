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
    """Analyzes dataset and provides intelligence for AutoLLM decisions"""
    
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

        # Guard: if label column is numeric and cardinality is very high relative to the
        # dataset size, it is almost certainly a regression target, not a class label.
        # Heuristic: numeric dtype + more than 20% unique values + more than 20 unique values.
        label_is_numeric = pd.api.types.is_numeric_dtype(df[label_column])
        cardinality_ratio = num_classes / len(df)
        if label_is_numeric and num_classes > 20 and cardinality_ratio > 0.2:
            raise ValueError(
                f"Label column '{label_column}' looks like a continuous/regression target: "
                f"{num_classes} unique numeric values out of {len(df)} rows "
                f"({cardinality_ratio*100:.1f}% cardinality). "
                f"This pipeline only supports classification. "
                f"If these are genuinely discrete classes, cast the column to string before running."
            )

        # Warn when there are suspiciously many classes relative to samples
        if num_classes > 50:
            logger.warning(
                f"⚠️  Label column has {num_classes} unique classes on {len(df)} rows. "
                f"This is very high cardinality — verify that '{label_column}' is the correct label column."
            )

        # Detect whether label column contains categorical strings (e.g. spam/ham)
        # or numeric class IDs (e.g. 0/1/2).  By the time this runs, load_and_validate
        # has already normalised float/int columns to strings, so a purely numeric-string
        # column (all values parse as int) signals the original was numeric.
        all_numeric_strings = pd.to_numeric(df[label_column], errors='coerce').notna().all()
        label_type = "numeric" if all_numeric_strings else "categorical"
        logger.info(f"Label type: {label_type} ({'e.g. 0/1/2' if label_type == 'numeric' else 'e.g. spam/ham'})")

        task_type = "binary" if num_classes == 2 else "multiclass"

        class_dist = df[label_column].value_counts()

        info = {
            'task_type': task_type,
            'label_type': label_type,
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
        Analyze text length distribution using real tokenizer token counts.

        Loads bert-base-uncased (shared vocabulary for all pipeline models) to
        measure actual token lengths on up to 500 sampled texts, then sets
        max_length at p95 capped to 512.  Falls back to the char÷4 heuristic
        if the tokenizer cannot be loaded (e.g. offline environment).
        """
        texts = df[text_column].astype(str).tolist()

        # Sample for speed — tokenising 50 K emails just for analysis is slow
        _SAMPLE_SIZE = 500
        if len(texts) > _SAMPLE_SIZE:
            rng = np.random.default_rng(42)
            sample_texts = rng.choice(texts, size=_SAMPLE_SIZE, replace=False).tolist()
        else:
            sample_texts = texts

        token_lengths = None
        tokenizer_used = None

        try:
            from transformers import AutoTokenizer
            _tok = AutoTokenizer.from_pretrained(
                "bert-base-uncased",
                use_fast=True,
            )
            tokenizer_used = "bert-base-uncased"
            encodings = _tok(
                sample_texts,
                add_special_tokens=True,
                truncation=False,        # no truncation — we want real lengths
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            token_lengths = np.array([len(ids) for ids in encodings["input_ids"]])
            logger.info(
                f"Token lengths measured with {tokenizer_used} "
                f"on {len(sample_texts)} sampled texts"
            )
        except Exception as exc:
            logger.warning(
                f"Tokenizer unavailable for length analysis "
                f"(falling back to char÷4): {exc}"
            )

        if token_lengths is not None:
            p95_tokens = int(np.percentile(token_lengths, 95))
            p99_tokens = int(np.percentile(token_lengths, 99))
            avg_tokens  = float(token_lengths.mean())
            min_tokens  = int(token_lengths.min())
            max_length  = max(min(p95_tokens, 512), 16)

            # Warn early when emails are very long — users should know truncation happens
            if p95_tokens > 512:
                logger.warning(
                    f"⚠️  p95 token length = {p95_tokens} exceeds the 512 transformer "
                    f"hard limit — {int((token_lengths > 512).mean() * 100)}% of sampled "
                    f"texts will be truncated. Consider enabling chunked inference."
                )

            # Char stats kept for the transparency report
            char_lengths = np.array([len(t) for t in sample_texts])

            info = {
                'avg_length':      avg_tokens,
                'avg_char_length': float(char_lengths.mean()),
                'min_length':      min_tokens,
                'max_length':      max_length,
                'p95_length':      max_length,          # used as training max_length
                'p95_tokens_raw':  p95_tokens,          # before 512 cap — for reporting
                'p99_length':      min(p99_tokens, 512),
                'tokenizer_used':  tokenizer_used,
                'measurement':     'real_tokens',
            }
            logger.info(
                f"Token lengths — Min: {min_tokens}, Avg: {avg_tokens:.0f}, "
                f"p95: {p95_tokens}, p99: {p99_tokens} → max_length = {max_length}"
            )

        else:
            # Fallback: char ÷ 4 approximation
            char_lengths = np.array([len(t) for t in texts])
            p95_chars = int(np.percentile(char_lengths, 95))
            p99_chars = int(np.percentile(char_lengths, 99))
            max_length = max(min(int(p95_chars / 4), 512), 16)

            info = {
                'avg_length':      float(char_lengths.mean()),
                'avg_char_length': float(char_lengths.mean()),
                'min_length':      int(char_lengths.min()),
                'max_length':      max_length,
                'p95_length':      max_length,
                'p95_tokens_raw':  max_length,
                'p99_length':      max(min(int(p99_chars / 4), 512), 16),
                'tokenizer_used':  None,
                'measurement':     'char_div4_approx',
            }
            logger.info(
                f"Text length (char÷4 approx) — Avg: {info['avg_length']:.0f} chars "
                f"→ max_length = {max_length}"
            )

        return info
    
    def _select_models(self, dataset_size: int, imbalance_info: Dict) -> List[str]:
        """
        Select models based on dataset size.

        Full model roster (smallest → largest)
        -------------------------------------------
        prajjwal1/bert-tiny       4.4 M params  L=2  H=128  — fastest, lowest overfit risk
        prajjwal1/bert-mini      11.3 M params  L=4  H=256  — very fast, good for small data
        google/mobilebert-uncased 25 M params  L=24*        — mobile-optimised, fast inference
        distilbert-base-uncased   66 M params  L=6          — strong balance of speed & accuracy
        bert-base-uncased        110 M params  L=12         — strongest baseline

        (* MobileBERT uses bottleneck layers — 24 thin layers cost similar to ~6 standard layers)

        Tiers
        -----
        < 2 000 samples  : tiny + mini + mobilebert
                           (lighter models; bert-base tends to overfit on small data)
        2 000 – 10 000   : mini + mobilebert + distilbert + bert
                           (full comparison across the size spectrum)
        > 10 000 samples : mobilebert + distilbert + bert
                           (quality-focused; tiny/mini less competitive at scale)
        """
        if dataset_size < 2000:
            models = [
                "prajjwal1/bert-tiny",
                "prajjwal1/bert-mini",
                "google/mobilebert-uncased",
            ]
            logger.info(
                f"Small dataset ({dataset_size} samples) → "
                f"bert-tiny + bert-mini + mobilebert"
            )
        elif dataset_size < 10000:
            models = [
                "prajjwal1/bert-mini",
                "google/mobilebert-uncased",
                "distilbert-base-uncased",
                "bert-base-uncased",
            ]
            logger.info(
                f"Medium dataset ({dataset_size} samples) → "
                f"bert-mini + mobilebert + distilbert + bert"
            )
        else:
            models = [
                "google/mobilebert-uncased",
                "distilbert-base-uncased",
                "bert-base-uncased",
            ]
            logger.info(
                f"Large dataset ({dataset_size} samples) → "
                f"mobilebert + distilbert + bert"
            )

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
            warmup_ratio = 0.06
            early_stopping_patience = 7
            weight_decay = 0.02  # Stronger regularization
            max_grad_norm = 0.5  # Tighter gradient clipping
            lr_bounds = (2e-4, 5e-4)       # Optuna skips CRITICAL, but bounds set for completeness
            wd_bounds = (0.01, 0.04)
            lr_scheduler_type = 'cosine'
            dropout = 0.3
            label_smoothing_factor = 0.1
            
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
            warmup_ratio = 0.06
            early_stopping_patience = 6
            weight_decay = 0.015
            max_grad_norm = 0.75
            lr_bounds = (1e-4, 5e-4)       # Higher LR range — small data needs stronger signal
            wd_bounds = (0.005, 0.03)
            lr_scheduler_type = 'cosine'
            dropout = 0.2
            label_smoothing_factor = 0.1
            
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
            warmup_ratio = 0.06
            early_stopping_patience = 4
            weight_decay = 0.01
            max_grad_norm = 1.0
            lr_bounds = (5e-5, 2e-4)       # Standard fine-tuning range
            wd_bounds = (0.0, 0.02)
            lr_scheduler_type = 'linear'
            dropout = 0.15
            label_smoothing_factor = 0.05
            
            logger.info(f"\n🟢 MODERATE DATA STRATEGY: {samples_per_class:.1f} samples/class")
            logger.info(f"    → Standard training with careful monitoring")
            
        else:
            # GOOD: Sufficient data
            strategy = "GOOD"
            num_epochs_range = [5, 8]
            batch_size = 16
            learning_rate = 1e-4
            gradient_accumulation_steps = 1
            warmup_ratio = 0.06
            early_stopping_patience = 3
            weight_decay = 0.01
            max_grad_norm = 1.0
            lr_bounds = (1e-5, 1e-4)       # Conservative range — more data means lower LR works well
            wd_bounds = (0.0, 0.02)
            lr_scheduler_type = 'linear'
            dropout = 0.1
            label_smoothing_factor = 0.0
            
            logger.info(f"\n✅ GOOD DATA STRATEGY: {samples_per_class:.1f} samples/class")
            logger.info(f"    → Standard fine-tuning approach")
        
        # ADJUST for severe imbalance even if samples_per_class is decent
        if imbalance_ratio > 10:
            logger.info(f"\n⚠️  SEVERE IMBALANCE DETECTED: {imbalance_ratio:.1f}x")
            logger.info(f"    → Adjusting early stopping patience")
            early_stopping_patience = max(early_stopping_patience + 2, 5)
            weight_decay = weight_decay * 1.2

        # Focal gamma scaled by imbalance (only active when use_focal_loss=True)
        if imbalance_ratio > 20:
            focal_gamma = 4.0
        elif imbalance_ratio > 10:
            focal_gamma = 3.0
        else:
            focal_gamma = 2.0

        config = {
            'strategy': strategy,
            'num_epochs_range': num_epochs_range,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'lr_bounds': lr_bounds,
            'weight_decay': weight_decay,
            'weight_decay_bounds': wd_bounds,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'warmup_ratio': warmup_ratio,
            'max_grad_norm': max_grad_norm,
            'early_stopping_patience': early_stopping_patience,
            'lr_scheduler_type': lr_scheduler_type,
            'dropout': dropout,
            'focal_gamma': focal_gamma,
            'label_smoothing_factor': label_smoothing_factor,
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
        logger.info(f"Scheduler: {lr_scheduler_type} | Warmup ratio: {warmup_ratio}")
        logger.info(f"Dropout: {dropout} | Label smoothing: {label_smoothing_factor}")
        logger.info(f"Focal gamma: {focal_gamma} | Weight Decay: {weight_decay} | Max Grad Norm: {max_grad_norm}")
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
        measurement = text_info.get('measurement', 'char_div4_approx')
        print(f"\n📝 Text Analysis:")
        if measurement == 'real_tokens':
            print(f"   Measurement: real token counts ({text_info.get('tokenizer_used')})")
            print(f"   Avg length:  {text_info['avg_length']:.0f} tokens  "
                  f"({text_info.get('avg_char_length', 0):.0f} chars)")
        else:
            print(f"   Measurement: char÷4 approximation (tokenizer unavailable)")
            print(f"   Avg length:  {text_info['avg_length']:.0f} chars")
        print(f"   p95 → max_length: {text_info['p95_length']} tokens"
              + (f"  (raw p95: {text_info.get('p95_tokens_raw')})" if text_info.get('p95_tokens_raw', 0) > 512 else ""))
        print(f"   p99:              {text_info['p99_length']} tokens")
        
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
