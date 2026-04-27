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


def compute_val_split(n_samples: int) -> float:
    """
    Single authoritative val-split ratio — used by both ModelTrainer.prepare_data()
    and DataIntelligence._recommend_run_options() so estimates are always consistent
    with what the trainer will actually produce.

    Mirrors the adaptive logic in ModelTrainer:
      < 200 samples   → 15%  (preserve more training data)
      > 25 000 samples → cap val at ~5 000 rows (diminishing returns beyond that)
      otherwise        → 20%
    """
    if n_samples < 200:
        return 0.15
    elif n_samples > 25000:
        return max(0.05, 5000.0 / n_samples)
    else:
        return 0.20


class DataIntelligence:
    """Analyzes dataset and provides intelligence for AutoLLM decisions"""
    
    def __init__(self):
        pass
    
    def analyze(
        self,
        df: pd.DataFrame,
        label_column: str,
        text_column: str,
        is_multi_label: bool = False,
    ) -> Dict:
        """
        Comprehensive dataset analysis

        Args:
            df: DataFrame
            label_column: Label column name
            text_column: Text column name
            is_multi_label: Whether labels are multi-label (comma/pipe separated)

        Returns:
            Dictionary with analysis results
        """
        analysis = {}

        # 1. Task Detection
        analysis['task_info'] = self._detect_task(df, label_column, is_multi_label=is_multi_label)

        # 2. Class Imbalance Detection
        analysis['imbalance_info'] = self._detect_imbalance(
            df, label_column, is_multi_label=is_multi_label
        )

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

        # 6. Auto-recommend CV and Optuna based on data characteristics
        # For multi-label: use total label occurrences / num_unique_labels so that
        # a 200-row dataset with 10 labels and avg 3 labels/row gives ~60 spc (not 20).
        # For single-label: sum(class_dist) == len(df), so the formula is identical.
        _total_label_occ = sum(analysis['task_info']['class_distribution'].values())
        _spc = _total_label_occ / max(analysis['task_info']['num_classes'], 1)
        analysis['run_recommendations'] = self._recommend_run_options(
            dataset_size=len(df),
            samples_per_class=_spc,
            imbalance_ratio=analysis['imbalance_info']['imbalance_ratio'],
            strategy=analysis['training_config']['strategy'],
        )

        return analysis
    
    def _detect_task(
        self,
        df: pd.DataFrame,
        label_column: str,
        is_multi_label: bool = False,
    ) -> Dict:
        """
        Detect task type: binary, multiclass, or multi_label
        """
        # ── Multi-label branch ────────────────────────────────────────────────
        if is_multi_label:
            from automl.data_validator import DataValidator
            parsed = DataValidator._parse_multi_labels(df[label_column])
            all_atomic = sorted({lbl for row in parsed for lbl in row})
            num_classes = len(all_atomic)
            # Compute per-label frequency for class-distribution dict
            from collections import Counter
            label_counts = Counter(lbl for row in parsed for lbl in row)
            class_dist = dict(label_counts.most_common())
            info = {
                'task_type': 'multi_label',
                'label_type': 'categorical',
                'num_classes': num_classes,
                'class_distribution': class_dist,
                'class_weights': {lbl: 1.0 for lbl in all_atomic},  # placeholder; computed in train_model
            }
            logger.info(f"Task detected: multi_label ({num_classes} unique labels)")
            logger.info(f"Label distribution: {class_dist}")
            return info

        # ── Single-label branch (existing logic) ─────────────────────────────
        num_classes = df[label_column].nunique()

        # Guard: fewer than 2 classes → cannot train a classifier
        if num_classes < 2:
            raise ValueError(
                f"Label column '{label_column}' has only {num_classes} unique value(s). "
                f"Classification requires at least 2 distinct classes."
            )

        # Guard: stratified split requires at least 2 samples per class
        class_counts = df[label_column].value_counts()
        min_class_size = int(class_counts.min())
        if min_class_size < 2:
            tiny_class = class_counts.idxmin()
            raise ValueError(
                f"Class '{tiny_class}' has only {min_class_size} sample(s). "
                f"Stratified train/val splitting requires at least 2 samples per class. "
                f"Remove or merge classes with fewer than 2 samples before running."
            )

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
    
    def _detect_imbalance(
        self,
        df: pd.DataFrame,
        label_column: str,
        is_multi_label: bool = False,
    ) -> Dict:
        """
        Detect class imbalance and recommend handling strategies.
        For multi-label datasets, imbalance is measured across per-label frequencies.
        """
        if is_multi_label:
            from automl.data_validator import DataValidator
            from collections import Counter
            parsed = DataValidator._parse_multi_labels(df[label_column])
            label_counts = Counter(lbl for row in parsed for lbl in row)
            if len(label_counts) > 0:
                max_count = max(label_counts.values())
                min_count = min(label_counts.values())
                imbalance_ratio = max_count / max(min_count, 1)
                minority_ratio  = min_count / len(df)
            else:
                imbalance_ratio, minority_ratio = 1.0, 1.0
            info = {
                'imbalance_ratio': imbalance_ratio,
                # For multi-label we use pos_weight in BCEWithLogitsLoss instead
                # of per-class CrossEntropy weights.
                'use_class_weights': imbalance_ratio > 2.0,
                'use_focal_loss': False,  # focal loss targets single-label tasks
                'minority_class_ratio': minority_ratio,
            }
            logger.info(f"Multi-label imbalance ratio (max/min label freq): {imbalance_ratio:.2f}")
            if info['use_class_weights']:
                logger.info("✓ Will use per-label pos_weight for multi-label imbalance")
            return info

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

    def _recommend_run_options(
        self,
        dataset_size: int,
        samples_per_class: float,
        imbalance_ratio: float,
        strategy: str,
    ) -> Dict:
        """
        Recommend whether to run Cross-Validation and/or Optuna based purely
        on data characteristics — no user input required.

        Returns a dict:
            use_cv       : bool — whether CV is recommended
            cv_folds     : int  — how many folds
            cv_reason    : str  — human-readable reason shown in UI
            use_optuna   : bool — whether Optuna is recommended
            optuna_trials: int  — how many trials
            optuna_reason: str  — human-readable reason shown in UI
        """
        # ── Cross-Validation recommendation ──────────────────────────────────
        # Use compute_val_split() — the same function the trainer uses — so the
        # estimated val size is always exact, not a hardcoded 0.15 guess.
        val_ratio        = compute_val_split(dataset_size)
        val_size_actual  = int(dataset_size * val_ratio)
        if val_size_actual < 60:
            use_cv    = True
            cv_folds  = 5
            cv_reason = (
                f"Dataset has only {dataset_size} samples — "
                f"a single {val_ratio:.0%} val split ({val_size_actual} rows) is too small "
                f"to trust metrics. 5-fold CV gives a reliable mean ± std."
            )
        elif val_size_actual < 120 or imbalance_ratio > 5:
            use_cv    = True
            cv_folds  = 5
            cv_reason = (
                f"Val split ({val_size_actual} rows) is small"
                + (f" and class imbalance is {imbalance_ratio:.1f}×" if imbalance_ratio > 5 else "")
                + ". 5-fold CV provides more stable metrics."
            )
        elif dataset_size > 5000:
            use_cv    = False
            cv_folds  = 5
            cv_reason = (
                f"Dataset has {dataset_size} samples — "
                f"val split ({val_size_actual} rows) is large enough to be reliable. "
                f"CV skipped to save time."
            )
        else:
            use_cv    = False
            cv_folds  = 5
            cv_reason = (
                f"Val split ({val_size_actual} rows) is adequate. "
                f"CV optional — enable if you want confidence intervals."
            )

        # ── Optuna recommendation ─────────────────────────────────────────────
        # Dict-based config: every tier is explicit, unknown tiers get a warning
        # and fall back to MODERATE — never silently to the most aggressive option.
        # Min trials is 3 (not 0) so that if a user force-enables Optuna on a
        # CRITICAL dataset the tuner still runs safely.
        _OPTUNA_CFG = {
            #  strategy   : (use_optuna, trials, reason)
            "CRITICAL": (
                False, 3,
                f"Only {samples_per_class:.0f} samples/class (CRITICAL tier). "
                f"Optuna proxy trials need ~30% of data each — too noisy for reliable "
                f"signal. Rule-based config is more stable here."
            ),
            "SMALL": (
                False, 5,
                f"{samples_per_class:.0f} samples/class (SMALL tier). "
                f"Optuna has marginal benefit — rule-based lr is already well-chosen "
                f"for this range. Enable if you want to experiment."
            ),
            "MODERATE": (
                True, 8,
                f"{samples_per_class:.0f} samples/class (MODERATE tier). "
                f"Optuna recommended — enough data for reliable proxy trials, "
                f"lr/wd tuning typically improves F1 by 1–3%."
            ),
            "GOOD": (
                True, 10,
                f"{samples_per_class:.0f} samples/class (GOOD tier). "
                f"Optuna strongly recommended — stable gradients make search reliable, "
                f"tuned lr/wd consistently outperforms fixed values."
            ),
        }
        if strategy not in _OPTUNA_CFG:
            logger.warning(
                "Unknown strategy %r — falling back to MODERATE Optuna config. "
                "Add this tier to _OPTUNA_CFG if it is intentional.",
                strategy,
            )
        use_optuna, optuna_trials, optuna_reason = _OPTUNA_CFG.get(
            strategy, _OPTUNA_CFG["MODERATE"]
        )

        rec = {
            "use_cv":        use_cv,
            "cv_folds":      cv_folds,
            "cv_reason":     cv_reason,
            "use_optuna":    use_optuna,
            "optuna_trials": optuna_trials,
            "optuna_reason": optuna_reason,
        }
        logger.info(
            "AUTO-RECOMMEND — CV: %s (%d-fold)  |  Optuna: %s (%d trials)",
            use_cv, cv_folds, use_optuna, optuna_trials,
        )
        logger.info("  CV reason    : %s", cv_reason)
        logger.info("  Optuna reason: %s", optuna_reason)
        return rec

    @staticmethod
    def _detect_hardware_batch() -> int:
        """
        Detect the maximum safe batch size for the current hardware.

        GPU tiers (by total VRAM):
          >=16 GB -> 32  (A100, RTX 4090, large server GPUs)
          8-16 GB -> 16  (RTX 3080/4070, T4)
          4-8 GB  ->  8  (RTX 3060, consumer GPUs)
          2-4 GB  ->  4  (low-end / laptop GPU)
          <2 GB   ->  2  (very limited VRAM)

        CPU fallback uses core count.
        """
        import torch
        import os
        if torch.cuda.is_available():
            try:
                total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
                if total_vram_gb >= 16:
                    hw_batch = 32
                elif total_vram_gb >= 8:
                    hw_batch = 16
                elif total_vram_gb >= 4:
                    hw_batch = 8
                elif total_vram_gb >= 2:
                    hw_batch = 4
                else:
                    hw_batch = 2
                logger.info(
                    f"GPU: {total_vram_gb:.1f} GB VRAM -> hardware max batch = {hw_batch}"
                )
                return hw_batch
            except Exception:
                logger.warning("Could not read GPU properties -- defaulting to batch=8")
                return 8
        else:
            cpu_cores = os.cpu_count() or 4
            hw_batch = min(16, max(4, cpu_cores // 2))
            logger.info(f"CPU: {cpu_cores} cores -> hardware max batch = {hw_batch}")
            return hw_batch

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
        
        # ── Hardware-aware batch size ─────────────────────────────────────────
        # Detect the maximum safe batch size for the current hardware FIRST.
        # Each tier defines an ideal batch; final batch = min(hw_max, tier_ideal)
        # Gradient accumulation compensates when hw_max < tier_ideal so the
        # effective batch (batch x accum) always equals the tier ideal.
        _hw_max_batch = self._detect_hardware_batch()

        if samples_per_class < 50:
            strategy = "CRITICAL"
            num_epochs_range = [15, 20]
            _tier_batch = 4          # tiny data: small batch keeps gradient noise (helps generalisation)
            learning_rate = 5e-4
            warmup_ratio = 0.06
            early_stopping_patience = 7
            weight_decay = 0.0       # off by default; set below only if imbalance detected
            max_grad_norm = 0.5      # tighter clipping for unstable small-data gradients
            lr_bounds = (2e-4, 5e-4)
            wd_bounds = (0.01, 0.04)
            lr_scheduler_type = 'cosine'
            dropout = 0.2            # classifier-head dropout -- essential on tiny data
            label_smoothing_factor = 0.0  # off until data-augmentation stage

            logger.info(f"\n WARNING CRITICAL STRATEGY: Only {samples_per_class:.1f} samples/class")
            logger.info(f"    -> Extreme overfitting risk")
            logger.info(f"    -> Classifier dropout ON (0.2)")
            logger.info(f"    -> Action: Consider collecting more data or data augmentation")

        elif samples_per_class < 200:
            strategy = "SMALL"
            num_epochs_range = [10, 15]
            _tier_batch = 8
            learning_rate = 2e-4
            warmup_ratio = 0.06
            early_stopping_patience = 6
            weight_decay = 0.0
            max_grad_norm = 0.75
            lr_bounds = (1e-4, 5e-4)
            wd_bounds = (0.005, 0.03)
            lr_scheduler_type = 'cosine'
            dropout = 0.15           # moderate classifier dropout
            label_smoothing_factor = 0.0

            logger.info(f"\n SMALL DATA STRATEGY: {samples_per_class:.1f} samples/class")
            logger.info(f"    -> Classifier dropout ON (0.15)")

        elif samples_per_class < 500:
            strategy = "MODERATE"
            num_epochs_range = [8, 12]
            _tier_batch = 16
            learning_rate = 1e-4
            warmup_ratio = 0.06
            early_stopping_patience = 4
            weight_decay = 0.0
            max_grad_norm = 1.0
            lr_bounds = (5e-5, 2e-4)
            wd_bounds = (0.0, 0.02)
            lr_scheduler_type = 'linear'
            dropout = 0.1            # light classifier dropout (approaching the 500 boundary)
            label_smoothing_factor = 0.0

            logger.info(f"\n MODERATE DATA STRATEGY: {samples_per_class:.1f} samples/class")
            logger.info(f"    -> Classifier dropout ON (0.1)")

        else:
            strategy = "GOOD"
            num_epochs_range = [5, 8]
            _tier_batch = 32
            learning_rate = 1e-4
            warmup_ratio = 0.06
            early_stopping_patience = 3
            weight_decay = 0.0
            max_grad_norm = 1.0
            lr_bounds = (1e-5, 1e-4)
            wd_bounds = (0.0, 0.02)
            lr_scheduler_type = 'linear'
            dropout = 0.0            # >=500 samples/class: no classifier dropout needed
            label_smoothing_factor = 0.0

            logger.info(f"\n GOOD DATA STRATEGY: {samples_per_class:.1f} samples/class")
            logger.info(f"    -> Classifier dropout OFF (>=500 samples/class)")

        # ── Final batch: hardware-capped, gradient-accum compensated ─────────
        batch_size = min(_hw_max_batch, _tier_batch)
        gradient_accumulation_steps = max(1, _tier_batch // batch_size)
        if gradient_accumulation_steps > 1:
            logger.info(
                f"    -> Batch {batch_size} (hw limit) x "
                f"{gradient_accumulation_steps} grad_accum = {_tier_batch} effective"
            )
        else:
            logger.info(f"    -> Batch size: {batch_size}")

        # ── Weight decay -- ONLY when class imbalance warrants it ─────────────
        # Default is 0.0 for all tiers. Imbalance makes the loss surface noisy
        # (minority-class gradients dominate) so L2 regularisation helps stabilise.
        # Label smoothing stays 0.0 -- reserved for the data-augmentation stage.
        if imbalance_ratio > 20:
            weight_decay = 0.03
            early_stopping_patience = max(early_stopping_patience + 2, 5)
            logger.info(
                f"\n SEVERE IMBALANCE ({imbalance_ratio:.1f}x) "
                f"-> weight_decay=0.03, patience+2"
            )
        elif imbalance_ratio > 10:
            weight_decay = 0.02
            early_stopping_patience = max(early_stopping_patience + 2, 5)
            logger.info(
                f"\n HIGH IMBALANCE ({imbalance_ratio:.1f}x) "
                f"-> weight_decay=0.02, patience+2"
            )
        elif imbalance_ratio > 5:
            weight_decay = 0.01
            logger.info(
                f"\n MODERATE IMBALANCE ({imbalance_ratio:.1f}x) "
                f"-> weight_decay=0.01"
            )
        # else: weight_decay stays 0.0

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
        logger.info(f"TRAINING CONFIG: {strategy} Strategy")
        logger.info(f"{'='*70}")
        logger.info(f"Epochs: {num_epochs_range} | Batch: {batch_size} (eff: {batch_size * gradient_accumulation_steps}) | LR: {learning_rate}")
        logger.info(f"Scheduler: {lr_scheduler_type} | Warmup ratio: {warmup_ratio}")
        logger.info(f"Classifier Dropout: {dropout} | Weight Decay: {weight_decay} (imbalance-driven)")
        logger.info(f"Focal gamma: {focal_gamma} | Max Grad Norm: {max_grad_norm}")
        logger.info(f"Early Stopping Patience: {early_stopping_patience} (monitors val loss)")
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
