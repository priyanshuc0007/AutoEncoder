"""
Model Trainer Module
Handles model fine-tuning with hyperparameter tuning using Optuna
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)

# Models that don't declare model_type in their config.json and need an explicit class
MODEL_CLASS_OVERRIDES = {
    "prajjwal1/bert-tiny":   BertForSequenceClassification,
    "prajjwal1/bert-mini":   BertForSequenceClassification,
    "prajjwal1/bert-small":  BertForSequenceClassification,
    "prajjwal1/bert-medium": BertForSequenceClassification,
    # google/mobilebert-uncased and distilbert/bert use AutoModelForSequenceClassification fine
}

# Models that share another model's tokenizer vocab
_GLOBAL_TOKENIZER_OVERRIDES = {
    "prajjwal1/bert-tiny":   "bert-base-uncased",
    "prajjwal1/bert-mini":   "bert-base-uncased",
    "prajjwal1/bert-small":  "bert-base-uncased",
    "prajjwal1/bert-medium": "bert-base-uncased",
    # google/mobilebert-uncased has its own tokenizer
}
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import logging
from pathlib import Path
import time
from typing import Dict, Tuple, List

from automl.dataset import TextDataset          # shared dataset
from automl.data_intelligence import compute_val_split  # single source of truth for val split ratio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class _EpochLogCallback(TrainerCallback):
    """
    Logs per-epoch metrics (train loss, val loss, val F1) to our pipeline logger
    so every epoch is visible in pipeline.log instead of just the final result.
    Also reports when early stopping triggers and which epoch was the best.
    """

    def __init__(self, model_name: str, total_epochs: int):
        self.model_name = model_name
        self.total_epochs = total_epochs
        self._best_val_loss = float("inf")
        self._best_epoch = 0
        self._last_train_loss: float | None = None

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if not logs:
            return
        # Capture rolling train loss (logged mid-epoch by the Trainer)
        if "loss" in logs and "eval_loss" not in logs:
            self._last_train_loss = logs["loss"]
            return
        # When eval metrics arrive (end of each epoch):
        if "eval_loss" not in logs:
            return
        epoch = round(state.epoch or 0)
        val_loss = logs["eval_loss"]
        best_marker = ""
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._best_epoch = epoch
            best_marker = " ← best"

        parts = [f"Epoch {epoch:2d}/{self.total_epochs}"]
        if self._last_train_loss is not None:
            parts.append(f"train_loss={self._last_train_loss:.4f}")
        parts.append(f"val_loss={val_loss:.4f}{best_marker}")
        if "eval_f1" in logs:
            parts.append(f"val_f1={logs['eval_f1']:.4f}")
        if "eval_accuracy" in logs:
            parts.append(f"val_acc={logs['eval_accuracy']:.4f}")
        logger.info("  %s", " | ".join(parts))

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        completed = round(state.epoch or 0)
        if completed < self.total_epochs:
            logger.info(
                "  Early stopping triggered at epoch %d/%d "
                "(val_loss did not improve for patience epochs)",
                completed, self.total_epochs,
            )
        logger.info(
            "  Best checkpoint: epoch %d  val_loss=%.4f",
            self._best_epoch, self._best_val_loss,
        )


class WeightedTrainer(Trainer):
    """Trainer subclass that applies per-class weights and optional focal / multi-label loss."""

    def __init__(
        self,
        class_weights_tensor=None,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
        label_smoothing_factor: float = 0.0,
        is_multi_label: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.class_weights_tensor = class_weights_tensor
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.label_smoothing_factor = label_smoothing_factor
        self.is_multi_label = is_multi_label

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Use .get() + filtered copy instead of .pop() to avoid mutating the
        # shared batch dict — Trainer callbacks may inspect it after this call.
        labels = inputs.get("labels")
        inputs_no_labels = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**inputs_no_labels)
        logits = outputs.logits

        # ── Multi-label: BCEWithLogitsLoss ────────────────────────────────
        if self.is_multi_label:
            # labels shape: (batch, num_classes) float  — multi-hot vectors
            loss = nn.BCEWithLogitsLoss(pos_weight=self.class_weights_tensor)(
                logits, labels.float()
            )
            return (loss, outputs) if return_outputs else loss

        # ── Single-label: CrossEntropyLoss / Focal (existing logic) ────────
        if self.use_focal_loss:
            # Focal loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
            # alpha_t is handled via class_weights_tensor (if provided)
            ce_loss = nn.CrossEntropyLoss(
                weight=self.class_weights_tensor, reduction='none',
                label_smoothing=self.label_smoothing_factor,
            )(logits, labels)
            probs = torch.softmax(logits, dim=1)
            # p_t: probability assigned to the correct class
            p_t = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            focal_weight = (1.0 - p_t) ** self.focal_gamma
            loss = (focal_weight * ce_loss).mean()
        else:
            loss_fn = nn.CrossEntropyLoss(
                weight=self.class_weights_tensor,
                label_smoothing=self.label_smoothing_factor,
            )
            loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


class ModelTrainer:
    """Train transformer models for text classification"""
    
    def __init__(self, output_dir: str = "models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def _compute_metrics(self, eval_pred):
        """Compute evaluation metrics for single-label training."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        return {
            'accuracy': accuracy_score(labels, predictions),
            'f1': f1_score(labels, predictions, average='weighted', zero_division=0),
            'precision': precision_score(labels, predictions, average='weighted', zero_division=0),
            'recall': recall_score(labels, predictions, average='weighted', zero_division=0),
        }

    def _compute_metrics_multilabel(self, eval_pred):
        """Compute evaluation metrics for multi-label training."""
        logits, labels = eval_pred
        # Sigmoid + 0.5 threshold to get binary predictions
        probs = 1.0 / (1.0 + np.exp(-logits))  # numerically stable sigmoid
        preds = (probs >= 0.5).astype(int)
        labels = labels.astype(int)

        return {
            'accuracy': float(accuracy_score(labels, preds)),
            'f1': float(f1_score(labels, preds, average='micro', zero_division=0)),
            'precision': float(precision_score(labels, preds, average='micro', zero_division=0)),
            'recall': float(recall_score(labels, preds, average='micro', zero_division=0)),
        }
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        label_column: str,
        text_column: str,
        test_size: float = 0.2,
        is_multi_label: bool = False,
    ) -> Tuple[Dict, object]:
        """
        Prepare and split data.

        For multi-label datasets, uses MultiLabelBinarizer to produce multi-hot
        float matrices; for single-label uses a LabelEncoder as before.

        Returns:
            Tuple of (data_dict, label_encoder_or_binarizer)
        """
        texts = df[text_column].tolist()
        raw_labels = df[label_column].tolist()
        n_samples = len(texts)

        # Guard: replace empty/NaN texts with a placeholder so the tokenizer
        # never receives an empty string (causes silent padding-only sequences).
        texts = [
            t if (isinstance(t, str) and t.strip()) else "[EMPTY]"
            for t in texts
        ]

        # Adaptive test_size — delegate to the shared compute_val_split() so that
        # ModelTrainer and DataIntelligence always use identical thresholds.
        if test_size == 0.2:          # only override the default, not an explicit caller value
            test_size = compute_val_split(n_samples)
        logger.info(
            f"Train/val split: {(1 - test_size) * 100:.0f}% / {test_size * 100:.0f}%"
            f" (n={n_samples})"
        )

        if is_multi_label:
            from automl.data_validator import DataValidator
            parsed_labels = DataValidator._parse_multi_labels(df[label_column])

            # Random split (no stratify for multi-hot labels)
            import random
            rng = random.Random(42)
            indices = list(range(len(texts)))
            rng.shuffle(indices)
            split_at = int(len(indices) * (1 - test_size))
            train_idx, val_idx = indices[:split_at], indices[split_at:]

            train_texts  = [texts[i] for i in train_idx]
            val_texts    = [texts[i] for i in val_idx]
            train_parsed = [parsed_labels[i] for i in train_idx]
            val_parsed   = [parsed_labels[i] for i in val_idx]

            # Fit MLB on ALL parsed labels (not just train) so that rare labels
            # that only appear in the val split are still encoded correctly.
            # Val ground truth would be silently zeroed otherwise.
            mlb = MultiLabelBinarizer()
            mlb.fit(parsed_labels)
            train_labels = mlb.transform(train_parsed).astype('float32')
            val_labels   = mlb.transform(val_parsed).astype('float32')

            logger.info(f"Multi-label binarizer classes: {list(mlb.classes_)}")
            logger.info(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Num labels: {len(mlb.classes_)}")

            data = {
                'train_texts':  train_texts,
                'train_labels': train_labels,  # shape (N, C) float32
                'val_texts':    val_texts,
                'val_labels':   val_labels,
                'num_classes':  len(mlb.classes_),
                'label_encoder': mlb,          # MultiLabelBinarizer in multi-label mode
            }
            return data, mlb

        # ── Single-label path (unchanged) ───────────────────────────────────
        # Split data first — fit LabelEncoder only on train labels to prevent leakage
        try:
            train_texts, val_texts, train_labels_raw, val_labels_raw = train_test_split(
                texts, raw_labels, test_size=test_size, random_state=42, stratify=raw_labels
            )
        except ValueError:
            # Stratified split requires ≥ 2 samples per class; fall back to random shuffle.
            logger.warning(
                "⚠️ Stratified split failed (a class has < 2 samples) — "
                "using random split. Consider collecting more data."
            )
            train_texts, val_texts, train_labels_raw, val_labels_raw = train_test_split(
                texts, raw_labels, test_size=test_size, random_state=42
            )

        # Guard: if any class appears only in val (not in train), LabelEncoder.transform
        # will raise 'unseen labels'. Move one example per unseen class from val → train.
        _val_unseen = set(val_labels_raw) - set(train_labels_raw)
        if _val_unseen:
            logger.warning(
                f"⚠️ {len(_val_unseen)} class(es) have no training sample after split — "
                f"redistributing one example per class from val → train."
            )
            _t_texts, _t_lbls = list(train_texts), list(train_labels_raw)
            _v_texts, _v_lbls = list(val_texts), list(val_labels_raw)
            for _cls in _val_unseen:
                for _i in reversed(range(len(_v_lbls))):
                    if _v_lbls[_i] == _cls:
                        _t_texts.append(_v_texts.pop(_i))
                        _t_lbls.append(_v_lbls.pop(_i))
                        break
            train_texts, val_texts = _t_texts, _v_texts
            train_labels_raw, val_labels_raw = _t_lbls, _v_lbls

        if len(val_texts) == 0:
            raise ValueError(
                "Validation set is empty after split. "
                "Use more data or switch to cross-validation."
            )
        if len(val_texts) < 5:
            logger.warning(
                f"⚠️ Only {len(val_texts)} validation samples — metrics will be unreliable. "
                f"Consider using cross-validation (use_cv=True)."
            )

        le = LabelEncoder()
        train_labels = le.fit_transform(train_labels_raw)
        val_labels   = le.transform(val_labels_raw)

        logger.info(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

        data = {
            'train_texts':  train_texts,
            'train_labels': train_labels,
            'val_texts':    val_texts,
            'val_labels':   val_labels,
            'num_classes':  len(le.classes_),
            'label_encoder': le,
        }

        logger.info(f"Train: {len(train_texts)}, Val: {len(val_texts)}")

        return data, le
    
    def train_model(
        self,
        model_name: str,
        data: Dict,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        max_length: int,
        use_class_weights: bool = False,
        class_weights: Dict = None,
        use_focal_loss: bool = False,
        experiment_name: str = "default",
        gradient_accumulation_steps: int = 1,
        warmup_ratio: float = 0.06,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        early_stopping_patience: int = 3,
        lr_scheduler_type: str = 'linear',
        dropout: float = 0.1,
        focal_gamma: float = 2.0,
        label_smoothing_factor: float = 0.0,
        is_multi_label: bool = False,
    ) -> Dict:
        """
        Train a single model with adaptive hyperparameters

        Args:
            model_name: HuggingFace model name
            data: Data dictionary from prepare_data()
            num_epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            max_length: Max sequence length
            use_class_weights: Whether to use class weights
            class_weights: Class weights dictionary
            experiment_name: Name for this training run
            gradient_accumulation_steps: Gradient accumulation steps
            warmup_ratio: Fraction of training steps used for warmup
            weight_decay: Weight decay
            max_grad_norm: Max gradient norm for clipping
            early_stopping_patience: Early stopping patience
            lr_scheduler_type: 'cosine' or 'linear'
            dropout: Classifier dropout probability
            focal_gamma: Focal loss gamma (used when use_focal_loss=True)
            label_smoothing_factor: Label smoothing (0.0 = disabled)

        Returns:
            Dictionary with training results
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Training {model_name}")
        logger.info(f"{'='*70}")
        logger.info(f"Epochs: {num_epochs}, Batch: {batch_size}, LR: {learning_rate}")
        logger.info(f"Grad Accumulation: {gradient_accumulation_steps}, Warmup: {warmup_ratio}")
        if is_multi_label:
            logger.info(f"\U0001f3f7\ufe0f  Multi-label mode: BCEWithLogitsLoss + sigmoid threshold")
        if use_class_weights and class_weights:
            logger.info(f"✓ Using class weights: {class_weights}")
        logger.info(f"{'='*70}\n")
        
        start_time = time.time()
        
        # Load tokenizer and model
        logger.info("Loading tokenizer and model...")
        # prajjwal1/* models have no tokenizer files — they share bert-base-uncased vocab
        tokenizer_name = _GLOBAL_TOKENIZER_OVERRIDES.get(model_name, model_name)
        if tokenizer_name != model_name:
            logger.info(f"Using tokenizer '{tokenizer_name}' for model '{model_name}'")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        model_cls = MODEL_CLASS_OVERRIDES.get(model_name, AutoModelForSequenceClassification)
        if model_cls is not AutoModelForSequenceClassification:
            logger.info(f"Using explicit class '{model_cls.__name__}' for model '{model_name}'")
        # Load config and apply dropout BEFORE constructing the model so that
        # the nn.Dropout(p=...) layers are initialised with the right values.
        # Modifying model.config after from_pretrained() has no effect on existing
        # Dropout instances.
        # Models in MODEL_CLASS_OVERRIDES (bert-tiny, bert-mini, etc.) don't declare
        # model_type in their config.json, so AutoConfig.from_pretrained() raises
        # "Unrecognized model". Use BertConfig directly for those models.
        from transformers import AutoConfig, BertConfig
        if model_cls is BertForSequenceClassification:
            model_config = BertConfig.from_pretrained(model_name, num_labels=data['num_classes'])
        else:
            model_config = AutoConfig.from_pretrained(model_name, num_labels=data['num_classes'])
        # Set ONLY the classifier head dropout — never touch hidden_dropout_prob.
        # Internal transformer layer dropout (0.1 in BERT) stays at pre-trained value.
        if hasattr(model_config, 'classifier_dropout'):
            model_config.classifier_dropout = dropout
        model = model_cls.from_pretrained(
            model_name,
            config=model_config,
        ).to(self.device)
        if dropout > 0.0:
            logger.info(f"Classifier dropout set to {dropout}")
        else:
            logger.info("Classifier dropout disabled (GOOD tier / sufficient data)")

        # Create datasets
        train_dataset = TextDataset(
            data['train_texts'],
            data['train_labels'],
            tokenizer,
            max_length
        )

        val_dataset = TextDataset(
            data['val_texts'],
            data['val_labels'],
            tokenizer,
            max_length
        )

        # Training arguments with adaptive hyperparameters
        model_output_dir = self.output_dir / f"{experiment_name}_{model_name.split('/')[-1]}"

        # Pre-flight GPU memory check: if available VRAM is below 2 GB, halve
        # batch_size and compensate with gradient_accumulation_steps so the
        # effective batch stays the same.  Prevents most CUDA OOM crashes before
        # they happen.
        if torch.cuda.is_available() and batch_size > 4:
            try:
                _free_gb = (
                    torch.cuda.get_device_properties(0).total_memory
                    - torch.cuda.memory_allocated()
                ) / 1024 ** 3
                if _free_gb < 2.0:
                    _old_bs = batch_size
                    batch_size = max(4, batch_size // 2)
                    gradient_accumulation_steps = max(
                        gradient_accumulation_steps, _old_bs // batch_size
                    )
                    logger.warning(
                        f"⚠️ Low GPU memory ({_free_gb:.1f} GB free) — "
                        f"batch_size {_old_bs}→{batch_size}, "
                        f"grad_accum→{gradient_accumulation_steps}"
                    )
            except Exception:
                pass

        training_args = TrainingArguments(
            output_dir=str(model_output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            max_grad_norm=max_grad_norm,
            lr_scheduler_type=lr_scheduler_type,
            label_smoothing_factor=label_smoothing_factor,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="loss",   # monitor val loss: catches overfitting earlier than F1
            greater_is_better=False,          # lower loss = better checkpoint
            save_total_limit=1,          # keep only the single best checkpoint → saves disk space
            logging_steps=max(1, len(train_dataset) // (batch_size * 4)),  # Log 4 times per epoch
            seed=42,
            report_to=[],  # Disable wandb/tensorboard logging
            dataloader_pin_memory=torch.cuda.is_available(),  # only pin memory when a GPU is present
        )
        
        # Prepare class weights tensor
        if is_multi_label:
            # For multi-label: compute per-label pos_weight = (neg_count / pos_count)
            # Labels matrix shape (N, C); train_labels stored as float32 numpy array
            train_lbl_mat = data['train_labels']  # (N, C)
            pos_counts = train_lbl_mat.sum(axis=0).clip(min=1)  # shape (C,)
            neg_counts = (len(train_lbl_mat) - train_lbl_mat.sum(axis=0)).clip(min=1)
            # Cap pos_weight at 50 — extremely rare labels (e.g. 1 pos / 10K neg)
            # would produce pos_weight ≈ 10 000 which explodes the BCE loss and
            # destabilises training.  50 is a safe upper bound for any real dataset.
            pos_weight = torch.tensor(
                (neg_counts / pos_counts).clip(max=50.0),
                dtype=torch.float,
            ).to(self.device)
            class_weights_tensor = pos_weight if use_class_weights else None
        elif use_class_weights and class_weights:
            le = data['label_encoder']
            int_weights = [class_weights.get(le.classes_[i], 1.0) for i in range(data['num_classes'])]
            class_weights_tensor = torch.tensor(int_weights).float().to(self.device)
            logger.info(f"Class weights tensor: {class_weights_tensor}")
        else:
            class_weights_tensor = None

        # Select compute_metrics function
        compute_metrics_fn = (
            self._compute_metrics_multilabel if is_multi_label else self._compute_metrics
        )

        # Create trainer — always WeightedTrainer (handles both single and multi-label)
        use_custom_loss = is_multi_label or class_weights_tensor is not None or use_focal_loss
        trainer_cls = WeightedTrainer if use_custom_loss else Trainer
        trainer_kwargs = dict(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics_fn,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=early_stopping_patience),
                _EpochLogCallback(model_name=model_name, total_epochs=num_epochs),
            ],
        )
        if use_custom_loss:
            trainer_kwargs['class_weights_tensor'] = class_weights_tensor
            trainer_kwargs['use_focal_loss'] = use_focal_loss and not is_multi_label
            trainer_kwargs['focal_gamma'] = focal_gamma
            trainer_kwargs['label_smoothing_factor'] = label_smoothing_factor if not is_multi_label else 0.0
            trainer_kwargs['is_multi_label'] = is_multi_label
            if use_focal_loss and not is_multi_label:
                logger.info(f"Using focal loss (gamma={focal_gamma}) to handle class imbalance")
        trainer = trainer_cls(**trainer_kwargs)
        
        # Train
        try:
            logger.info("Starting training...")
            trainer.train()
            logger.info("✓ Training completed successfully")
        except RuntimeError as e:
            # Surface a clear, actionable CUDA OOM message instead of a raw
            # CUDA stack trace that is hard for users to interpret.
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                raise RuntimeError(
                    f"CUDA Out of Memory while training '{model_name}' "
                    f"(batch_size={batch_size}). "
                    f"Try: (1) select a smaller model, "
                    f"(2) disable GPU to use CPU, or "
                    f"(3) reduce max_length by shortening your text column."
                ) from e
            raise
        except KeyboardInterrupt:
            logger.warning("⚠️ Training interrupted by user")
            raise
        except Exception as e:
            logger.error(f"❌ Training failed: {str(e)}", exc_info=True)
            raise
        
        training_time = time.time() - start_time
        
        # Get evaluation results
        eval_results = trainer.evaluate()
        
        results = {
            'model_name': model_name,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'training_time': training_time,
            'eval_loss': eval_results.get('eval_loss', None),
            'eval_f1': eval_results.get('eval_f1', None),
            'eval_accuracy': eval_results.get('eval_accuracy', None),
            'model_path': str(model_output_dir),
            'tokenizer': tokenizer,
            'label_encoder': data['label_encoder'],
        }
        
        logger.info(f"Training completed in {training_time:.2f}s")
        logger.info(f"Eval Loss: {eval_results.get('eval_loss', 'N/A')}")
        
        # Save model
        model.save_pretrained(model_output_dir)
        tokenizer.save_pretrained(model_output_dir)
        
        logger.info(f"Model saved to {model_output_dir}")
        
        return results
    
    def train_multiple_models(
        self,
        model_names: List[str],
        data: Dict,
        hyperparams_ranges: Dict,
        max_length: int,
        use_class_weights: bool = False,
        class_weights: Dict = None,
        use_focal_loss: bool = False,
        experiment_name: str = "default",
        use_optuna: bool = False,
        optuna_trials: int = 10,
        is_multi_label: bool = False,
    ) -> List[Dict]:
        """
        Train multiple models with hyperparameter tuning
        
        Args:
            model_names: List of model names to train
            data: Data dictionary
            hyperparams_ranges: Hyperparameter ranges
            max_length: Max sequence length
            use_class_weights: Use class weights
            class_weights: Class weights
            experiment_name: Experiment name
            use_optuna: Run Optuna search for lr + weight_decay before training
            optuna_trials: Number of Optuna trials per model (clamped 3–20)
            
        Returns:
            List of training results
        """
        all_results = []

        # Run Optuna per model and store each model's config separately.
        # Previously, a single merged_config was overwritten each iteration so all
        # models ended up with the last model's hyperparameters — now fixed.
        optuna_configs = {}
        for model_name in model_names:
            cfg = dict(hyperparams_ranges)  # shallow copy — never mutate original
            if use_optuna:
                try:
                    from automl.tuner import HyperparameterTuner
                    tuner = HyperparameterTuner()
                    best_params = tuner.tune(
                        model_name=model_name,
                        data=data,
                        config=hyperparams_ranges,
                        max_length=max_length,
                        use_class_weights=use_class_weights,
                        class_weights=class_weights,
                        use_focal_loss=use_focal_loss,
                        n_trials=optuna_trials,
                    )
                    # Merge only the params Optuna returned; keep all else rule-based
                    if best_params:
                        cfg.update(best_params)
                        logger.info(
                            f"✅ Using Optuna params for {model_name}: "
                            f"LR={cfg['learning_rate']:.2e}  "
                            f"WD={cfg['weight_decay']:.4f}"
                        )
                except Exception as e:
                    logger.warning(f"Optuna failed for {model_name} ({e}). Using rule-based config.")
            optuna_configs[model_name] = cfg

        for model_name in model_names:
            merged_config = optuna_configs[model_name]
            # Try different hyperparameter combinations
            for num_epochs in merged_config['num_epochs_range']:
                for lr_scale in [1.0]:  # single learning rate per epoch configuration
                    
                    learning_rate = merged_config['learning_rate'] * lr_scale
                    batch_size = merged_config['batch_size']
                    gradient_accumulation_steps = merged_config.get('gradient_accumulation_steps', 1)
                    warmup_ratio = merged_config.get('warmup_ratio', 0.06)
                    weight_decay = merged_config.get('weight_decay', 0.01)
                    max_grad_norm = merged_config.get('max_grad_norm', 1.0)
                    early_stopping_patience = merged_config.get('early_stopping_patience', 3)
                    lr_scheduler_type = merged_config.get('lr_scheduler_type', 'linear')
                    dropout = merged_config.get('dropout', 0.1)
                    focal_gamma = merged_config.get('focal_gamma', 2.0)
                    label_smoothing_factor = merged_config.get('label_smoothing_factor', 0.0)

                    try:
                        result = self.train_model(
                            model_name=model_name,
                            data=data,
                            num_epochs=num_epochs,
                            batch_size=batch_size,
                            learning_rate=learning_rate,
                            max_length=max_length,
                            use_class_weights=use_class_weights,
                            class_weights=class_weights,
                            use_focal_loss=use_focal_loss,
                            experiment_name=experiment_name,
                            gradient_accumulation_steps=gradient_accumulation_steps,
                            warmup_ratio=warmup_ratio,
                            weight_decay=weight_decay,
                            max_grad_norm=max_grad_norm,
                            early_stopping_patience=early_stopping_patience,
                            lr_scheduler_type=lr_scheduler_type,
                            dropout=dropout,
                            focal_gamma=focal_gamma,
                            label_smoothing_factor=label_smoothing_factor,
                            is_multi_label=is_multi_label,
                        )
                        all_results.append(result)

                    except Exception as e:
                        import traceback
                        logger.error(
                            "\n%s\n"
                            "TRAINING FAILED: %s  (epochs=%d)\n"
                            "Reason : %s\n"
                            "Traceback:\n%s"
                            "%s\n",
                            "=" * 70,
                            model_name, num_epochs,
                            str(e),
                            traceback.format_exc(),
                            "=" * 70,
                        )
                        logger.error(
                            "Model '%s' will be SKIPPED — other models continue.", model_name
                        )
                        continue
                    finally:
                        # Free GPU memory after each model to prevent OOM on the next model
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                # Only train the first (minimum) epoch value per model.
                # The upper bound of num_epochs_range is reserved for future
                # multi-epoch grid search; break keeps current single-variant behaviour.
                break
        
        return all_results
