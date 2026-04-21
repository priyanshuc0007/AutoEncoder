"""
Hyperparameter Tuner
Uses Optuna to search learning_rate and weight_decay only.
Everything else (batch_size, epochs, max_length, etc.) stays rule-based.

Design principles:
- Skips automatically on CRITICAL tier (too few samples for reliable signal)
- Runs lightweight proxy trials: small stratified slice, 2 epochs max, no checkpoints
- Tier-aware search bounds: DataIntelligence narrows the range before Optuna searches
- Falls back silently to rule-based config on any failure
"""

import copy
import logging
import tempfile
import shutil
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from automl.dataset import TextDataset

logger = logging.getLogger(__name__)

_TOKENIZER_OVERRIDES = {
    "prajjwal1/bert-tiny":   "bert-base-uncased",
    "prajjwal1/bert-mini":   "bert-base-uncased",
    "prajjwal1/bert-small":  "bert-base-uncased",
    "prajjwal1/bert-medium": "bert-base-uncased",
    # google/mobilebert-uncased has its own tokenizer
}

_MODEL_CLASS_OVERRIDES = {
    "prajjwal1/bert-tiny":   BertForSequenceClassification,
    "prajjwal1/bert-mini":   BertForSequenceClassification,
    "prajjwal1/bert-small":  BertForSequenceClassification,
    "prajjwal1/bert-medium": BertForSequenceClassification,
    # google/mobilebert-uncased auto-dispatches via AutoModelForSequenceClassification
}


class HyperparameterTuner:
    """
    Lightweight Optuna tuner for learning_rate and weight_decay.

    Runs N short proxy trials per model. Each trial trains for 2 epochs on a
    small stratified slice of the training data and evaluates on a capped
    subset of the validation set. No checkpoints are written.
    """

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def tune(
        self,
        model_name: str,
        data: Dict,
        config: Dict,
        max_length: int,
        use_class_weights: bool = False,
        class_weights: Optional[Dict] = None,
        use_focal_loss: bool = False,
        n_trials: int = 10,
    ) -> Dict:
        """
        Search for best learning_rate and weight_decay using Optuna.

        Args:
            model_name:        HuggingFace model identifier
            data:              Data dict from ModelTrainer.prepare_data()
            config:            Rule-based config from DataIntelligence.
                               Must contain: strategy, lr_bounds, weight_decay_bounds, batch_size
            max_length:        Max token sequence length
            use_class_weights: Whether to apply class weights
            class_weights:     Class weight dict {label_str: float}
            use_focal_loss:    Whether to use focal loss in trials
            n_trials:          Number of Optuna trials (clamped to 3–20)

        Returns:
            Dict with 'learning_rate' and 'weight_decay' (best found values),
            or empty dict if tuning was skipped or failed — caller should
            treat empty dict as "keep rule-based values".
        """
        # Guard: Optuna must be installed
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning(
                "Optuna not installed — skipping hyperparameter tuning. "
                "Install with: pip install optuna"
            )
            return {}

        # Guard: skip on CRITICAL tier
        strategy = config.get('strategy', 'GOOD')
        if strategy == 'CRITICAL':
            logger.info(
                "⚡ Optuna skipped: CRITICAL tier has too few samples for reliable tuning signal. "
                "Using rule-based hyperparameters."
            )
            return {}

        n_trials = max(3, min(20, n_trials))
        lr_low, lr_high = config.get('lr_bounds', (1e-5, 1e-4))
        wd_low, wd_high = config.get('weight_decay_bounds', (0.0, 0.05))
        num_epochs_range = config.get('num_epochs_range', [3, 5])
        batch_size = min(config.get('batch_size', 16), 16)  # cap proxy batch at 16

        logger.info(f"\n{'='*70}")
        logger.info(f"🔍 OPTUNA HYPERPARAMETER SEARCH — {model_name}")
        logger.info(f"   Strategy: {strategy}  |  Trials: {n_trials}")
        logger.info(f"   LR range : [{lr_low:.1e}, {lr_high:.1e}]")
        logger.info(f"   WD range : [{wd_low:.4f}, {wd_high:.4f}]")
        logger.info(f"{'='*70}")

        # Build proxy dataset once — reused across all trials
        proxy = self._build_proxy_data(data)
        if proxy is None:
            logger.warning("Could not build proxy dataset. Skipping Optuna.")
            return {}

        # Load tokenizer once — shared across all trials
        tokenizer_name = _TOKENIZER_OVERRIDES.get(model_name, model_name)
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception as e:
            logger.warning(f"Optuna skipped — tokenizer load failed: {e}")
            return {}

        proxy_train_ds = TextDataset(
            proxy['train_texts'], proxy['train_labels'], tokenizer, max_length
        )
        proxy_val_ds = TextDataset(
            proxy['val_texts'], proxy['val_labels'], tokenizer, max_length
        )

        # Class weights tensor — built once, reused
        cw_tensor = None
        if use_class_weights and class_weights:
            le = data['label_encoder']
            int_weights = [
                class_weights.get(le.classes_[i], 1.0)
                for i in range(data['num_classes'])
            ]
            cw_tensor = torch.tensor(int_weights).float().to(self.device)

        # Load the pretrained base model ONCE — each trial receives a deep copy in
        # memory instead of re-reading weights from disk (eliminates n_trials disk reads).
        model_cls = _MODEL_CLASS_OVERRIDES.get(model_name, AutoModelForSequenceClassification)
        try:
            base_model = model_cls.from_pretrained(
                model_name, num_labels=data['num_classes']
            ).to(self.device)
        except Exception as e:
            logger.warning(f"Optuna skipped — base model load failed: {e}")
            return {}

        def objective(trial):
            lr = trial.suggest_float('learning_rate', lr_low, lr_high, log=True)
            wd = trial.suggest_float('weight_decay', wd_low, wd_high)
            num_epochs = trial.suggest_int('num_epochs', num_epochs_range[0], num_epochs_range[1])
            return self._run_trial(
                base_model=base_model,
                train_ds=proxy_train_ds,
                val_ds=proxy_val_ds,
                num_classes=data['num_classes'],
                lr=lr,
                wd=wd,
                num_epochs=num_epochs,
                batch_size=batch_size,
                cw_tensor=cw_tensor,
                use_focal_loss=use_focal_loss,
            )

        try:
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

            best = study.best_params
            logger.info(
                f"\n\u2705 Optuna complete \u2014 "
                f"LR: {best['learning_rate']:.2e}  "
                f"WD: {best['weight_decay']:.4f}  "
                f"Epochs: {best['num_epochs']}  "
                f"(proxy F1: {study.best_value:.4f})"
            )
            logger.info(
                f"   Rule-based was \u2014 "
                f"LR: {config.get('learning_rate', '?'):.2e}  "
                f"WD: {config.get('weight_decay', '?'):.4f}"
            )
            return {
                'learning_rate': best['learning_rate'],
                'weight_decay':  best['weight_decay'],
                'num_epochs_range': [best['num_epochs'], best['num_epochs']],
            }

        except Exception as e:
            logger.warning(f"Optuna search failed ({e}). Falling back to rule-based config.")
            return {}

        finally:
            del base_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_proxy_data(self, data: Dict) -> Optional[Dict]:
        """
        Build a small stratified proxy slice for fast trials.

        Proxy training : at most 300 samples (min 40), stratified.
        Proxy validation: at most 200 samples carved from the training set —
                          the real val set is never used here to avoid leakage.
        """
        try:
            train_texts  = data['train_texts']
            train_labels = np.array(data['train_labels'])

            n_total       = len(train_texts)
            n_proxy_val   = max(20, min(200, int(n_total * 0.1)))
            n_proxy_train = max(40, min(300, int(n_total * 0.3)))

            if n_proxy_train + n_proxy_val >= n_total:
                # Dataset too small — proxy train and val both use all training data
                proxy_train_texts  = list(train_texts)
                proxy_train_labels = train_labels.copy()
                proxy_val_texts    = list(train_texts)
                proxy_val_labels   = train_labels.copy()
            else:
                # Step 1: carve out proxy val from train (stratified, no real val touched)
                sss_pv = StratifiedShuffleSplit(n_splits=1, test_size=n_proxy_val, random_state=0)
                remaining_idx, pv_idx = next(sss_pv.split(train_texts, train_labels))
                proxy_val_texts  = [train_texts[i] for i in pv_idx]
                proxy_val_labels = train_labels[pv_idx]

                # Step 2: carve proxy train from the remaining pool (stratified)
                rem_texts  = [train_texts[i] for i in remaining_idx]
                rem_labels = train_labels[remaining_idx]
                if n_proxy_train >= len(rem_texts):
                    proxy_train_texts  = rem_texts
                    proxy_train_labels = rem_labels
                else:
                    sss_pt = StratifiedShuffleSplit(
                        n_splits=1, test_size=len(rem_texts) - n_proxy_train, random_state=42
                    )
                    keep_idx, _ = next(sss_pt.split(rem_texts, rem_labels))
                    proxy_train_texts  = [rem_texts[i] for i in keep_idx]
                    proxy_train_labels = rem_labels[keep_idx]

            logger.info(
                f"   Proxy data — train: {len(proxy_train_texts)} samples, "
                f"val: {len(proxy_val_texts)} samples (both from training set only)"
            )
            return {
                'train_texts':  proxy_train_texts,
                'train_labels': proxy_train_labels,
                'val_texts':    proxy_val_texts,
                'val_labels':   proxy_val_labels,
            }

        except Exception as e:
            logger.warning(f"Proxy data build failed: {e}")
            return None

    def _run_trial(
        self,
        base_model,
        train_ds,
        val_ds,
        num_classes: int,
        lr: float,
        wd: float,
        num_epochs: int,
        batch_size: int,
        cw_tensor,
        use_focal_loss: bool,
    ) -> float:
        """
        Run one Optuna trial. Returns weighted-F1 on proxy val set.
        Trains for num_epochs epochs, writes no permanent checkpoints.
        Uses copy.deepcopy(base_model) — no disk I/O per trial.
        """
        tmp_dir = tempfile.mkdtemp()
        model = None
        try:
            # Deep-copy the pre-loaded base model — same init weights, no disk read
            model = copy.deepcopy(base_model)

            args = TrainingArguments(
                output_dir=tmp_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=lr,
                weight_decay=wd,
                eval_strategy='epoch',
                save_strategy='no',
                logging_strategy='no',
                report_to=[],
                seed=42,
                dataloader_pin_memory=torch.cuda.is_available(),
            )

            def compute_metrics(eval_pred):
                preds, labels = eval_pred
                preds = np.argmax(preds, axis=1)
                return {
                    'f1': f1_score(labels, preds, average='weighted', zero_division=0)
                }

            use_custom_loss = (cw_tensor is not None) or use_focal_loss
            if use_custom_loss:
                from automl.model_trainer import WeightedTrainer
                trainer = WeightedTrainer(
                    class_weights_tensor=cw_tensor,
                    use_focal_loss=use_focal_loss,
                    model=model,
                    args=args,
                    train_dataset=train_ds,
                    eval_dataset=val_ds,
                    compute_metrics=compute_metrics,
                )
            else:
                trainer = Trainer(
                    model=model,
                    args=args,
                    train_dataset=train_ds,
                    eval_dataset=val_ds,
                    compute_metrics=compute_metrics,
                )

            trainer.train()
            metrics = trainer.evaluate()
            return float(metrics.get('eval_f1', 0.0))

        except Exception as e:
            logger.debug(f"Trial failed: {e}")
            return 0.0

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            if model is not None:
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
