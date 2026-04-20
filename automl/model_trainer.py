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
)

# Models that don't declare model_type in their config.json and need an explicit class
MODEL_CLASS_OVERRIDES = {
    "prajjwal1/bert-tiny":   BertForSequenceClassification,
    "prajjwal1/bert-mini":   BertForSequenceClassification,
    "prajjwal1/bert-small":  BertForSequenceClassification,
    "prajjwal1/bert-medium": BertForSequenceClassification,
}
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import logging
from pathlib import Path
import time
from typing import Dict, Tuple, List

from automl.dataset import TextDataset  # shared dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeightedTrainer(Trainer):
    """Trainer subclass that applies per-class weights and optional focal loss."""

    def __init__(self, class_weights_tensor=None, use_focal_loss: bool = False, focal_gamma: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.class_weights_tensor = class_weights_tensor
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Use .get() + filtered copy instead of .pop() to avoid mutating the
        # shared batch dict — Trainer callbacks may inspect it after this call.
        labels = inputs.get("labels")
        inputs_no_labels = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**inputs_no_labels)
        logits = outputs.logits

        if self.use_focal_loss:
            # Focal loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
            # alpha_t is handled via class_weights_tensor (if provided)
            ce_loss = nn.CrossEntropyLoss(
                weight=self.class_weights_tensor, reduction='none'
            )(logits, labels)
            probs = torch.softmax(logits, dim=1)
            # p_t: probability assigned to the correct class
            p_t = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            focal_weight = (1.0 - p_t) ** self.focal_gamma
            loss = (focal_weight * ce_loss).mean()
        else:
            loss_fn = nn.CrossEntropyLoss(weight=self.class_weights_tensor)
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
        """Compute evaluation metrics for training"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        return {
            'accuracy': accuracy_score(labels, predictions),
            'f1': f1_score(labels, predictions, average='weighted', zero_division=0),
            'precision': precision_score(labels, predictions, average='weighted', zero_division=0),
            'recall': recall_score(labels, predictions, average='weighted', zero_division=0),
        }
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        label_column: str,
        text_column: str,
        test_size: float = 0.2,
    ) -> Tuple[Dict, LabelEncoder]:
        """
        Prepare and split data
        
        Args:
            df: DataFrame
            label_column: Label column name
            text_column: Text column name
            test_size: Test split size
            
        Returns:
            Tuple of (data_dict, label_encoder)
        """
        # Split data first — fit LabelEncoder only on train labels to prevent leakage
        texts = df[text_column].tolist()
        raw_labels = df[label_column].tolist()
        train_texts, val_texts, train_labels_raw, val_labels_raw = train_test_split(
            texts, raw_labels, test_size=test_size, random_state=42, stratify=raw_labels
        )

        le = LabelEncoder()
        train_labels = le.fit_transform(train_labels_raw)
        val_labels   = le.transform(val_labels_raw)

        logger.info(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        data = {
            'train_texts': train_texts,
            'train_labels': train_labels,
            'val_texts': val_texts,
            'val_labels': val_labels,
            'num_classes': len(le.classes_),
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
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        early_stopping_patience: int = 3,
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
            warmup_steps: Warmup steps
            weight_decay: Weight decay
            max_grad_norm: Max gradient norm for clipping
            early_stopping_patience: Early stopping patience
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Training {model_name}")
        logger.info(f"{'='*70}")
        logger.info(f"Epochs: {num_epochs}, Batch: {batch_size}, LR: {learning_rate}")
        logger.info(f"Grad Accumulation: {gradient_accumulation_steps}, Warmup: {warmup_steps}")
        if use_class_weights and class_weights:
            logger.info(f"✓ Using class weights: {class_weights}")
        logger.info(f"{'='*70}\n")
        
        start_time = time.time()
        
        # Load tokenizer and model
        logger.info("Loading tokenizer and model...")
        # prajjwal1/bert-tiny has no tokenizer files — it shares bert-base-uncased vocab
        TOKENIZER_OVERRIDES = {
            "prajjwal1/bert-tiny":   "bert-base-uncased",
            "prajjwal1/bert-mini":   "bert-base-uncased",
            "prajjwal1/bert-small":  "bert-base-uncased",
            "prajjwal1/bert-medium": "bert-base-uncased",
        }
        tokenizer_name = TOKENIZER_OVERRIDES.get(model_name, model_name)
        if tokenizer_name != model_name:
            logger.info(f"Using tokenizer '{tokenizer_name}' for model '{model_name}'")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        model_cls = MODEL_CLASS_OVERRIDES.get(model_name, AutoModelForSequenceClassification)
        if model_cls is not AutoModelForSequenceClassification:
            logger.info(f"Using explicit class '{model_cls.__name__}' for model '{model_name}'")
        model = model_cls.from_pretrained(
            model_name,
            num_labels=data['num_classes'],
        ).to(self.device)
        
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
        
        training_args = TrainingArguments(
            output_dir=str(model_output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            max_grad_norm=max_grad_norm,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            save_total_limit=1,          # keep only the single best checkpoint → saves disk space
            logging_steps=max(1, len(train_dataset) // (batch_size * 4)),  # Log 4 times per epoch
            seed=42,
            report_to=[],  # Disable wandb/tensorboard logging
            dataloader_pin_memory=torch.cuda.is_available(),  # only pin memory when a GPU is present
        )
        
        # Prepare class weights tensor (map string class names → integer indices)
        if use_class_weights and class_weights:
            le = data['label_encoder']
            int_weights = [class_weights.get(le.classes_[i], 1.0) for i in range(data['num_classes'])]
            class_weights_tensor = torch.tensor(int_weights).float().to(self.device)
            logger.info(f"Class weights tensor: {class_weights_tensor}")
        else:
            class_weights_tensor = None

        # Create trainer (weighted/focal when applicable, standard otherwise)
        use_custom_loss = class_weights_tensor is not None or use_focal_loss
        trainer_cls = WeightedTrainer if use_custom_loss else Trainer
        trainer_kwargs = dict(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
        )
        if use_custom_loss:
            trainer_kwargs['class_weights_tensor'] = class_weights_tensor
            trainer_kwargs['use_focal_loss'] = use_focal_loss
            if use_focal_loss:
                logger.info("Using focal loss (gamma=2.0) to handle class imbalance")
        trainer = trainer_cls(**trainer_kwargs)
        
        # Train
        try:
            logger.info("Starting training...")
            trainer.train()
            logger.info("✓ Training completed successfully")
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
                    warmup_steps = merged_config.get('warmup_steps', 500)
                    weight_decay = merged_config.get('weight_decay', 0.01)
                    max_grad_norm = merged_config.get('max_grad_norm', 1.0)
                    early_stopping_patience = merged_config.get('early_stopping_patience', 3)
                    
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
                            warmup_steps=warmup_steps,
                            weight_decay=weight_decay,
                            max_grad_norm=max_grad_norm,
                            early_stopping_patience=early_stopping_patience,
                        )
                        all_results.append(result)

                    except Exception as e:
                        logger.error(f"Error training {model_name} with epochs={num_epochs}: {str(e)}")
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
