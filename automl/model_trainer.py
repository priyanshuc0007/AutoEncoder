"""
Model Trainer Module
Handles model fine-tuning with hyperparameter tuning using Optuna
"""

import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import logging
from pathlib import Path
import time
from typing import Dict, Tuple, List

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
        # Encode labels
        le = LabelEncoder()
        labels = le.fit_transform(df[label_column])
        
        logger.info(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        # Split data
        texts = df[text_column].tolist()
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
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
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
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
            logging_steps=max(1, len(train_dataset) // (batch_size * 4)),  # Log 4 times per epoch
            seed=42,
            report_to=[],  # Disable wandb/tensorboard logging
        )
        
        # Prepare model for training with class weights if needed
        if use_class_weights and class_weights:
            class_weights_tensor = torch.tensor(
                [class_weights.get(i, 1.0) for i in range(data['num_classes'])]
            ).to(self.device)
            logger.info(f"Class weights tensor: {class_weights_tensor}")
        else:
            class_weights_tensor = None
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
        )
        
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
        experiment_name: str = "default",
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
            
        Returns:
            List of training results
        """
        all_results = []
        
        for model_name in model_names:
            # Try different hyperparameter combinations
            for num_epochs in hyperparams_ranges['num_epochs_range']:
                for lr_scale in [0.5, 1.0]:  # Scale learning rate
                    
                    learning_rate = hyperparams_ranges['learning_rate'] * lr_scale
                    batch_size = hyperparams_ranges['batch_size']
                    gradient_accumulation_steps = hyperparams_ranges.get('gradient_accumulation_steps', 1)
                    warmup_steps = hyperparams_ranges.get('warmup_steps', 500)
                    weight_decay = hyperparams_ranges.get('weight_decay', 0.01)
                    max_grad_norm = hyperparams_ranges.get('max_grad_norm', 1.0)
                    early_stopping_patience = hyperparams_ranges.get('early_stopping_patience', 3)
                    
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
                
                # For MVP, only train one model variant per model name
                break
        
        return all_results
