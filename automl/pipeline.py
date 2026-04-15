"""
Main AutoML Pipeline Orchestrator
Coordinates all components: validation, intelligence, training, evaluation
"""

import pickle
import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

from automl.data_validator import DataValidator
from automl.data_intelligence import DataIntelligence
from automl.model_trainer import ModelTrainer
from automl.evaluator import ModelEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoMLPipeline:
    """Main AutoML pipeline orchestrator"""
    
    def __init__(self, output_dir: str = "experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.validator = DataValidator()
        self.intelligence = DataIntelligence()
        self.trainer = ModelTrainer(output_dir="models")
        self.evaluator = ModelEvaluator()
        
        self.experiment_dir = None
        self.analysis = None
        self.data = None
        self.training_results = None
        self.comparison = None
    
    def run(
        self,
        csv_path: str,
        label_column: str,
        text_column: Optional[str] = None,
        text_columns: Optional[list] = None,
        experiment_name: Optional[str] = None,
    ) -> Dict:
        """
        Run complete AutoML pipeline
        
        Args:
            csv_path: Path to CSV file
            label_column: Label column name
            text_column: Single text column name (auto-detected if None)
            text_columns: List of text columns to merge (overrides text_column if provided)
            experiment_name: Name for this experiment
            
        Returns:
            Dictionary with pipeline results
        """
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.experiment_dir = self.output_dir / experiment_name
        self.experiment_dir.mkdir(exist_ok=True)
        
        logger.info(f"\n{'='*70}")
        logger.info("🚀 AUTOML PIPELINE STARTED")
        logger.info(f"Experiment: {experiment_name}")
        logger.info(f"{'='*70}\n")
        
        try:
            # Step 1: Validate and load data
            logger.info("\n📂 STEP 1: Data Validation")
            logger.info("-" * 70)
            df, label_column, text_column = self.validator.load_and_validate(
                csv_path, label_column, text_column
            )
            
            # Detect all text columns if not specified
            detected_columns = self.validator.detect_text_columns(df, label_column)
            
            # If multiple columns specified, merge them
            if text_columns and len(text_columns) > 1:
                logger.info(f"\nMerging multiple text columns: {text_columns}")
                df, text_column = self.validator.merge_text_columns(df, text_columns)
            elif text_columns and len(text_columns) == 1:
                text_column = text_columns[0]
                logger.info(f"Using single text column: {text_column}")
            
            logger.info(f"Final text column for training: {text_column}")
            
            # Step 2: Data intelligence
            logger.info("\n🧠 STEP 2: Data Intelligence Analysis")
            logger.info("-" * 70)
            self.analysis = self.intelligence.analyze(df, label_column, text_column)
            self.intelligence.print_summary(self.analysis)
            
            # Step 3: Prepare data
            logger.info("\n🔄 STEP 3: Data Preparation")
            logger.info("-" * 70)
            self.data, label_encoder = self.trainer.prepare_data(
                df, label_column, text_column
            )
            
            # Step 4: Train models
            logger.info("\n🤖 STEP 4: Model Training")
            logger.info("-" * 70)
            self.training_results = self.trainer.train_multiple_models(
                model_names=self.analysis['model_selection'],
                data=self.data,
                hyperparams_ranges=self.analysis['training_config'],
                max_length=self.analysis['text_info']['p95_length'],
                use_class_weights=self.analysis['imbalance_info']['use_class_weights'],
                class_weights=self.analysis['task_info']['class_weights'],
                experiment_name=experiment_name,
            )
            
            logger.info(f"\n✓ Trained {len(self.training_results)} model(s)")
            
            # Check if any models were trained
            if not self.training_results:
                logger.error("❌ No models were successfully trained!")
                return {
                    'status': 'failed',
                    'error': 'No models were successfully trained. Check your data and configuration.',
                    'experiment_name': experiment_name,
                }
            
            # Step 5: Evaluate models
            logger.info("\n📊 STEP 5: Model Evaluation")
            logger.info("-" * 70)

            evaluation_results = []
            for train_result in self.training_results:
                eval_result = self.evaluator.evaluate_model(
                    model_path=train_result['model_path'],
                    texts=self.data['val_texts'],
                    labels=self.data['val_labels'],
                    tokenizer=train_result['tokenizer'],
                    max_length=self.analysis['text_info']['p95_length'],
                    label_encoder=label_encoder,
                    split='val',
                )

                # Merge with training result
                eval_result['model_name'] = train_result['model_name']
                eval_result['model_path'] = train_result['model_path']
                eval_result['training_time'] = train_result['training_time']
                evaluation_results.append(eval_result)
            
            # Step 6: Select best model
            logger.info("\n🏆 STEP 6: Best Model Selection")
            logger.info("-" * 70)
            self.comparison = self.evaluator.compare_models(evaluation_results)
            
            best_model_result = self.comparison['best_model']
            
            # Check if a model was selected
            if best_model_result is None:
                logger.error("❌ No best model could be selected!")
                return {
                    'status': 'failed',
                    'error': 'No best model could be selected from evaluation results.',
                    'experiment_name': experiment_name,
                }
            
            best_model_path = best_model_result['model_path']
            
            # Step 7: Generate report
            logger.info("\n📋 STEP 7: Report Generation")
            logger.info("-" * 70)

            # Also evaluate best model on training data for overfitting comparison
            best_train_result = None
            try:
                best_train_result = self.evaluator.evaluate_model(
                    model_path=best_model_path,
                    texts=self.data['train_texts'],
                    labels=self.data['train_labels'],
                    tokenizer=self.training_results[0]['tokenizer'],
                    max_length=self.analysis['text_info']['p95_length'],
                    label_encoder=label_encoder,
                    split='train',
                )
                best_train_result['model_name'] = best_model_result.get('model_name')
                logger.info(
                    f"Train  — Accuracy: {best_train_result['accuracy']:.4f}, "
                    f"F1: {best_train_result['f1_score']:.4f}"
                )
                logger.info(
                    f"Val    — Accuracy: {best_model_result['accuracy']:.4f}, "
                    f"F1: {best_model_result['f1_score']:.4f}"
                )
                gap = best_train_result['f1_score'] - best_model_result['f1_score']
                logger.info(f"F1 Gap (train - val): {gap:+.4f}")
            except Exception as e:
                logger.warning(f"Could not evaluate on training data: {e}")

            report_path = self.experiment_dir / "best_model_report.txt"
            self.evaluator.generate_report(
                best_model_result,
                label_encoder,
                output_path=str(report_path),
                train_result=best_train_result,
            )
            
            # Save configuration
            self._save_configuration(label_encoder)
            
            logger.info(f"\n{'='*70}")
            logger.info("✓ AUTOML PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"Best Model: {best_model_result.get('model_name', 'Unknown')}")
            logger.info(f"F1 Score: {best_model_result['f1_score']:.4f}")
            logger.info(f"Experiment saved to: {self.experiment_dir}")
            logger.info(f"{'='*70}\n")
            
            return {
                'status': 'success',
                'experiment_name': experiment_name,
                'best_model_path': best_model_path,
                'best_model_name': best_model_result.get('model_name'),
                'best_model_metrics': {
                    'f1_score': best_model_result['f1_score'],
                    'accuracy': best_model_result['accuracy'],
                    'latency_ms': best_model_result['avg_inference_time_ms'],
                },
                'experiment_dir': str(self.experiment_dir),
                'data_analysis': self.analysis,
                'comparison_df': self.comparison['comparison_df'],
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            return {
                'status': 'failed',
                'error': str(e),
                'experiment_name': experiment_name,
            }
    
    def _save_configuration(self, label_encoder) -> None:
        """Save experiment configuration"""
        config = {
            'analysis': self.analysis,
            'training_results': self.training_results,
            'label_encoder': label_encoder,
            'timestamp': datetime.now().isoformat(),
        }
        
        config_path = self.experiment_dir / "config.pkl"
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)
        
        logger.info(f"Configuration saved to {config_path}")
