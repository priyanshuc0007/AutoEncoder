"""
AutoLLM Pipeline Orchestrator
Coordinates all components: validation, intelligence, training, evaluation
"""

import pickle
import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import pandas as pd

from automl.data_validator import DataValidator
from automl.data_intelligence import DataIntelligence
from automl.model_trainer import ModelTrainer
from automl.evaluator import ModelEvaluator
from automl.cross_validator import CrossValidator

# Trust layer — non-invasive reliability and transparency modules
from automl.trust.reproducibility import set_global_seeds
from automl.trust.pipeline_tracker import PipelineTracker
from automl.trust.decisions_logger import DecisionsLogger
from automl.trust.baseline import compute_majority_baseline, save_baseline_comparison
from automl.trust.data_quality import check_data_quality
from automl.trust.explainability import run_explainability

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoLLMPipeline:
    """Main AutoLLM pipeline orchestrator"""
    
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
        self._df = None            # full validated DataFrame (needed for CV)
        self._label_column = None  # resolved label column name
        self._text_column = None   # resolved text column name

        # Trust layer objects (never affect training or return values)
        self._seed = 42
        self.tracker = PipelineTracker()
        self.decisions = DecisionsLogger()

    def run(
        self,
        csv_path: str,
        label_column: str,
        text_column: Optional[str] = None,
        text_columns: Optional[list] = None,
        experiment_name: Optional[str] = None,
        use_cv: bool = False,
        cv_folds: int = 5,
        use_optuna: bool = False,
        optuna_trials: int = 10,
    ) -> Dict:
        """
        Run complete AutoLLM pipeline
        
        Args:
            csv_path: Path to CSV file
            label_column: Label column name
            text_column: Single text column name (auto-detected if None)
            text_columns: List of text columns to merge (overrides text_column if provided)
            experiment_name: Name for this experiment
            use_cv: Whether to run cross-validation on the best model
            cv_folds: Number of CV folds (used only when use_cv=True)
            use_optuna: Whether to run Optuna hyperparameter search (lr + weight_decay)
            optuna_trials: Number of Optuna trials per model (clamped 3–20)
            
        Returns:
            Dictionary with pipeline results
        """
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.experiment_dir = self.output_dir / experiment_name
        self.experiment_dir.mkdir(exist_ok=True)

        # Trust: fix all random seeds + start step tracking
        try:
            set_global_seeds(self._seed)
            self.tracker.start(experiment_name, self.experiment_dir, self._seed)
        except Exception as _e:
            logger.debug("[Trust] seed/tracker init failed (non-fatal): %s", _e)

        logger.info(f"\n{'='*70}")
        logger.info("🚀 AUTOLLM PIPELINE STARTED")
        logger.info(f"Experiment: {experiment_name}")
        logger.info(f"{'='*70}\n")
        
        try:
            # Step 1: Validate and load data
            try: self.tracker.begin_step("data_validation")
            except Exception: pass
            logger.info("\n📂 STEP 1: Data Validation")
            logger.info("-" * 70)
            df, label_column, text_column = self.validator.load_and_validate(
                csv_path, label_column, text_column
            )
            
            # Detect all text columns — used for informational logging and
            # to provide the primary column when text_columns is not supplied
            # AND the caller did not explicitly specify text_column.
            detected_columns = self.validator.detect_text_columns(df, label_column)

            # Only fall back to auto-detection when the caller gave neither
            # text_column nor text_columns — an explicit text_column must not
            # be silently replaced by a longer-text column.
            if not text_columns and text_column is None:
                text_column = detected_columns['primary_text_column']
                logger.info(f"Using auto-detected primary text column: {text_column}")
            elif not text_columns:
                logger.info(f"Using caller-supplied text column: {text_column}")
            
            # Validate every entry in text_columns before use
            if text_columns:
                for col in text_columns:
                    if col not in df.columns:
                        raise ValueError(
                            f"text_columns entry '{col}' not found in CSV columns: "
                            f"{list(df.columns)}"
                        )

            # If multiple columns specified, merge them
            if text_columns and len(text_columns) > 1:
                logger.info(f"\nMerging multiple text columns: {text_columns}")
                df, text_column = self.validator.merge_text_columns(df, text_columns)
            elif text_columns and len(text_columns) == 1:
                text_column = text_columns[0]
                logger.info(f"Using single text column: {text_column}")
            
            logger.info(f"Final text column for training: {text_column}")

            # Store for use by optional CV step (needs full df, not just train split)
            self._df = df
            self._label_column = label_column
            self._text_column = text_column

            # Trust: data quality audit (saved as data_quality_report.txt)
            try:
                check_data_quality(df, text_column, label_column, self.experiment_dir)
            except Exception as e:
                logger.warning(f"Data quality check skipped: {e}")
            try:
                self.tracker.complete_step("data_validation")
                self.tracker.begin_step("data_intelligence")
            except Exception:
                pass

            # Step 2: Data intelligence
            logger.info("\n🧠 STEP 2: Data Intelligence Analysis")
            logger.info("-" * 70)
            self.analysis = self.intelligence.analyze(df, label_column, text_column)
            self.intelligence.print_summary(self.analysis)

            # Trust: log all automated decisions (saved as decisions_log.json)
            try:
                self.decisions.log_from_analysis(self.analysis)
                self.decisions.save(self.experiment_dir)
                self.tracker.complete_step("data_intelligence")
                self.tracker.begin_step("data_preparation")
            except Exception as _e:
                logger.debug("[Trust] decisions/tracker failed (non-fatal): %s", _e)

            # Step 3: Prepare data
            logger.info("\n🔄 STEP 3: Data Preparation")
            logger.info("-" * 70)
            self.data, label_encoder = self.trainer.prepare_data(
                df, label_column, text_column
            )

            # Recompute class weights from training labels only — prevents val label
            # distribution from leaking into the training loss via class weight statistics.
            train_class_weights = self.analysis['task_info']['class_weights']  # fallback
            if self.analysis['imbalance_info']['use_class_weights']:
                train_label_strings = pd.Series(
                    label_encoder.inverse_transform(self.data['train_labels'])
                )
                train_class_weights = self.intelligence._compute_class_weights(train_label_strings)
                logger.info("ℹ️  Class weights recomputed from training set only (no val leakage)")

            # Step 4: Train models
            try:
                self.tracker.complete_step("data_preparation")
                self.tracker.begin_step("model_training")
            except Exception: pass
            logger.info("\n🤖 STEP 4: Model Training")
            logger.info("-" * 70)
            self.training_results = self.trainer.train_multiple_models(
                model_names=self.analysis['model_selection'],
                data=self.data,
                hyperparams_ranges=self.analysis['training_config'],
                max_length=self.analysis['text_info']['p95_length'],
                use_class_weights=self.analysis['imbalance_info']['use_class_weights'],
                class_weights=train_class_weights,
                use_focal_loss=self.analysis['imbalance_info']['use_focal_loss'],
                experiment_name=experiment_name,
                use_optuna=use_optuna,
                optuna_trials=optuna_trials,
            )
            
            logger.info(f"\n✓ Trained {len(self.training_results)} model(s)")
            
            # Check if any models were trained
            if not self.training_results:
                logger.error("❌ No models were successfully trained!")
                try:
                    self.tracker.mark_failed("No models were successfully trained")
                except Exception as _e:
                    logger.debug("[Trust] tracker mark_failed skipped: %s", _e)
                return {
                    'status': 'failed',
                    'error': 'No models were successfully trained. Check your data and configuration.',
                    'experiment_name': experiment_name,
                }
            
            # Step 5: Evaluate models
            try:
                self.tracker.complete_step("model_training")
                self.tracker.begin_step("model_evaluation")
            except Exception: pass
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
                try:
                    self.tracker.mark_failed("No best model could be selected from evaluation results")
                except Exception as _e:
                    logger.debug("[Trust] tracker mark_failed skipped: %s", _e)
                return {
                    'status': 'failed',
                    'error': 'No best model could be selected from evaluation results.',
                    'experiment_name': experiment_name,
                }
            
            best_model_path = best_model_result['model_path']

            # Trust: baseline comparison (saved as baseline_comparison.txt)
            try:
                baseline = compute_majority_baseline(self.data['val_labels'])
                save_baseline_comparison(
                    baseline, best_model_result, self.experiment_dir, label_encoder
                )
                self.tracker.complete_step("model_evaluation")
                self.tracker.begin_step("best_model_selection")
                self.tracker.complete_step("best_model_selection")
                self.tracker.begin_step("report_generation")
            except Exception as _e:
                logger.debug("[Trust] baseline/tracker failed (non-fatal): %s", _e)

            # Step 7: Generate report
            logger.info("\n📋 STEP 7: Report Generation")
            logger.info("-" * 70)

            # Also evaluate best model on training data for overfitting comparison
            # Resolve the tokenizer that belongs to the winning model
            best_tr = next(
                (r for r in self.training_results if r['model_path'] == best_model_path),
                self.training_results[0],
            )

            best_train_result = None
            try:
                best_train_result = self.evaluator.evaluate_model(
                    model_path=best_model_path,
                    texts=self.data['train_texts'],
                    labels=self.data['train_labels'],
                    tokenizer=best_tr['tokenizer'],
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
            try:
                self.evaluator.generate_report(
                    best_model_result,
                    label_encoder,
                    output_path=str(report_path),
                    train_result=best_train_result,
                )
            except Exception as e:
                logger.warning(f"Report generation failed (model still saved): {e}")

            # Save configuration — non-fatal: fast tokenizers are not always picklable
            try:
                self._save_configuration(label_encoder)
            except Exception as e:
                logger.warning(f"Could not save configuration pickle: {e}")

            # Trust: explainability & reliability (Pillar 5)
            logger.info("\n🔍 TRUST PILLAR 5: Explainability & Reliability")
            logger.info("-" * 70)
            explainability_data = {}
            try:
                explainability_data = run_explainability(
                    model_path=best_model_path,
                    tokenizer=best_tr['tokenizer'],
                    val_texts=self.data['val_texts'],
                    val_labels=self.data['val_labels'],
                    max_length=self.analysis['text_info']['p95_length'],
                    experiment_dir=self.experiment_dir,
                    label_encoder=label_encoder,
                )
            except Exception as e:
                logger.warning(f"Explainability step skipped: {e}")

            # Optional: cross-validation on best model architecture
            cv_results = None
            if use_cv:
                logger.info(f"\n🔁 CROSS-VALIDATION ({cv_folds}-fold) — {best_model_result.get('model_name')}")
                logger.info("-" * 70)
                logger.info("NOTE: Each fold retrains the model from scratch. This will take "
                            f"~{cv_folds}× the time of a single training run.")
                try:
                    best_model_name_for_cv = best_model_result.get('model_name') or \
                        best_tr.get('model_name', 'distilbert-base-uncased')
                    cv = CrossValidator(output_dir="models")
                    cv_results = cv.run(
                        model_name=best_model_name_for_cv,
                        df=self._df,
                        label_column=self._label_column,
                        text_column=self._text_column,
                        hyperparams=self.analysis['training_config'],
                        max_length=self.analysis['text_info']['p95_length'],
                        label_encoder=label_encoder,
                        use_class_weights=self.analysis['imbalance_info']['use_class_weights'],
                        class_weights=train_class_weights,
                        use_focal_loss=self.analysis['imbalance_info']['use_focal_loss'],
                        n_splits=cv_folds,
                        experiment_name=experiment_name,
                    )
                    # Save CV summary to experiment dir
                    self._save_cv_report(cv_results)
                except Exception as e:
                    logger.warning(f"Cross-validation skipped: {e}")

            # Trust: finalise tracking
            try:
                self.tracker.complete_step("report_generation")
                self.tracker.finish()
            except Exception as _e:
                logger.debug("[Trust] tracker finish failed (non-fatal): %s", _e)

            logger.info(f"\n{'='*70}")
            logger.info("✓ AUTOLLM PIPELINE COMPLETED SUCCESSFULLY")
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
                'explainability': explainability_data,
                'cv_results': cv_results,
            }
            
        except Exception as e:
            # Trust: record failure in pipeline state
            try:
                self.tracker.mark_failed(str(e))
            except Exception as _e:
                logger.debug("[Trust] tracker mark_failed skipped: %s", _e)
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

    def _save_cv_report(self, cv_results: Dict) -> None:
        """Write a human-readable cross_validation_report.txt to the experiment folder."""
        path = self.experiment_dir / "cross_validation_report.txt"
        sep = "=" * 70
        lines = [
            sep,
            f"CROSS-VALIDATION REPORT  ({cv_results['n_splits']}-fold Stratified)",
            sep,
            f"Model         : {cv_results['model_name']}",
            f"Folds         : {cv_results['n_splits']}  "
            f"({cv_results['n_successful_folds']} successful)",
            "",
        ]

        # Show error reason if CV failed before any folds ran
        if cv_results.get("error") and cv_results["n_successful_folds"] == 0:
            lines += [
                "CV FAILED",
                "-" * 50,
                f"  Reason: {cv_results['error']}",
                "",
            ]

        summary = cv_results.get("summary", {})
        if summary:
            lines += [
                "SUMMARY  (mean ± std  across successful folds)",
                "-" * 50,
                f"  F1 Score  : {summary['f1']['mean']:.4f} ± {summary['f1']['std']:.4f}"
                f"  (min {summary['f1']['min']:.4f} / max {summary['f1']['max']:.4f})",
                f"  Accuracy  : {summary['accuracy']['mean']:.4f} ± {summary['accuracy']['std']:.4f}",
                f"  Precision : {summary['precision']['mean']:.4f} ± {summary['precision']['std']:.4f}",
                f"  Recall    : {summary['recall']['mean']:.4f} ± {summary['recall']['std']:.4f}",
                "",
            ]

        lines += [
            "PER-FOLD RESULTS",
            "-" * 50,
            f"  {'Fold':<6} {'Train':>6} {'Val':>6} {'F1':>8} {'Accuracy':>10} {'Status'}",
            "  " + "-" * 46,
        ]
        for r in cv_results.get("fold_results", []):
            if r["f1"] is not None:
                lines.append(
                    f"  {r['fold']:<6} {r['n_train']:>6} {r['n_val']:>6} "
                    f"{r['f1']:>8.4f} {r['accuracy']:>10.4f}  ✅"
                )
            else:
                lines.append(
                    f"  {r['fold']:<6} {r['n_train']:>6} {r['n_val']:>6} "
                    f"{'N/A':>8} {'N/A':>10}  ❌ {r.get('error', '')[:40]}"
                )

        lines.append(sep)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            logger.info(f"Cross-validation report saved to {path}")
        except Exception as e:
            logger.warning(f"Could not save CV report: {e}")
