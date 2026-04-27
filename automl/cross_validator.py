"""
Cross-Validator
================
Runs stratified k-fold cross-validation on the best model architecture
chosen by the main pipeline.  Each fold trains a fresh model, evaluates
on the held-out fold, then cleans up the checkpoint to save disk space.

Why cross-validation?
---------------------
A single 80/20 split can be lucky or unlucky depending on how the rows
fall.  k-fold CV gives a statistically reliable estimate: mean ± std
across k independent splits of the data.

  k=5 (default) → 5 models trained, each on 80% of data.
  k=3            → 3 models (faster, less reliable).

The CV only runs on the BEST model architecture selected by the pipeline,
not on all candidates (that would multiply cost by num_models × k).

Fold checkpoints are deleted after each evaluation — only the summary
statistics and per-fold numbers are kept.
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class CrossValidator:
    """Stratified k-fold cross-validator for the best model architecture."""

    def __init__(self, output_dir: str = "models"):
        from automl.model_trainer import ModelTrainer
        from automl.evaluator import ModelEvaluator

        self.trainer = ModelTrainer(output_dir=output_dir)
        self.evaluator = ModelEvaluator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        model_name: str,
        df: pd.DataFrame,
        label_column: str,
        text_column: str,
        hyperparams: Dict,
        max_length: int,
        label_encoder: LabelEncoder,
        use_class_weights: bool,
        class_weights: Dict,
        use_focal_loss: bool,
        n_splits: int = 5,
        experiment_name: str = "cv",
        is_multi_label: bool = False,
    ) -> Dict:
        """
        Run stratified k-fold CV for *model_name*.

        Args:
            model_name       : HuggingFace model identifier (e.g. 'prajjwal1/bert-tiny').
            df               : Full validated DataFrame (before any train/val split).
            label_column     : Name of the label column.
            text_column      : Name of the (merged) text column.
            hyperparams      : Training config dict from DataIntelligence.analyze().
            max_length       : Max token length used during training.
            label_encoder    : Fitted LabelEncoder (already fit on full dataset labels).
            use_class_weights: Whether to apply class weighting.
            class_weights    : Dict {class_name: weight} from the analysis.
            use_focal_loss   : Whether to use focal loss.
            n_splits         : Number of folds (default 5).
            experiment_name  : Prefix for fold checkpoint directories.

        Returns:
            Dict with:
              model_name        : model identifier
              n_splits          : number of folds requested
              n_successful_folds: number of folds that completed without error
              fold_results      : list of per-fold metric dicts
              summary           : {metric: {mean, std, min, max}} for each metric
        """
        texts = df[text_column].tolist()
        num_classes = len(label_encoder.classes_)

        # ── Multi-label: random KFold instead of stratified ────────────────────
        if is_multi_label:
            from sklearn.model_selection import KFold
            from sklearn.preprocessing import MultiLabelBinarizer
            from automl.data_validator import DataValidator
            parsed_all = DataValidator._parse_multi_labels(df[label_column])
            # Re-fit a fresh MultiLabelBinarizer on the FULL CV dataset so all labels
            # are covered, not just those seen in the original 80% training split.
            # NOTE: this MLB is used *only inside this CV run* for fold encoding/decoding.
            # It intentionally has a different class order than the pipeline's MLB when
            # rare labels only appear in the held-out portion. CV metrics are internally
            # consistent (same MLB for encode + decode within every fold) even if the
            # class indices differ from the main pipeline's MLB.
            fresh_mlb = MultiLabelBinarizer()
            multi_hot_all = fresh_mlb.fit_transform(parsed_all).astype('float32')
            num_classes = len(fresh_mlb.classes_)
            label_encoder = fresh_mlb  # use the full-data MLB for this CV run
            encoded_labels = multi_hot_all
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            # dummy X for KFold.split (it only needs length)
            try:
                splits = list(splitter.split(texts))
            except ValueError as e:
                logger.error(f"KFold failed for multi-label CV: {e}")
                return {
                    'model_name': model_name,
                    'n_splits': n_splits,
                    'n_successful_folds': 0,
                    'fold_results': [],
                    'summary': {},
                }
        else:
            # Re-fit a fresh LabelEncoder on the full CV dataframe so it covers all
            # classes, not just those present in the original 80% training slice.
            from sklearn.preprocessing import LabelEncoder as _LE
            fresh_le = _LE()
            encoded_labels = fresh_le.fit_transform(df[label_column].tolist())
            num_classes = len(fresh_le.classes_)
            label_encoder = fresh_le  # use full-data LE for this CV run

            # Guard: StratifiedKFold requires ≥ n_splits samples per class.
            # Auto-reduce folds to the smallest class count (floor 2) when needed.
            _min_cls_count = int(pd.Series(encoded_labels).value_counts().min())
            if _min_cls_count < n_splits:
                _safe_splits = max(2, _min_cls_count)
                logger.warning(
                    f"⚠️ Smallest class has only {_min_cls_count} samples — "
                    f"reducing CV folds from {n_splits} to {_safe_splits}."
                )
                n_splits = _safe_splits

            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            try:
                splits = list(skf.split(texts, encoded_labels))
            except ValueError as e:
                logger.error(
                    f"StratifiedKFold failed — not enough samples per class for "
                    f"{n_splits}-fold CV: {e}"
                )
                return {
                    'model_name': model_name,
                    'n_splits': n_splits,
                    'n_successful_folds': 0,
                    'fold_results': [],
                    'summary': {},
                }

        # Use the minimum epoch count from the range for CV — keeps each fold fast
        num_epochs = hyperparams.get("num_epochs_range", [3, 5])[0]

        logger.info(f"\n{'='*70}")
        logger.info(f"CROSS-VALIDATION  ({n_splits}-fold)  model={model_name}")
        logger.info(
            f"Training {num_epochs} epoch(s) per fold  |  "
            f"{len(texts)} total samples  |  {'multi_label' if is_multi_label else f'{num_classes} classes'}"
        )
        logger.info(f"{'='*70}")

        fold_results: List[Dict] = []

        for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
            logger.info(f"\n─── Fold {fold_idx}/{n_splits}  "
                        f"(train={len(train_idx)}, val={len(val_idx)}) ───")

            train_texts = [texts[i] for i in train_idx]
            val_texts   = [texts[i] for i in val_idx]
            train_labels = encoded_labels[train_idx]
            val_labels   = encoded_labels[val_idx]

            fold_data = {
                "train_texts":  train_texts,
                "train_labels": train_labels,
                "val_texts":    val_texts,
                "val_labels":   val_labels,
                "num_classes":  num_classes,
                "label_encoder": label_encoder,
            }

            fold_exp_name = f"{experiment_name}_cv{n_splits}f{fold_idx}"

            # Recompute class weights from this fold's training labels only —
            # prevents the held-out fold's label distribution from leaking into
            # the training loss via class weight statistics.
            fold_class_weights = class_weights  # fallback (used when weighting is off)
            if use_class_weights and not is_multi_label:
                from automl.data_intelligence import DataIntelligence as _DI
                fold_train_strings = pd.Series(
                    label_encoder.inverse_transform(train_labels)
                )
                fold_class_weights = _DI()._compute_class_weights(fold_train_strings)

            try:
                train_result = self.trainer.train_model(
                    model_name=model_name,
                    data=fold_data,
                    num_epochs=num_epochs,
                    batch_size=hyperparams.get("batch_size", 16),
                    learning_rate=hyperparams.get("learning_rate", 2e-5),
                    max_length=max_length,
                    use_class_weights=use_class_weights,
                    class_weights=fold_class_weights,
                    use_focal_loss=use_focal_loss,
                    experiment_name=fold_exp_name,
                    gradient_accumulation_steps=hyperparams.get(
                        "gradient_accumulation_steps", 1
                    ),
                    warmup_ratio=hyperparams.get("warmup_ratio", 0.06),
                    weight_decay=hyperparams.get("weight_decay", 0.01),
                    max_grad_norm=hyperparams.get("max_grad_norm", 1.0),
                    early_stopping_patience=hyperparams.get("early_stopping_patience", 3),
                    lr_scheduler_type=hyperparams.get("lr_scheduler_type", "linear"),
                    dropout=hyperparams.get("dropout", 0.1),
                    focal_gamma=hyperparams.get("focal_gamma", 2.0),
                    label_smoothing_factor=hyperparams.get("label_smoothing_factor", 0.0),
                    is_multi_label=is_multi_label,
                )

                eval_result = self.evaluator.evaluate_model(
                    model_path=train_result["model_path"],
                    texts=val_texts,
                    labels=val_labels,
                    tokenizer=train_result["tokenizer"],
                    max_length=max_length,
                    label_encoder=label_encoder,
                    split="val",
                    is_multi_label=is_multi_label,
                )

                fold_results.append(
                    {
                        "fold":      fold_idx,
                        "f1":        eval_result["f1_score"],
                        "accuracy":  eval_result["accuracy"],
                        "precision": eval_result["precision"],
                        "recall":    eval_result["recall"],
                        "n_train":   int(len(train_idx)),
                        "n_val":     int(len(val_idx)),
                        "error":     None,
                    }
                )

                logger.info(
                    f"Fold {fold_idx} ✓ — F1={eval_result['f1_score']:.4f}  "
                    f"Acc={eval_result['accuracy']:.4f}"
                )

            except Exception as exc:
                logger.error(f"Fold {fold_idx} failed: {exc}")
                fold_results.append(
                    {
                        "fold":    fold_idx,
                        "f1":      None,
                        "accuracy": None,
                        "precision": None,
                        "recall":  None,
                        "n_train": int(len(train_idx)),
                        "n_val":   int(len(val_idx)),
                        "error":   str(exc),
                    }
                )

            finally:
                # Always clean up fold checkpoint — only the stats are kept
                fold_model_path = Path(
                    self.trainer.output_dir
                    / f"{fold_exp_name}_{model_name.split('/')[-1]}"
                )
                if fold_model_path.exists():
                    shutil.rmtree(fold_model_path, ignore_errors=True)
                    logger.info(f"Cleaned up fold {fold_idx} checkpoint")

        # ── Summary statistics ────────────────────────────────────────────────
        valid = [r for r in fold_results if r["f1"] is not None]
        summary: Dict = {}

        if valid:
            for metric in ("f1", "accuracy", "precision", "recall"):
                vals = [r[metric] for r in valid]
                summary[metric] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "std":  round(float(np.std(vals)),  4),
                    "min":  round(float(np.min(vals)),  4),
                    "max":  round(float(np.max(vals)),  4),
                }

            logger.info(
                f"\n{'='*70}\n"
                f"CV SUMMARY  ({len(valid)}/{n_splits} folds successful)\n"
                f"  F1       : {summary['f1']['mean']:.4f} ± {summary['f1']['std']:.4f}"
                f"  (min {summary['f1']['min']:.4f} / max {summary['f1']['max']:.4f})\n"
                f"  Accuracy : {summary['accuracy']['mean']:.4f} ± {summary['accuracy']['std']:.4f}\n"
                f"{'='*70}"
            )

        return {
            "model_name":         model_name,
            "n_splits":           n_splits,
            "n_successful_folds": len(valid),
            "fold_results":       fold_results,
            "summary":            summary,
        }
