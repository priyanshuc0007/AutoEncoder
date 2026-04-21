"""
Decisions Logger  (Pillar 2)
==============================
Captures every automated decision made by DataIntelligence and saves
it as `decisions_log.json` in the experiment folder.

For each decision this records:
  - category       : what kind of decision (model_selection, hyperparameters, …)
  - decision       : plain-English statement of what was chosen
  - reason         : data-driven explanation WHY (numbers, thresholds)
  - value          : the raw value for programmatic re-use
  - alternatives_not_chosen : what was available but rejected

This makes the pipeline fully auditable — a user can open the JSON
and understand exactly why DistilBERT was chosen over BERT, why class
weights were enabled, or why 15 epochs were used.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DecisionsLogger:
    """Records every automated pipeline decision with its reasoning."""

    def __init__(self):
        self._decisions: List[Dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(
        self,
        category: str,
        decision: str,
        reason: str,
        value: Any = None,
        alternatives: Optional[List[str]] = None,
    ) -> None:
        """
        Record a single decision.

        Args:
            category    : Grouping label (e.g. "model_selection").
            decision    : Human-readable statement of what was chosen.
            reason      : Data-driven explanation of why.
            value       : Raw chosen value (for programmatic inspection).
            alternatives: Other options that existed but were not chosen.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "category": category,
            "decision": decision,
            "reason": reason,
            "value": value,
            "alternatives_not_chosen": alternatives or [],
        }
        self._decisions.append(entry)
        logger.info(f"[Trust/Decisions] {category}: {decision}")

    def log_from_analysis(self, analysis: Dict) -> None:
        """
        Auto-extract and log all decisions embedded in a
        DataIntelligence analysis dictionary.

        Args:
            analysis: dict returned by DataIntelligence.analyze()
        """
        task = analysis.get("task_info", {})
        imbalance = analysis.get("imbalance_info", {})
        text_info = analysis.get("text_info", {})
        config = analysis.get("training_config", {})
        models = analysis.get("model_selection", [])

        # 1. Task type -------------------------------------------------
        num_classes = task.get("num_classes", "?")
        task_type = task.get("task_type", "unknown")
        self.log(
            category="task_detection",
            decision=f"Task classified as '{task_type}'",
            reason=(
                f"Dataset has {num_classes} unique label value(s). "
                f"Binary = 2 classes, multiclass = 3+."
            ),
            value=task_type,
            alternatives=["binary", "multiclass"] if task_type == "binary" else ["binary"],
        )

        # 2. Class imbalance -------------------------------------------
        use_weights = imbalance.get("use_class_weights", False)
        ratio = imbalance.get("imbalance_ratio", 1.0)
        minority_ratio = imbalance.get("minority_class_ratio", 0.0)
        self.log(
            category="class_imbalance",
            decision=f"Class weights {'ENABLED' if use_weights else 'DISABLED'}",
            reason=(
                f"Imbalance ratio = {ratio:.2f}x "
                f"(minority class = {minority_ratio:.1%} of data). "
                f"Threshold to enable class weights: ratio > 2.0."
            ),
            value=use_weights,
            alternatives=[] if use_weights else ["Enable class weights if ratio > 2.0"],
        )

        # 3. Focal loss note -------------------------------------------
        use_focal = imbalance.get("use_focal_loss", False)
        if use_focal:
            self.log(
                category="class_imbalance",
                decision="Focal loss recommended (ratio > 5.0)",
                reason=(
                    f"Imbalance ratio = {ratio:.2f}x exceeds the 5.0 threshold. "
                    f"Focal loss down-weights easy examples to focus on hard ones."
                ),
                value=True,
            )

        # 4. Sequence length -------------------------------------------
        p95 = text_info.get("p95_length", "?")
        avg = text_info.get("avg_length", "?")
        measurement = text_info.get("measurement", "char_div4_approx")
        tokenizer_used = text_info.get("tokenizer_used", None)
        p95_raw = text_info.get("p95_tokens_raw", p95)
        if measurement == "real_tokens":
            method_note = (
                f"Real token counts measured with {tokenizer_used} on up to 500 sampled texts. "
                f"Raw p95 = {p95_raw} tokens (capped at 512)."
            )
        else:
            method_note = "Estimated via char÷4 approximation (tokenizer unavailable at analysis time)."
        self.log(
            category="sequence_length",
            decision=f"Max token length set to {p95}",
            reason=(
                f"{method_note} "
                f"Average text length = {avg:.0f} tokens." if isinstance(avg, float) else
                f"{method_note} Average text length = {avg}."
            ),
            value=p95,
        )

        # 5. Model selection -------------------------------------------
        strategy = config.get("strategy", "?")
        samples_per_class = config.get("samples_per_class", 0)
        all_candidates = [
            "prajjwal1/bert-tiny",
            "prajjwal1/bert-mini",
            "google/mobilebert-uncased",
            "distilbert-base-uncased",
            "bert-base-uncased",
        ]
        not_chosen = [m for m in all_candidates if m not in models]
        self.log(
            category="model_selection",
            decision=f"Selected model(s): {', '.join(models)}",
            reason=(
                f"Strategy = '{strategy}' based on {samples_per_class:.0f} samples/class. "
                f"Rule: < 2 000 samples → bert-tiny + bert-mini + mobilebert; "
                f"2 000–10 000 → bert-mini + mobilebert + distilbert + bert; "
                f"> 10 000 → mobilebert + distilbert + bert."
            ) if isinstance(samples_per_class, (int, float)) else
            f"Strategy = '{strategy}'. Models: {models}.",
            value=models,
            alternatives=not_chosen if not_chosen else ["(all candidates selected)"],
        )

        # 6. Hyperparameter strategy -----------------------------------
        self.log(
            category="hyperparameters",
            decision=f"Hyperparameter strategy: '{strategy}'",
            reason=(
                f"samples/class={samples_per_class:.0f}, imbalance_ratio={ratio:.2f}. "
                f"Chosen epochs={config.get('num_epochs_range')}, "
                f"lr={config.get('learning_rate')}, "
                f"batch_size={config.get('batch_size')}, "
                f"weight_decay={config.get('weight_decay')}, "
                f"early_stopping_patience={config.get('early_stopping_patience')}."
            ) if isinstance(samples_per_class, (int, float)) else "See training_config.",
            value={
                "strategy": strategy,
                "epochs_range": config.get("num_epochs_range"),
                "learning_rate": config.get("learning_rate"),
                "batch_size": config.get("batch_size"),
                "weight_decay": config.get("weight_decay"),
                "gradient_accumulation_steps": config.get("gradient_accumulation_steps"),
                "warmup_steps": config.get("warmup_steps"),
                "early_stopping_patience": config.get("early_stopping_patience"),
            },
        )

    def save(self, experiment_dir: Path) -> None:
        """
        Write decisions_log.json to the experiment directory.

        Args:
            experiment_dir: Path to the current experiment folder.
        """
        path = experiment_dir / "decisions_log.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "total_decisions": len(self._decisions),
                        "note": (
                            "Every entry below is an automated decision made by the "
                            "pipeline, with the data-driven reason that triggered it."
                        ),
                        "decisions": self._decisions,
                    },
                    f,
                    indent=2,
                    default=str,
                )
            logger.info(f"[Trust/Decisions] Saved {len(self._decisions)} decisions → {path}")
        except Exception as exc:
            logger.warning(f"[Trust/Decisions] Could not save decisions log: {exc}")
