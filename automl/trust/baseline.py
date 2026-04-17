"""
Baseline Evaluator  (Pillar 4)
================================
Computes a naive "majority-class" dummy classifier and compares it
against the trained model.  This answers the critical question:

    "Did the model actually *learn* anything, or is it just
     predicting the most common class every time?"

Output: `baseline_comparison.txt` in the experiment folder.

The dummy classifier (sklearn DummyClassifier, strategy="most_frequent")
always predicts whichever label appears most often in the validation set.
Its accuracy equals the majority-class ratio; its weighted F1 is typically
much lower on imbalanced data — making any real model look reasonable.

A model that barely beats the dummy is a red flag worth surfacing.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def compute_majority_baseline(labels: List[int]) -> Dict:
    """
    Compute accuracy and weighted F1 for a majority-class dummy classifier.

    Args:
        labels: Encoded integer labels from the validation split.

    Returns:
        Dict with keys: strategy, accuracy, f1_score, majority_class_ratio.
    """
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import accuracy_score, f1_score

    labels_arr = np.array(labels)
    dummy = DummyClassifier(strategy="most_frequent", random_state=42)
    dummy.fit(labels_arr.reshape(-1, 1), labels_arr)
    preds = dummy.predict(labels_arr.reshape(-1, 1))

    return {
        "strategy": "majority_class",
        "accuracy": float(accuracy_score(labels_arr, preds)),
        "f1_score": float(
            f1_score(labels_arr, preds, average="weighted", zero_division=0)
        ),
        "majority_class_ratio": float(np.bincount(labels_arr).max() / len(labels_arr)),
    }


def save_baseline_comparison(
    baseline: Dict,
    model_result: Dict,
    experiment_dir: Path,
    label_encoder=None,
) -> None:
    """
    Write a human-readable `baseline_comparison.txt` to the experiment folder.

    Args:
        baseline      : Output of compute_majority_baseline().
        model_result  : Best-model evaluation dict from ModelEvaluator.
        experiment_dir: Path to the current experiment folder.
        label_encoder : sklearn LabelEncoder (to decode majority class name).
    """
    # Resolve majority class name
    majority_class_name = None
    true_labels = model_result.get("true_labels")
    if label_encoder is not None and true_labels is not None:
        try:
            counts = np.bincount(np.array(true_labels))
            majority_idx = int(np.argmax(counts))
            majority_class_name = label_encoder.inverse_transform([majority_idx])[0]
        except Exception:
            pass

    model_f1 = model_result["f1_score"]
    model_acc = model_result["accuracy"]
    baseline_f1 = baseline["f1_score"]
    baseline_acc = baseline["accuracy"]

    f1_lift = model_f1 - baseline_f1
    acc_lift = model_acc - baseline_acc
    f1_lift_pct = (f1_lift / baseline_f1 * 100) if baseline_f1 > 0 else float("inf")

    # Verdict
    if f1_lift < 0.02:
        verdict = (
            "⚠️  WARNING: The model barely outperforms a trivial dummy classifier.\n"
            "         This strongly suggests a data, label, or training problem.\n"
            "         Do NOT deploy this model."
        )
    elif f1_lift < 0.10:
        verdict = (
            "🟡 MARGINAL: Modest improvement over baseline. The model learned\n"
            "         something, but the gap is small. Consider more data,\n"
            "         better features, or longer training."
        )
    else:
        verdict = (
            "✅ SOLID: The model shows meaningful improvement over the baseline.\n"
            "         It is genuinely learning patterns in the data."
        )

    path = experiment_dir / "baseline_comparison.txt"
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("=" * 62 + "\n")
            f.write("BASELINE vs TRAINED MODEL\n")
            f.write("=" * 62 + "\n\n")

            f.write("DUMMY BASELINE  (always predicts the majority class)\n")
            f.write("-" * 62 + "\n")
            if majority_class_name is not None:
                f.write(f"  Majority class        : {majority_class_name}\n")
            f.write(
                f"  Majority class ratio  : {baseline['majority_class_ratio']:.1%}  "
                f"(this is the baseline accuracy)\n"
            )
            f.write(f"  Accuracy              : {baseline_acc:.4f}\n")
            f.write(f"  F1 Score (weighted)   : {baseline_f1:.4f}\n\n")

            f.write("TRAINED MODEL\n")
            f.write("-" * 62 + "\n")
            f.write(f"  Model                 : {model_result.get('model_name', 'Unknown')}\n")
            f.write(f"  Accuracy              : {model_acc:.4f}\n")
            f.write(f"  F1 Score (weighted)   : {model_f1:.4f}\n\n")

            f.write("IMPROVEMENT OVER BASELINE\n")
            f.write("-" * 62 + "\n")
            f.write(f"  F1  lift              : {f1_lift:+.4f}  ({f1_lift_pct:+.1f}%)\n")
            f.write(f"  Acc lift              : {acc_lift:+.4f}\n\n")

            f.write("VERDICT\n")
            f.write("-" * 62 + "\n")
            f.write(f"  {verdict}\n")

        logger.info(f"[Trust/Baseline] Baseline comparison saved → {path}")
    except Exception as exc:
        logger.warning(f"[Trust/Baseline] Could not save comparison: {exc}")
