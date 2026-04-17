"""
Data Quality Checker  (Pillar 1)
==================================
Audits the dataset BEFORE training starts, catching problems that
could silently inflate reported accuracy or make the model useless
in production.

Checks performed
----------------
1. Very short texts    — texts with < 5 characters after stripping
                         (often empty rows that slipped past NaN checks)
2. Near-duplicate texts — identical content after normalisation
                         (lowercase + remove non-alphanumeric chars)
                         exact duplicates are already removed by DataValidator;
                         these are near-dupes like "Great!" vs "great"
3. Label inconsistency — same normalised text labelled differently
                         (classic label-noise signal)
4. Tiny classes        — any class with fewer than 20 samples
                         (near-impossible to learn from, very high overfit risk)

Output: `data_quality_report.txt` in the experiment folder.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase + strip non-alphanumeric characters for near-dupe detection."""
    return re.sub(r"[^a-z0-9 ]", "", text.lower().strip())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_data_quality(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    experiment_dir: Path,
) -> Dict:
    """
    Run all quality checks, log warnings, and save `data_quality_report.txt`.

    Args:
        df             : The validated DataFrame (post DataValidator).
        text_column    : Name of the text column.
        label_column   : Name of the label column.
        experiment_dir : Path to the current experiment folder.

    Returns:
        Dict with check results (used by PipelineTracker for state info).
    """
    results: Dict = {}
    warnings: List[str] = []

    texts = df[text_column].astype(str)
    labels = df[label_column].astype(str)

    # ------------------------------------------------------------------
    # 1. Very short texts
    # ------------------------------------------------------------------
    short_mask = texts.str.strip().str.len() < 5
    short_count = int(short_mask.sum())
    results["short_texts"] = short_count
    if short_count > 0:
        pct = short_count / len(df) * 100
        warnings.append(
            f"⚠️  {short_count} texts ({pct:.1f}%) are very short (< 5 chars) — "
            f"likely empty or near-empty rows. They may confuse the model."
        )

    # ------------------------------------------------------------------
    # 2. Near-duplicate texts (same normalised content)
    # ------------------------------------------------------------------
    normalized = texts.apply(_normalize)
    near_dup_mask = normalized.duplicated(keep=False)
    near_dup_count = int(near_dup_mask.sum())
    results["near_duplicates"] = near_dup_count
    if near_dup_count > 0:
        pct = near_dup_count / len(df) * 100
        warnings.append(
            f"⚠️  {near_dup_count} texts ({pct:.1f}%) are near-duplicates after "
            f"normalisation (same content, different punctuation/casing). "
            f"These may leak between train/val splits causing artificially high accuracy."
        )

    # ------------------------------------------------------------------
    # 3. Label inconsistency (same text → multiple labels)
    # ------------------------------------------------------------------
    text_label_df = pd.DataFrame({"norm_text": normalized, "label": labels})
    labels_per_text = text_label_df.groupby("norm_text")["label"].nunique()
    inconsistent = int((labels_per_text > 1).sum())
    results["label_inconsistencies"] = inconsistent
    if inconsistent > 0:
        warnings.append(
            f"⚠️  {inconsistent} unique normalised text(s) are assigned to more than "
            f"one label — classic label noise. "
            f"Review your annotation process or source data."
        )

    # ------------------------------------------------------------------
    # 4. Tiny classes (< 20 samples)
    # ------------------------------------------------------------------
    class_counts = df[label_column].value_counts()
    tiny_classes = class_counts[class_counts < 20].to_dict()
    results["tiny_classes"] = tiny_classes
    for cls, cnt in tiny_classes.items():
        warnings.append(
            f"⚠️  Class '{cls}' has only {cnt} sample(s). "
            f"The model will almost certainly overfit or ignore this class."
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    results["total_warnings"] = len(warnings)
    results["passed"] = len(warnings) == 0

    _save_report(df, text_column, label_column, class_counts, results, warnings, experiment_dir)
    return results


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _save_report(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    class_counts: pd.Series,
    results: Dict,
    warnings: List[str],
    experiment_dir: Path,
) -> None:
    path = experiment_dir / "data_quality_report.txt"
    try:
        max_count = class_counts.max()
        with open(path, "w", encoding="utf-8") as f:
            f.write("=" * 62 + "\n")
            f.write("DATA QUALITY REPORT\n")
            f.write("=" * 62 + "\n\n")

            # Overview
            f.write("DATASET OVERVIEW\n")
            f.write("-" * 62 + "\n")
            f.write(f"  Total samples         : {len(df)}\n")
            f.write(f"  Text column           : {text_column}\n")
            f.write(f"  Label column          : {label_column}\n")
            f.write(f"  Number of classes     : {df[label_column].nunique()}\n\n")

            # Class distribution bar chart
            f.write("CLASS DISTRIBUTION\n")
            f.write("-" * 62 + "\n")
            for cls, cnt in class_counts.items():
                bar_len = int(cnt / max_count * 30) if max_count > 0 else 0
                bar = "█" * bar_len
                f.write(f"  {str(cls):<22} {cnt:>6}  {bar}\n")
            f.write("\n")

            # Check results
            f.write("QUALITY CHECKS\n")
            f.write("-" * 62 + "\n")
            f.write(f"  Very short texts (< 5 chars)   : {results['short_texts']}\n")
            f.write(f"  Near-duplicate texts            : {results['near_duplicates']}\n")
            f.write(f"  Label inconsistencies           : {results['label_inconsistencies']}\n")
            tiny = results.get("tiny_classes", {})
            if tiny:
                f.write(f"  Tiny classes (< 20 samples)    : {list(tiny.keys())}\n")
            else:
                f.write(f"  Tiny classes (< 20 samples)    : None\n")
            f.write("\n")

            # Warnings
            if warnings:
                f.write("WARNINGS\n")
                f.write("-" * 62 + "\n")
                for w in warnings:
                    f.write(f"  {w}\n")
                f.write("\n")
            else:
                f.write("✅ No data quality issues detected.\n\n")

            # Final verdict
            status = (
                "✅ PASSED — no issues found."
                if results["passed"]
                else f"⚠️  {results['total_warnings']} WARNING(S) — review before trusting results."
            )
            f.write(f"Overall Status : {status}\n")

        logger.info(f"[Trust/DataQuality] Report saved → {path}")
    except Exception as exc:
        logger.warning(f"[Trust/DataQuality] Could not save report: {exc}")
