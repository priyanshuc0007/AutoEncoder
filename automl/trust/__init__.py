"""
Trust Layer — non-invasive reliability and transparency modules.

Each module in this package adds transparency, reproducibility, or
audit trail capabilities WITHOUT modifying any training logic.

Modules
-------
data_quality     – Pillar 1: pre-training data audit
decisions_logger – Pillar 2: automated decision audit log
reproducibility  – Pillar 3: global seed management
baseline         – Pillar 4: dummy baseline comparison
explainability   – Pillar 5: token importance + confidence calibration
pipeline_tracker – Pillar 6: live step-by-step state tracking
"""

from automl.trust.reproducibility import set_global_seeds
from automl.trust.pipeline_tracker import PipelineTracker
from automl.trust.decisions_logger import DecisionsLogger
from automl.trust.baseline import compute_majority_baseline, save_baseline_comparison
from automl.trust.data_quality import check_data_quality
from automl.trust.explainability import run_explainability

__all__ = [
    "set_global_seeds",
    "PipelineTracker",
    "DecisionsLogger",
    "compute_majority_baseline",
    "save_baseline_comparison",
    "check_data_quality",
    "run_explainability",
]
