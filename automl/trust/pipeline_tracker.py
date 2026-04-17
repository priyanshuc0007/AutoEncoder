"""
Pipeline Tracker  (Pillar 6)
==============================
Tracks each pipeline step's status and timestamps in real-time,
writing `pipeline_state.json` to the experiment folder after every
state change.  If the pipeline crashes, the file shows exactly which
step failed and what the error was — no guessing required.

Schema of pipeline_state.json
------------------------------
{
  "experiment_name": "20260416_120000",
  "seed": 42,
  "started_at": "2026-04-16T12:00:00",
  "finished_at": "2026-04-16T12:05:00",   # null until done
  "status": "running | completed | failed",
  "steps": {
    "data_validation":      {"status": "completed", "started_at": ..., "finished_at": ..., "error": null},
    "data_intelligence":    {"status": "running",   "started_at": ..., "finished_at": null, "error": null},
    "data_preparation":     {"status": "pending",   ...},
    "model_training":       {"status": "pending",   ...},
    "model_evaluation":     {"status": "pending",   ...},
    "best_model_selection": {"status": "pending",   ...},
    "report_generation":    {"status": "pending",   ...}
  }
}
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Canonical step names in execution order
PIPELINE_STEPS = [
    "data_validation",
    "data_intelligence",
    "data_preparation",
    "model_training",
    "model_evaluation",
    "best_model_selection",
    "report_generation",
]


class PipelineTracker:
    """Tracks pipeline execution state and writes live status to disk."""

    def __init__(self):
        self.experiment_dir: Optional[Path] = None
        self.state: dict = {}
        self._current_step: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, experiment_name: str, experiment_dir: Path, seed: int) -> None:
        """Initialise tracking for a new run."""
        self.experiment_dir = experiment_dir
        self.state = {
            "experiment_name": experiment_name,
            "seed": seed,
            "started_at": datetime.now().isoformat(),
            "finished_at": None,
            "status": "running",
            "steps": {
                step: {
                    "status": "pending",
                    "started_at": None,
                    "finished_at": None,
                    "error": None,
                }
                for step in PIPELINE_STEPS
            },
        }
        self._save()
        logger.info("[Trust/Tracker] Pipeline tracking started")

    def begin_step(self, step: str) -> None:
        """Mark a step as currently running."""
        if step not in self.state.get("steps", {}):
            return
        self._current_step = step
        self.state["steps"][step]["status"] = "running"
        self.state["steps"][step]["started_at"] = datetime.now().isoformat()
        self._save()
        logger.info(f"[Trust/Tracker] ▶  {step}")

    def complete_step(self, step: str) -> None:
        """Mark a step as successfully completed."""
        if step not in self.state.get("steps", {}):
            return
        self.state["steps"][step]["status"] = "completed"
        self.state["steps"][step]["finished_at"] = datetime.now().isoformat()
        # Ensure started_at is populated even if begin_step was not called
        if self.state["steps"][step]["started_at"] is None:
            self.state["steps"][step]["started_at"] = self.state["steps"][step]["finished_at"]
        self._save()
        logger.info(f"[Trust/Tracker] ✅ {step}")

    def fail_step(self, step: str, error: str) -> None:
        """Mark a step as failed with an error message."""
        if step not in self.state.get("steps", {}):
            return
        self.state["steps"][step]["status"] = "failed"
        self.state["steps"][step]["finished_at"] = datetime.now().isoformat()
        self.state["steps"][step]["error"] = str(error)[:500]  # cap length
        self.state["status"] = "failed"
        self._save()
        logger.warning(f"[Trust/Tracker] ❌ {step} — {error}")

    def finish(self) -> None:
        """Mark the whole pipeline as successfully completed."""
        self.state["status"] = "completed"
        self.state["finished_at"] = datetime.now().isoformat()
        self._save()
        logger.info("[Trust/Tracker] Pipeline completed successfully")

    def mark_failed(self, error: str) -> None:
        """Mark the whole pipeline as failed (when step is unknown)."""
        self.state["status"] = "failed"
        self.state["finished_at"] = datetime.now().isoformat()
        # Mark current running step as failed
        for step, info in self.state.get("steps", {}).items():
            if info["status"] == "running":
                info["status"] = "failed"
                info["error"] = str(error)[:500]
                info["finished_at"] = self.state["finished_at"]
                break
        self._save()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _save(self) -> None:
        if self.experiment_dir is None:
            return
        try:
            path = self.experiment_dir / "pipeline_state.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2, default=str)
        except Exception as exc:
            logger.debug(f"[Trust/Tracker] Could not save state: {exc}")
