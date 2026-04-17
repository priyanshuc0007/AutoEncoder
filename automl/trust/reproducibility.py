"""
Reproducibility Module  (Pillar 3)
===================================
Sets every random seed that touches the training stack so that
re-running the same experiment with the same seed always produces
the same model weights, splits, and metrics.

Covers: Python built-in, NumPy, PyTorch (CPU + all CUDA devices),
and the PYTHONHASHSEED environment variable (affects dict ordering
in Python < 3.7 and some hash-based ops).

The seed is also written into experiment_config.json by the
PipelineTracker, so every experiment is fully replayable.
"""

import os
import random
import logging

logger = logging.getLogger(__name__)


def set_global_seeds(seed: int = 42) -> int:
    """
    Set all random seeds for reproducibility.

    Args:
        seed: Integer seed value (default 42).

    Returns:
        The seed that was applied (same value, useful for logging).
    """
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic CUDA ops — may reduce throughput slightly on GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Affects Python hash randomisation (str, bytes, datetime)
    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.info(
        f"[Trust/Reproducibility] Global seed set to {seed} "
        f"(Python, NumPy, PyTorch CPU+CUDA, PYTHONHASHSEED)"
    )
    return seed
