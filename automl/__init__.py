"""
AutoLLM Package
Automated Fine-Tuning Pipeline for Text Classification
"""

from automl.pipeline import AutoLLMPipeline
from automl.data_validator import DataValidator
from automl.data_intelligence import DataIntelligence
from automl.model_trainer import ModelTrainer
from automl.evaluator import ModelEvaluator

# Backwards-compatible alias
AutoMLPipeline = AutoLLMPipeline

__version__ = "1.0.0"

__all__ = [
    'AutoLLMPipeline',
    'AutoMLPipeline',  # backwards compat
    'DataValidator',
    'DataIntelligence',
    'ModelTrainer',
    'ModelEvaluator',
]
