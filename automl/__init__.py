"""
AutoML Package
Automated Machine Learning for Text Classification
"""

from automl.pipeline import AutoMLPipeline
from automl.data_validator import DataValidator
from automl.data_intelligence import DataIntelligence
from automl.model_trainer import ModelTrainer
from automl.evaluator import ModelEvaluator

__version__ = "0.1.0"

__all__ = [
    'AutoMLPipeline',
    'DataValidator',
    'DataIntelligence',
    'ModelTrainer',
    'ModelEvaluator',
]
