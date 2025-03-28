"""
Training callbacks for PlantDoc.

Provides basic callbacks for early stopping, model checkpointing,
learning rate scheduling, and metrics logging.
"""

from .base import Callback
from .early_stopping import EarlyStopping
from .model_checkpoint import ModelCheckpoint
from .lr_scheduler import LearningRateSchedulerCallback
from .metrics_logger import MetricsLogger

__all__ = [
    "Callback",
    "EarlyStopping",
    "ModelCheckpoint",
    "LearningRateSchedulerCallback",
    "MetricsLogger",
]