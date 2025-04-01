"""
Training callbacks for plant disease classification.

Provides basic callbacks for early stopping, model checkpointing,
learning rate scheduling, and metrics logging.
"""

from .adaptive_weight import AdaptiveWeightAdjustmentCallback
from .base import Callback
from .confidence_monitor import ConfidenceMonitorCallback
from .early_stopping import EarlyStopping
from .gradcam_callback import GradCAMCallback
from .lr_scheduler import LearningRateSchedulerCallback
from .metrics_logger import MetricsLogger
from .model_checkpoint import ModelCheckpoint
from .swa import SWACallback

__all__ = [
    "Callback",
    "EarlyStopping",
    "ModelCheckpoint",
    "LearningRateSchedulerCallback",
    "MetricsLogger",
    "GradCAMCallback",
    "ConfidenceMonitorCallback",
    "SWACallback",
    "AdaptiveWeightAdjustmentCallback",
]
