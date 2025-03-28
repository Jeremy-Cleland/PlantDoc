"""
Core training module for

Includes the Trainer class, loss functions, optimizers, schedulers,
and basic callbacks.
"""

# Main training components
# Callbacks
from .callbacks import (
    Callback,
    EarlyStopping,
    LearningRateSchedulerCallback,
    MetricsLogger,
    ModelCheckpoint,
)
from .loss import WeightedCrossEntropyLoss, get_loss_fn
from .optimizers import get_optimizer
from .schedulers import get_scheduler, get_scheduler_with_callback
from .train import Trainer, train_model

__all__ = [
    # Trainer
    "Trainer",
    "train_model",
    # Loss Functions
    "get_loss_fn",
    "WeightedCrossEntropyLoss",
    # Optimizers
    "get_optimizer",
    # Schedulers
    "get_scheduler",
    "get_scheduler_with_callback",
    # Callbacks
    "Callback",
    "EarlyStopping",
    "ModelCheckpoint",
    "LearningRateSchedulerCallback",
    "MetricsLogger",
]
