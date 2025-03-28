"""
Core training module for PlantDoc.

Includes the Trainer class, loss functions, optimizers, schedulers,
and basic callbacks.
"""

# Main training components
from .train import Trainer, train_model
from .loss import WeightedCrossEntropyLoss, get_loss_fn
from .optimizers import get_optimizer
from .schedulers import get_scheduler, get_scheduler_with_callback

# Callbacks
from .callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateSchedulerCallback,
    MetricsLogger,
)

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