"""
Utilities for working with callbacks.
"""

import os
from typing import Dict, List, Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf

from core.training.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateSchedulerCallback,
    MetricsLogger,
    ModelCheckpoint,
)
from utils import ensure_dir, get_logger

logger = get_logger(__name__)


def get_callbacks(
    config: DictConfig,
    scheduler=None,
    experiment_dir: Optional[str] = None,
) -> List[Callback]:
    """
    Create a list of callbacks based on the configuration.

    Args:
        config: Configuration with callback settings
        scheduler: Learning rate scheduler to attach to callback
        experiment_dir: Experiment directory for saving outputs

    Returns:
        List of instantiated callbacks
    """
    callbacks = []

    # Check if we have callbacks config
    if not hasattr(config, "callbacks") or not config.callbacks:
        logger.warning("No callbacks configuration found, using defaults")
        return callbacks

    callback_config = config.callbacks

    # Add metrics logger callback
    if (
        hasattr(callback_config, "metrics_logger")
        and callback_config.metrics_logger.enabled
    ):
        logger.info("Adding MetricsLogger callback")
        metrics_logger = MetricsLogger(
            log_path=os.path.join(experiment_dir, "logs") if experiment_dir else None
        )
        callbacks.append(metrics_logger)

    # Add model checkpoint callback
    if (
        hasattr(callback_config, "model_checkpoint")
        and callback_config.model_checkpoint.enabled
    ):
        logger.info("Adding ModelCheckpoint callback")
        checkpoint_config = callback_config.model_checkpoint

        checkpoint_dir = (
            os.path.join(experiment_dir, "checkpoints") if experiment_dir else None
        )
        ensure_dir(checkpoint_dir)

        checkpoint_callback = ModelCheckpoint(
            checkpoint_dir=checkpoint_dir,
            monitor=checkpoint_config.get("monitor", "val_loss"),
            mode=checkpoint_config.get("mode", "min"),
            save_best_only=checkpoint_config.get("save_best_only", True),
            save_weights_only=checkpoint_config.get("save_weights_only", True),
            save_freq=checkpoint_config.get("save_freq", "epoch"),
            max_save=checkpoint_config.get("max_save", 3),
            filename=checkpoint_config.get(
                "filename", "model_{epoch:02d}_{val_loss:.4f}"
            ),
        )
        callbacks.append(checkpoint_callback)

    # Add early stopping callback
    if (
        hasattr(callback_config, "early_stopping")
        and callback_config.early_stopping.enabled
    ):
        logger.info("Adding EarlyStopping callback")
        early_config = callback_config.early_stopping
        early_stopping = EarlyStopping(
            monitor=early_config.get("monitor", "val_loss"),
            min_delta=early_config.get("min_delta", 0.0),
            patience=early_config.get("patience", 10),
            mode=early_config.get("mode", "min"),
            verbose=early_config.get("verbose", True),
        )
        callbacks.append(early_stopping)

    # Add learning rate scheduler callback
    if (
        scheduler is not None
        and hasattr(callback_config, "lr_scheduler")
        and callback_config.lr_scheduler.enabled
    ):
        logger.info("Adding LearningRateScheduler callback")
        lr_config = callback_config.lr_scheduler
        lr_callback = LearningRateSchedulerCallback(
            scheduler=scheduler,
            monitor=lr_config.get("monitor", "val_loss"),
            mode=lr_config.get("mode", "min"),
            min_lr=lr_config.get("min_lr", 1e-8),
            verbose=lr_config.get("verbose", True),
        )
        callbacks.append(lr_callback)

    return callbacks
