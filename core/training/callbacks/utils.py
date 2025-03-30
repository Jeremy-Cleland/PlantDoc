"""
Utilities for working with callbacks.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf

from core.training.callbacks import (
    Callback,
    EarlyStopping,
    GradCAMCallback,
    LearningRateSchedulerCallback,
    MetricsLogger,
    ModelCheckpoint,
)
from utils.logger import get_logger
from utils.paths import ensure_dir

logger = get_logger(__name__)


def get_callbacks(
    config: DictConfig,
    scheduler=None,
    experiment_dir: Optional[str] = None,
    test_data=None,
    skip_gradcam=False,
) -> List[Callback]:
    """
    Create a list of callbacks based on the configuration.

    Args:
        config: Configuration with callback settings
        scheduler: Learning rate scheduler to attach to callback
        experiment_dir: Experiment directory for saving outputs
        test_data: Optional test data to pass to the GradCAM callback
        skip_gradcam: If True, skip creating GradCAM callback

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

        # Get metrics directory from config paths if available
        if hasattr(config.paths, "metrics_dir"):
            metrics_dir = config.paths.metrics_dir
            logger.info(f"Using metrics_dir from config: {metrics_dir}")
        elif experiment_dir:
            metrics_dir = os.path.join(experiment_dir, "metrics")
            logger.info(f"Using metrics_dir from experiment_dir: {metrics_dir}")
        else:
            metrics_dir = os.path.join("outputs", "default", "metrics")
            logger.warning(f"Using fallback metrics_dir: {metrics_dir}")

        ensure_dir(metrics_dir)

        metrics_logger = MetricsLogger(metrics_dir=metrics_dir)
        callbacks.append(metrics_logger)

    # Add model checkpoint callback
    if (
        hasattr(callback_config, "model_checkpoint")
        and callback_config.model_checkpoint.enabled
    ):
        logger.info("Adding ModelCheckpoint callback")
        checkpoint_config = callback_config.model_checkpoint

        # Use checkpoint directory from config or derive from experiment_dir
        if hasattr(config.paths, "checkpoint_dir"):
            checkpoint_dir = config.paths.checkpoint_dir
            logger.info(f"Using checkpoint_dir from config: {checkpoint_dir}")
        elif experiment_dir:
            checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
            logger.info(f"Using checkpoint_dir from experiment_dir: {checkpoint_dir}")
        else:
            checkpoint_dir = os.path.join("outputs", "default", "checkpoints")
            logger.warning(f"Using fallback checkpoint_dir: {checkpoint_dir}")

        ensure_dir(checkpoint_dir)

        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor=checkpoint_config.get("monitor", "val_loss"),
            mode=checkpoint_config.get("mode", "min"),
            save_best_only=checkpoint_config.get("save_best_only", True),
            save_optimizer=checkpoint_config.get("save_optimizer", True),
            save_freq=checkpoint_config.get("save_freq", "epoch"),
            filename=checkpoint_config.get(
                "filename", "model_{epoch:02d}_{val_loss:.4f}"
            ),
            best_filename=checkpoint_config.get("best_filename", "best_model.pth"),
            last_filename=checkpoint_config.get("last_filename", "last_model.pth"),
            save_last=checkpoint_config.get("save_last", True),
            verbose=checkpoint_config.get("verbose", True),
        )
        callbacks.append(checkpoint_callback)

    # Add early stopping callback
    if (
        hasattr(callback_config, "early_stopping")
        and callback_config.early_stopping.enabled
    ):
        logger.info("Adding EarlyStopping callback")
        early_config = callback_config.early_stopping

        # Set default values if needed
        monitor = early_config.get("monitor", "val_loss")
        min_delta = early_config.get("min_delta", 0.0)
        patience = early_config.get("patience", 10)
        mode = early_config.get("mode", "min")
        verbose = early_config.get("verbose", True)

        early_stopping = EarlyStopping(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            mode=mode,
            verbose=verbose,
        )
        callbacks.append(early_stopping)

    # Add learning rate scheduler callback
    if (
        scheduler is not None
        and hasattr(callback_config, "lr_scheduler")
        and callback_config.lr_scheduler.enabled
        and not isinstance(scheduler, (str, Path))  # Ensure scheduler is not a path
    ):
        logger.info("Adding LearningRateScheduler callback")
        lr_config = callback_config.lr_scheduler

        # Set default values if needed
        monitor = lr_config.get("monitor", "val_loss")
        mode = lr_config.get("mode", "epoch")
        log_changes = lr_config.get("verbose", True)

        lr_callback = LearningRateSchedulerCallback(
            scheduler=scheduler,
            monitor=monitor,
            mode=mode,
            log_changes=log_changes,
        )
        callbacks.append(lr_callback)
    elif (
        hasattr(callback_config, "lr_scheduler")
        and callback_config.lr_scheduler.enabled
    ):
        logger.warning(
            "LR scheduler callback enabled but no valid scheduler provided - skipping"
        )

    # Add GradCAM visualization callback
    if (
        not skip_gradcam
        and hasattr(callback_config, "gradcam")
        and callback_config.gradcam.enabled
    ):
        logger.info("Adding GradCAM visualization callback")
        gradcam_config = callback_config.gradcam

        # Use gradcam directory from config or derive from experiment_dir
        if hasattr(config.paths, "gradcam_dir"):
            gradcam_dir = config.paths.gradcam_dir
            logger.info(f"Using gradcam_dir from config: {gradcam_dir}")
        elif experiment_dir:
            gradcam_dir = os.path.join(experiment_dir, "gradcam")
            logger.info(f"Using gradcam_dir from experiment_dir: {gradcam_dir}")
        else:
            gradcam_dir = os.path.join("outputs", "default", "gradcam")
            logger.warning(f"Using fallback gradcam_dir: {gradcam_dir}")

        ensure_dir(gradcam_dir)

        # Get test_data from function parameter first, then from config
        # Use test_data parameter if provided, otherwise try the config
        current_test_data = (
            test_data
            if test_data is not None
            else gradcam_config.get("test_data", None)
        )

        # Get class names from config or data module
        class_names = gradcam_config.get("class_names", None)
        if class_names is None and hasattr(config.data, "class_names"):
            class_names = config.data.class_names

        # Create the GradCAM callback if we have test data and class names
        if current_test_data is not None and class_names is not None:
            gradcam_callback = GradCAMCallback(
                gradcam_dir=gradcam_dir,
                test_data=current_test_data,
                class_names=class_names,
                sample_indices=gradcam_config.get("sample_indices", None),
                frequency=gradcam_config.get("frequency", 20),
                n_samples=gradcam_config.get("n_samples", 5),
                target_layer=gradcam_config.get("target_layer", None),
                input_size=gradcam_config.get("input_size", (224, 224)),
                mean=gradcam_config.get("mean", [0.485, 0.456, 0.406]),
                std=gradcam_config.get("std", [0.229, 0.224, 0.225]),
            )
            callbacks.append(gradcam_callback)
        else:
            logger.warning(
                "GradCAM callback enabled but missing test_data or class_names"
            )

    return callbacks
