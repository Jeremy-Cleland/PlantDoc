"""
Utilities for working with callbacks.
"""

import os
from pathlib import Path
from typing import List, Optional

from omegaconf import DictConfig

from core.training.callbacks import (
    AdaptiveWeightAdjustmentCallback,
    Callback,
    ConfidenceMonitorCallback,
    EarlyStopping,
    GradCAMCallback,
    LearningRateSchedulerCallback,
    MetricsLogger,
    ModelCheckpoint,
    SWACallback,
    VisualizationDataSaver,
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

        # Get class names from config if available
        class_names = None
        if hasattr(config.data, "class_names"):
            class_names = config.data.class_names

        metrics_logger = MetricsLogger(
            metrics_dir=metrics_dir,
            experiment_dir=experiment_dir,  # Pass experiment_dir
            class_names=class_names,  # Pass class_names
        )
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

    # Add confidence monitoring callback
    if (
        hasattr(callback_config, "confidence_monitor")
        and callback_config.confidence_monitor.enabled
    ):
        logger.info("Adding ConfidenceMonitor callback")
        conf_config = callback_config.confidence_monitor

        # Create the callback with settings from config
        confidence_monitor = ConfidenceMonitorCallback(
            monitor_frequency=conf_config.get("monitor_frequency", 1),
            threshold_warning=conf_config.get("threshold_warning", 0.7),
            ece_bins=conf_config.get("ece_bins", 10),
            log_per_class=conf_config.get("log_per_class", True),
        )
        callbacks.append(confidence_monitor)

    # Add Stochastic Weight Averaging callback
    if hasattr(callback_config, "swa") and callback_config.swa.enabled:
        logger.info("Adding SWA callback")
        swa_config = callback_config.swa

        # Create the callback with settings from config
        swa_callback = SWACallback(
            swa_start_frac=swa_config.get("swa_start_frac", 0.75),
            swa_lr=swa_config.get("swa_lr", 0.05),
            anneal_epochs=swa_config.get("anneal_epochs", 10),
            anneal_strategy=swa_config.get("anneal_strategy", "cos"),
            update_bn_epochs=swa_config.get("update_bn_epochs", 5),
        )
        callbacks.append(swa_callback)

    # Add Adaptive Weight Adjustment callback
    if (
        hasattr(callback_config, "adaptive_weight")
        and callback_config.adaptive_weight.enabled
    ):
        logger.info("Adding AdaptiveWeightAdjustment callback")
        adapt_config = callback_config.adaptive_weight

        # Create the callback with settings from config
        adaptive_callback = AdaptiveWeightAdjustmentCallback(
            focal_loss_gamma_range=adapt_config.get(
                "focal_loss_gamma_range", (2.0, 5.0)
            ),
            center_loss_weight_range=adapt_config.get(
                "center_loss_weight_range", (0.5, 2.0)
            ),
            adjust_frequency=adapt_config.get("adjust_frequency", 5),
        )
        callbacks.append(adaptive_callback)

    # Add Visualization Data Saver callback
    if (
        hasattr(callback_config, "visualization_data_saver")
        and callback_config.visualization_data_saver.enabled
    ):
        logger.info("Adding VisualizationDataSaver callback")
        vis_config = callback_config.visualization_data_saver

        # Create the callback with settings from config
        visualization_data_saver = VisualizationDataSaver(
            experiment_dir=experiment_dir,
            num_test_images=vis_config.get("num_test_images", 20),
            model_type=vis_config.get("model_type", "resnet"),
            save_augmentation_examples=vis_config.get(
                "save_augmentation_examples", True
            ),
        )
        callbacks.append(visualization_data_saver)
    else:
        # Always add the visualization data saver by default, if not explicitly disabled
        # This ensures we always generate the data files needed for enhanced visualizations
        if experiment_dir:
            logger.info("Adding default VisualizationDataSaver callback")
            visualization_data_saver = VisualizationDataSaver(
                experiment_dir=experiment_dir,
                num_test_images=20,
                model_type=(
                    getattr(config.model, "name", "resnet").lower()
                    if hasattr(config, "model")
                    else "resnet"
                ),
                save_augmentation_examples=True,
            )
            callbacks.append(visualization_data_saver)

    return callbacks
