"""
Learning rate schedulers for training optimization.
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
from plantdoc.utils.logging import get_logger
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    ReduceLROnPlateau,
    StepLR,
)

logger = get_logger(__name__)


def get_scheduler(
    cfg: Dict[str, Any],
    optimizer: Optimizer,
    num_epochs: Optional[int] = None,
    steps_per_epoch: Optional[int] = None
) -> Optional[Union[LRScheduler, ReduceLROnPlateau]]:
    """
    Create a learning rate scheduler based on the configuration.

    Args:
        cfg: Scheduler configuration dictionary (e.g., from config['scheduler']).
        optimizer: Optimizer to schedule.
        num_epochs: Total number of training epochs (required for some schedulers).
        steps_per_epoch: Number of optimizer steps per epoch (required for some schedulers).

    Returns:
        Configured learning rate scheduler instance, or None if no scheduler is configured.
    """
    if not isinstance(cfg, dict) or not cfg.get("name"):
        logger.info("No learning rate scheduler configured.")
        return None

    scheduler_name = cfg.get("name").lower()
    logger.info(f"Creating '{scheduler_name}' learning rate scheduler.")

    if scheduler_name == "step":
        step_size = cfg.get("step_size", max(1, num_epochs // 3) if num_epochs else 30)
        gamma = cfg.get("gamma", 0.1)
        logger.info(f"  StepLR params: step_size={step_size}, gamma={gamma:.2f}")
        return StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif scheduler_name == "reduce_on_plateau":
        patience = cfg.get("patience", 5)
        factor = cfg.get("factor", 0.1)
        min_lr = cfg.get("min_lr", 1e-6)
        mode = cfg.get("mode", "min")  # Monitor 'min' (loss) or 'max' (accuracy)
        logger.info(f"  ReduceLROnPlateau params: patience={patience}, factor={factor:.2f}, "
                   f"min_lr={min_lr:.2e}, mode='{mode}'")
        return ReduceLROnPlateau(
            optimizer, 
            mode=mode, 
            factor=factor, 
            patience=patience, 
            min_lr=min_lr, 
            verbose=True
        )

    elif scheduler_name == "cosine":
        if num_epochs is None:
            logger.error("Cosine scheduler requires 'num_epochs' parameter")
            return None
            
        eta_min = cfg.get("min_lr", 0.0)
        logger.info(f"  CosineAnnealingLR params: T_max={num_epochs}, min_lr={eta_min:.2e}")
        return CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=eta_min)

    else:
        logger.warning(f"Unsupported scheduler: '{scheduler_name}'")
        return None


def get_scheduler_with_callback(
    cfg: Dict[str, Any], 
    optimizer: Optimizer, 
    num_epochs: Optional[int] = None, 
    steps_per_epoch: Optional[int] = None
) -> Tuple[Optional[Union[LRScheduler, ReduceLROnPlateau]], Optional[Any]]:
    """
    Create a learning rate scheduler and its corresponding callback.

    Args:
        cfg: Scheduler configuration dictionary (e.g., from config['scheduler']).
        optimizer: Optimizer to schedule.
        num_epochs: Total number of training epochs (required for some schedulers).
        steps_per_epoch: Number of steps per epoch (required for some schedulers).

    Returns:
        Tuple of (scheduler_instance, scheduler_callback_instance) or (None, None).
    """
    # Import here to prevent circular imports
    from plantdoc.core.training.callbacks.lr_scheduler import (
        LearningRateSchedulerCallback,
    )
    
    scheduler = get_scheduler(cfg, optimizer, num_epochs, steps_per_epoch)
    if scheduler is None:
        return None, None

    # Determine the correct step_mode for the callback
    scheduler_name = cfg.get("name", "").lower()
    if scheduler_name == "reduce_on_plateau":
        step_mode = "epoch"  # Plateau MUST step per epoch
    else:
        step_mode = cfg.get("step_mode", "epoch")  # Default to epoch otherwise

    # Monitor metric for ReduceLROnPlateau
    monitor = cfg.get("monitor", "val_loss")
    # Logging changes
    log_changes = cfg.get("log_changes", True)

    logger.info(f"  Creating LearningRateSchedulerCallback with mode='{step_mode}', monitor='{monitor}'")
    callback = LearningRateSchedulerCallback(
        scheduler=scheduler,
        monitor=monitor,
        mode=step_mode,
        log_changes=log_changes
    )

    return scheduler, callback