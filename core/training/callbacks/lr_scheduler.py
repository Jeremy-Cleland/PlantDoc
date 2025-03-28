"""
Learning rate scheduler callback for adjusting learning rates during training.
"""

from typing import Any, Dict, Optional, Union

import torch
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from utils.logger import get_logger

from .base import Callback

logger = get_logger(__name__)


class LearningRateSchedulerCallback(Callback):
    """
    Callback for updating learning rate during training.

    Args:
        scheduler: Learning rate scheduler instance
        monitor: Metric to monitor (only used for ReduceLROnPlateau)
        mode: Either 'epoch' (step per epoch) or 'step' (step per batch)
        log_changes: Whether to log learning rate changes
    """

    priority = 85  # Run after model checkpoint (80) but before early stopping (90)

    def __init__(
        self,
        scheduler: Union[LRScheduler, ReduceLROnPlateau],
        monitor: str = "val_loss",
        mode: str = "epoch",
        log_changes: bool = True,
    ):
        super().__init__()
        self.scheduler = scheduler
        self.monitor = monitor
        self.mode = mode
        self.log_changes = log_changes

        # Validate mode
        if self.mode not in ["epoch", "step"]:
            raise ValueError(f"Mode must be 'epoch' or 'step', got {self.mode}")

        # Validate scheduler type for mode
        is_plateau = isinstance(scheduler, ReduceLROnPlateau)
        if is_plateau and self.mode != "epoch":
            logger.warning(
                "ReduceLROnPlateau scheduler can only be used with mode='epoch'. "
                "Forcing mode='epoch'."
            )
            self.mode = "epoch"

        logger.info(
            f"Initialized LearningRateSchedulerCallback: "
            f"scheduler={type(scheduler).__name__}, "
            f"mode='{self.mode}', "
            f"monitor='{self.monitor if is_plateau else 'N/A'}', "
            f"log_changes={self.log_changes}"
        )

        # Store initial learning rates
        self.initial_lrs = [
            group["lr"] for group in self.scheduler.optimizer.param_groups
        ]

    def _log_lr(self, epoch: Optional[int] = None, batch: Optional[int] = None) -> None:
        """Log current learning rates."""
        current_lrs = [group["lr"] for group in self.scheduler.optimizer.param_groups]

        # Determine if LR has changed
        has_changed = any(
            abs(old - new) > 1e-10 for old, new in zip(self.initial_lrs, current_lrs)
        )

        if has_changed or self.log_changes:
            prefix = f"Epoch {epoch + 1}" if epoch is not None else f"Batch {batch + 1}"

            if len(current_lrs) == 1:
                logger.info(f"{prefix}: Learning rate = {current_lrs[0]:.2e}")
            else:
                lr_str = ", ".join([f"{lr:.2e}" for lr in current_lrs])
                logger.info(f"{prefix}: Learning rates = [{lr_str}]")

        # Update initial_lrs to current values
        self.initial_lrs = current_lrs

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Step the scheduler at the end of an epoch."""
        logs = logs or {}

        # Skip if mode is not epoch
        if self.mode != "epoch":
            return

        # Handle ReduceLROnPlateau
        if isinstance(self.scheduler, ReduceLROnPlateau):
            monitor_value = logs.get(self.monitor)
            if monitor_value is None:
                logger.warning(
                    f"LearningRateSchedulerCallback: "
                    f"Monitor '{self.monitor}' not found in logs. "
                    f"Scheduler step skipped."
                )
                return

            # Step with monitored value
            self.scheduler.step(monitor_value)
        else:
            # Step regular scheduler
            self.scheduler.step()

        # Log updated LR
        self._log_lr(epoch=epoch)

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Step the scheduler at the end of a batch."""
        logs = logs or {}

        # Skip if mode is not step or scheduler is ReduceLROnPlateau
        if self.mode != "step" or isinstance(self.scheduler, ReduceLROnPlateau):
            return

        # Step regular scheduler
        self.scheduler.step()

        # Log updated LR (less frequently to avoid log spam)
        if batch % 100 == 0:  # Log every 100 batches
            self._log_lr(batch=batch)
