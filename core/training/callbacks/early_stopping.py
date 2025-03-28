"""
Early Stopping callback to halt training when a monitored metric stops improving.
"""

import numbers
from typing import Any, Dict, Optional

import torch

from utils.logger import get_logger

from .base import Callback

logger = get_logger(__name__)


class EarlyStopping(Callback):
    """
    Stop training when a monitored metric has stopped improving.

    Args:
        monitor: Quantity to monitor. Default: "val_loss"
        patience: Epochs with no improvement to wait before stopping. Default: 10
        mode: 'min' or 'max'. Default: "min"
        min_delta: Minimum change to qualify as improvement. Default: 0.0
        verbose: Log early stopping actions. Default: True
        restore_best_weights: Whether to restore model to best weights. Default: False
    """

    priority = 90  # Run before most other callbacks

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        mode: str = "min",
        min_delta: float = 0.0,
        verbose: bool = True,
        restore_best_weights: bool = False,
    ):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights

        if self.mode not in ["min", "max"]:
            raise ValueError(f"Mode must be 'min' or 'max', got {self.mode}")

        # For 'min' mode, improvement is decrease. For 'max' mode, improvement is increase.
        if self.mode == "min":
            self.is_better = lambda current, best: current < best - self.min_delta
        else:
            self.is_better = lambda current, best: current > best + self.min_delta

        self.wait = 0
        self.stopped_epoch = 0
        self.best = float("inf") if self.mode == "min" else float("-inf")
        self.stop_training = False
        self.best_weights = None

        if self.verbose:
            logger.info(
                f"Initialized EarlyStopping: monitor='{self.monitor}', "
                f"patience={self.patience}, mode='{self.mode}', "
                f"min_delta={self.min_delta}, "
                f"restore_best={self.restore_best_weights}"
            )

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Reset internal variables at the start of training."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best = float("inf") if self.mode == "min" else float("-inf")
        self.stop_training = False
        self.best_weights = None

        logger.debug("EarlyStopping state reset.")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Check for improvement at the end of each epoch."""
        logs = logs or {}
        current_value = logs.get(self.monitor)

        if current_value is None:
            logger.warning(
                f"EarlyStopping: Metric '{self.monitor}' not found at epoch {epoch + 1}. "
                f"Skipping check."
            )
            return

        if not isinstance(current_value, numbers.Number):
            logger.warning(
                f"EarlyStopping: Metric '{self.monitor}' is not a number "
                f"({type(current_value).__name__}). Skipping check."
            )
            return

        current_value = float(current_value)
        model = logs.get("model")

        # First epoch or initialization
        if self.best == float("inf") or self.best == float("-inf"):
            self.best = current_value
            if self.verbose:
                logger.debug(
                    f"EarlyStopping: Initial best {self.monitor}: {self.best:.6f}"
                )

            if self.restore_best_weights and model is not None:
                self.best_weights = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                logger.debug("EarlyStopping: Saved initial best weights")
            return

        # Check if improved
        if self.is_better(current_value, self.best):
            if self.verbose:
                logger.info(
                    f"Epoch {epoch + 1}: EarlyStopping {self.monitor} improved from "
                    f"{self.best:.6f} to {current_value:.6f}"
                )

            self.best = current_value
            self.wait = 0

            if self.restore_best_weights and model is not None:
                self.best_weights = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                logger.debug(
                    f"EarlyStopping: Saved new best weights at epoch {epoch + 1}"
                )
        else:
            self.wait += 1
            if self.verbose > 1:  # Extra verbosity
                logger.debug(
                    f"Epoch {epoch + 1}: EarlyStopping {self.monitor} did not improve "
                    f"({current_value:.6f}). Patience: {self.wait}/{self.patience}"
                )

            if self.wait >= self.patience:
                self.stopped_epoch = epoch + 1
                self.stop_training = True
                if self.verbose:
                    logger.info(
                        f"Epoch {epoch + 1}: EarlyStopping patience reached. "
                        f"Stopping training. Best {self.monitor}: {self.best:.6f}"
                    )

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Handle end of training, including best weight restoration if needed."""
        if self.stopped_epoch > 0 and self.verbose:
            logger.info(
                f"Training stopped early by EarlyStopping at epoch {self.stopped_epoch}."
            )
        elif self.verbose:
            logger.info("Training completed without early stopping.")

        if (
            self.restore_best_weights
            and self.best_weights is not None
            and logs is not None
        ):
            model = logs.get("model")
            if model is not None:
                device = next(model.parameters()).device  # Get model's current device
                # Load state dict ensuring weights are moved to the correct device
                state_dict_on_device = {
                    k: v.to(device) for k, v in self.best_weights.items()
                }
                model.load_state_dict(state_dict_on_device)
                logger.info(
                    f"EarlyStopping: Restored model to best weights (best {self.monitor}: {self.best:.6f})"
                )
            else:
                logger.warning(
                    "EarlyStopping: Could not restore best weights. Model not found in logs."
                )
