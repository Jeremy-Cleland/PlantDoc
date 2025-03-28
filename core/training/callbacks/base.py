"""
Base Callback class for customizing training loops.
"""

from typing import Any, Dict, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class Callback:
    """
    Abstract base class used to build new callbacks.

    Callbacks can be used to customize behavior during training, validation,
    or testing loops by hooking into specific events.

    Attributes:
        priority (int): Integer indicating callback execution order. Lower numbers
                        run earlier. Callbacks with the same priority run in the
                        order they were added. Default is 100 (runs later).
        stop_training (bool): Flag that can be set by a callback (e.g., EarlyStopping)
                              to signal the main training loop to terminate.
    """

    priority: int = 100  # Default priority (runs later)
    stop_training: bool = False  # Flag to signal trainer to stop

    # --- Training Lifecycle Hooks ---
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of the overall training process."""
        pass

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of the overall training process."""
        pass

    # --- Epoch Lifecycle Hooks ---
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of each training epoch."""
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of each training epoch."""
        pass

    # --- Batch Lifecycle Hooks (Training) ---
    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of each training batch."""
        pass

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of each training batch."""
        pass
