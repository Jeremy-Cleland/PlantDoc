# Path: plantdoc/core/training/callbacks/model_checkpoint.py
"""
Callback to save model checkpoints during training.
"""
import numbers
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

# Assuming utility functions are in plantdoc.utils
from plantdoc.utils import ensure_dir
from plantdoc.utils.logging import get_logger

# Use relative import for the base class
from .base import Callback

logger = get_logger(__name__)


class ModelCheckpoint(Callback):
    """
    Callback to save model checkpoints during training.

    Can save based on epoch frequency, metric improvement, or specific intervals.

    Args:
        dirpath: Directory to save the checkpoint files.
        filename: Filename pattern for saving checkpoints. Can contain formatting
                  options like `{epoch:03d}` and metric names like `{val_loss:.4f}`.
                  Default: "epoch_{epoch:03d}.pth".
        monitor: Metric name to monitor for saving the best model.
                 Default: "val_loss".
        save_best_only: If True, only the single best model (according to the
                        monitored quantity) will be saved to `best_filename`.
                        It will overwrite the previous best checkpoint.
                        If False, all checkpoints meeting save criteria are saved
                        using the `filename` pattern. Default: True.
        best_filename: Filename for the best model when `save_best_only=True`.
                       Default: "best_model.pth".
        mode: One of {"min", "max"}. In 'min' mode, saving occurs when the
              monitored quantity decreases; in 'max' mode, when it increases.
              Default: "min".
        save_last: If True, always saves the latest model checkpoint to
                   `last_filename`, overwriting the previous one. Useful for
                   resuming training. Default: True.
        last_filename: Filename for the last model checkpoint.
                       Default: "last_model.pth".
        save_optimizer: If True, saves the optimizer state_dict alongside the
                        model state_dict. Default: True.
        save_freq: Defines how often to save checkpoints.
                   - 'epoch': Save at the end of every epoch.
                   - integer `N`: Save at the end of every N epochs.
                   Default: "epoch".
        verbose: If True, prints messages when checkpoints are saved or improved.
                 Default: True.
    """
    priority = 80 # Run after metric calculation but before LR scheduling/early stopping

    def __init__(
        self,
        dirpath: Union[str, Path],
        filename: str = "epoch_{epoch:03d}.pth",
        monitor: str = "val_loss",
        save_best_only: bool = True,
        best_filename: str = "best_model.pth",
        mode: str = "min",
        save_last: bool = True,
        last_filename: str = "last_model.pth",
        save_optimizer: bool = True,
        save_freq: Union[str, int] = "epoch",
        verbose: bool = True,
    ):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.best_filename = best_filename
        self.save_last = save_last
        self.last_filename = last_filename
        self.save_optimizer = save_optimizer
        self.save_freq = save_freq
        self._current_epoch = 0
        self._best_epoch = -1

        if mode not in ["min", "max"]: raise ValueError(f"Mode must be 'min' or 'max', got {mode}")
        self.mode = mode
        self.best_metric = float("inf") if self.mode == "min" else float("-inf")
        self.is_better = (lambda current, best: current < best) if self.mode == "min" else (lambda current, best: current > best)

        if not (self.save_freq == "epoch" or (isinstance(self.save_freq, int) and self.save_freq > 0)):
            raise ValueError(f"save_freq must be 'epoch' or a positive integer, got {self.save_freq}")

        try:
            ensure_dir(self.dirpath)
            logger.info(f"Model checkpoints will be saved to: {self.dirpath.resolve()}")
        except Exception as e:
            logger.error(f"Failed to create checkpoint directory '{self.dirpath}': {e}", exc_info=True)
            logger.warning("Proceeding, but checkpoint saving may fail.")

    def _get_monitor_value(self, logs: Dict[str, Any]) -> Optional[float]:
        """Safely retrieves the monitored metric value from logs."""
        value = logs.get(self.monitor)
        if value is None:
            if self.save_best_only:
                logger.warning(f"ModelCheckpoint: Monitor '{self.monitor}' not found in logs: {list(logs.keys())}. Skipping best check.")
            return None
        elif not isinstance(value, numbers.Number):
            logger.warning(f"ModelCheckpoint: Monitor '{self.monitor}' is not a number ({type(value).__name__}). Skipping best check.")
            return None
        return float(value)

    def _save_checkpoint(
        self,
        filepath: Path,
        epoch: int, # 0-based internal epoch
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        metric_value: Optional[float] = None,
    ) -> None:
        """Saves the checkpoint state dictionary to a file atomically."""
        state = {
            "epoch": epoch + 1, # Save 1-based epoch
            "model_state_dict": model.state_dict(),
            "best_metric_value": self.best_metric, # Value when this checkpoint was saved (if it was best)
            "monitor": self.monitor,
            "mode": self.mode,
        }
        if self.save_optimizer and optimizer: state["optimizer_state_dict"] = optimizer.state_dict()
        if metric_value is not None: state[self.monitor] = metric_value # Current value

        temp_filepath = filepath.with_suffix(filepath.suffix + ".tmp")
        try:
            ensure_dir(filepath.parent)
            torch.save(state, temp_filepath)
            os.replace(temp_filepath, filepath) # Atomic rename/replace
            if self.verbose: logger.info(f"Checkpoint saved: '{filepath.name}' (Epoch {epoch+1})")
        except Exception as e:
            logger.error(f"Failed to save checkpoint to '{filepath}': {e}", exc_info=True)
            if temp_filepath.exists():
                try: temp_filepath.unlink()
                except OSError: logger.error(f"Failed to remove temporary file: {temp_filepath}")
        finally: # Ensure temp file cleanup on any exit path if rename failed
            if temp_filepath.exists():
                try: temp_filepath.unlink(); logger.warning(f"Removed incomplete temp checkpoint: {temp_filepath}")
                except OSError: pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Checks conditions and saves checkpoints at the end of an epoch."""
        self._current_epoch = epoch
        logs = logs or {}
        model = logs.get("model")
        optimizer = logs.get("optimizer")

        if not model: logger.error("ModelCheckpoint: 'model' not found. Cannot save."); return
        if self.save_optimizer and not optimizer: logger.warning("ModelCheckpoint: 'optimizer' not found. Optimizer state will not be saved.")

        save_this_epoch = (self.save_freq == "epoch") or \
                          (isinstance(self.save_freq, int) and (epoch + 1) % self.save_freq == 0)

        current_metric = self._get_monitor_value(logs)
        filepath_dict = {**logs, "epoch": epoch + 1} # Use 1-based epoch for filename formatting
        save_occurred = False

        # --- Best Model Logic ---
        save_as_best = False
        if current_metric is not None:
             # Handle initial state
             if self.best_metric == float('inf') or self.best_metric == float('-inf'):
                  logger.debug(f"ModelCheckpoint: Initializing best {self.monitor} to {current_metric:.6f}")
                  self.best_metric = current_metric
                  self._best_epoch = epoch
                  save_as_best = True # Save initial best
             elif self.is_better(current_metric, self.best_metric):
                  if self.verbose: logger.info(f"Epoch {epoch+1}: {self.monitor} improved from {self.best_metric:.6f} to {current_metric:.6f}.")
                  self.best_metric = current_metric
                  self._best_epoch = epoch
                  save_as_best = True

        # --- Determine Files to Save ---
        # 1. Save Best?
        if save_as_best:
            best_filepath = self.dirpath / self.best_filename
            self._save_checkpoint(best_filepath, epoch, model, optimizer, current_metric)
            save_occurred = True

        # 2. Save Epoch Checkpoint? (Only if not save_best_only and frequency matches)
        if not self.save_best_only and save_this_epoch:
            try:
                epoch_filename = self.filename.format_map(defaultdict(lambda: 'NA', filepath_dict)) # Use format_map with default
                epoch_filepath = self.dirpath / epoch_filename
                # Avoid saving if it's the same file as the best one already saved
                if not (save_occurred and epoch_filepath == best_filepath):
                     self._save_checkpoint(epoch_filepath, epoch, model, optimizer, current_metric)
                     save_occurred = True
            except KeyError as e: logger.error(f"Filename format error '{self.filename}'. Missing key: {e}. Available: {list(logs.keys())}")
            except Exception as e: logger.error(f"Failed to save epoch checkpoint: {e}", exc_info=True)

        # 3. Save Last?
        if self.save_last:
            last_filepath = self.dirpath / self.last_filename
            original_verbose = self.verbose
            # Reduce noise: only log 'last' save if verbose AND no other save happened this epoch
            if save_occurred: self.verbose = False
            self._save_checkpoint(last_filepath, epoch, model, optimizer, current_metric)
            self.verbose = original_verbose # Restore verbosity