"""
Stochastic Weight Averaging (SWA) callback.

SWA maintains a running average of model weights during training,
which can improve generalization and model performance.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim.swa_utils import SWALR, AveragedModel

from core.training.callbacks.base import Callback
from utils.logger import get_logger

logger = get_logger(__name__)


class SWACallback(Callback):
    """
    Applies Stochastic Weight Averaging (SWA).

    Averages model weights starting from a specified epoch. Can also update BN stats.

    Args:
        **kwargs: Configuration parameters. Expected keys:
            swa_start_frac (float): Fraction of total epochs to start SWA. Default: 0.75.
            swa_lr (float): Constant learning rate for SWA phase. Default: 0.05.
            anneal_epochs (int): Epochs for LR annealing before SWA LR. Default: 10.
            anneal_strategy (str): 'cos' or 'linear'. Default: 'cos'.
            update_bn_epochs (int): How often to update BN stats using val loader during SWA. 0 disables. Default: 5.
            # device: Should be passed explicitly by the caller.
            # swa_model: Optional pre-created AveragedModel. If None, created internally.
    """

    priority = (
        75  # Runs after LR scheduler changes normally happen, before checkpointing
    )

    def __init__(self, **kwargs):
        super().__init__()  # Priority set as class attribute

        # Pop 'enabled' - caller should handle this
        kwargs.pop("enabled", None)

        # SWA parameters from config
        self.swa_start_frac = kwargs.get("swa_start_frac", 0.75)
        self.swa_lr = kwargs.get("swa_lr", 0.05)
        self.anneal_epochs = kwargs.get("anneal_epochs", 10)
        self.anneal_strategy = kwargs.get("anneal_strategy", "cos")
        self.update_bn_freq = kwargs.get(
            "update_bn_epochs", 5
        )  # Frequency to update BN stats

        # External objects (passed by caller or found in logs)
        self.device = kwargs.get(
            "device"
        )  # Allow passing device via config as fallback
        self._swa_model_external = kwargs.get("swa_model")  # Allow passing SWA model
        self.experiment_dir = kwargs.get("experiment_dir")  # Store experiment directory

        # Internal state
        self._swa_model_internal: Optional[AveragedModel] = None
        self._swa_scheduler: Optional[SWALR] = None
        self._model: Optional[nn.Module] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._swa_start_epoch: Optional[int] = None
        self._total_epochs: Optional[int] = None
        self._is_swa_active = False

        # Validate
        if not (0 < self.swa_start_frac <= 1.0):
            raise ValueError("swa_start_frac must be between 0 and 1")
        if self.anneal_strategy not in ["cos", "linear"]:
            raise ValueError("anneal_strategy must be 'cos' or 'linear'")

        logger.info(
            f"Initialized SWACallback: start_frac={self.swa_start_frac}, swa_lr={self.swa_lr}, "
            f"anneal_epochs={self.anneal_epochs}, update_bn_freq={self.update_bn_freq}"
        )

        # Warn about unused keys (device and swa_model handled above)
        unused_keys = set(kwargs.keys()) - {
            "swa_start_frac",
            "swa_lr",
            "anneal_epochs",
            "anneal_strategy",
            "update_bn_epochs",
            "device",
            "swa_model",
            "experiment_dir",
        }
        if unused_keys:
            logger.warning(f"Unused config keys for SWACallback: {list(unused_keys)}")

    @property
    def swa_model(self) -> Optional[AveragedModel]:
        """Returns the active SWA model instance."""
        return (
            self._swa_model_external
            if self._swa_model_external is not None
            else self._swa_model_internal
        )

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Get model, optimizer, total epochs, and determine start epoch."""
        logs = logs or {}
        self._model = logs.get("model")
        self._optimizer = logs.get("optimizer")
        self._total_epochs = logs.get("total_epochs")  # Expect trainer to provide this

        # Get experiment_dir from logs if not provided during initialization
        if self.experiment_dir is None and "experiment_dir" in logs:
            self.experiment_dir = logs.get("experiment_dir")
            logger.info(
                f"SWACallback: Got experiment_dir from logs: {self.experiment_dir}"
            )

        if self._model is None or self._optimizer is None:
            logger.error(
                "SWACallback requires 'model' and 'optimizer' in logs on_train_begin."
            )
            return  # Cannot proceed

        if self._total_epochs is None:
            logger.warning(
                "SWACallback: 'total_epochs' not found in logs. Cannot determine SWA start epoch precisely based on fraction."
            )
            # Fallback: Maybe use a large number or disable? Or use a fixed start epoch?
            # Let's disable if total_epochs is unknown and using fraction start.
            if self.swa_start_frac > 0:
                logger.error("Disabling SWA because total_epochs is unknown.")
                self.swa_start_frac = 1.1  # Effectively disable based on fraction
                self._swa_start_epoch = float("inf")
            else:  # If start_frac was 0, maybe a fixed start epoch was intended? Needs clarification.
                self._swa_start_epoch = 0  # Assume start immediately if frac=0
        else:
            self._swa_start_epoch = int(self._total_epochs * self.swa_start_frac)

        # If device wasn't passed via kwargs, try to get from logs
        if self.device is None:
            self.device = logs.get("device")
            if self.device is None:
                logger.warning(
                    "SWACallback: Device not specified, SWA model might run on default device."
                )

        # Create internal SWA model if none provided
        if self._swa_model_external is None:
            logger.info("SWACallback: Creating internal AveragedModel.")
            self._swa_model_internal = AveragedModel(self._model)
            if self.device:
                self._swa_model_internal = self._swa_model_internal.to(self.device)
        elif self.device:  # If external model provided, ensure it's on correct device
            self._swa_model_external = self._swa_model_external.to(self.device)

        logger.info(
            f"SWACallback: Target start epoch: {self._swa_start_epoch if self._swa_start_epoch != float('inf') else 'disabled'}"
        )

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Check if SWA should start."""
        if self._swa_start_epoch is None:
            return  # Not initialized correctly

        if not self._is_swa_active and epoch >= self._swa_start_epoch:
            if self._optimizer is None:
                logger.error(
                    "SWACallback: Optimizer not available when trying to start SWA."
                )
                return

            logger.info(f"Epoch {epoch + 1}: Starting SWA phase.")
            self._is_swa_active = True
            # Create SWA LR scheduler
            try:
                self._swa_scheduler = SWALR(
                    self._optimizer,
                    swa_lr=self.swa_lr,
                    anneal_epochs=self.anneal_epochs,
                    anneal_strategy=self.anneal_strategy,
                )
                logger.info(f"  Initialized SWALR scheduler with swa_lr={self.swa_lr}")
            except Exception as e:
                logger.error(f"  Failed to create SWALR scheduler: {e}", exc_info=True)
                self._is_swa_active = False  # Disable SWA if scheduler fails

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Update SWA model and scheduler (if active). Optionally update BN stats."""
        if not self._is_swa_active or self.swa_model is None:
            return

        # --- Update SWA Model Weights ---
        try:
            self.swa_model.update_parameters(self._model)
            logger.debug(f"Epoch {epoch + 1}: Updated SWA model weights.")
        except Exception as e:
            logger.error(
                f"Epoch {epoch + 1}: Failed to update SWA model: {e}", exc_info=True
            )

        # --- Step SWA Scheduler ---
        if self._swa_scheduler:
            try:
                self._swa_scheduler.step()
                # Get current LR if available
                current_lr = getattr(
                    self._swa_scheduler, "get_last_lr", lambda: [None]
                )()[0]
                if current_lr is not None:
                    logger.debug(
                        f"Epoch {epoch + 1}: Stepped SWALR scheduler. Current LR: {current_lr:.6e}"
                    )
                    # Add to logs?
                    if logs is not None:
                        logs["swa_lr"] = current_lr
            except Exception as e:
                logger.error(
                    f"Epoch {epoch + 1}: Failed to step SWALR scheduler: {e}",
                    exc_info=True,
                )

        # --- Update Batch Normalization Statistics ---
        if self.update_bn_freq > 0 and (epoch + 1) % self.update_bn_freq == 0:
            # Try to find a suitable dataloader for updating BN stats
            # Order of preference: val_loader > test_loader > train_loader
            val_loader = logs.get("val_loader")
            if val_loader is None:
                # Try test_loader as second option
                test_loader = logs.get("test_loader")
                if test_loader is not None:
                    val_loader = test_loader
                    logger.info("SWACallback: Using test_loader for BN stats update.")
                else:
                    # Try train_loader as last resort
                    val_loader = logs.get("train_loader")
                    if val_loader is None:
                        # Try multiple options to get dataloaders
                        # 1. Try to get from trainer object
                        trainer = logs.get("trainer")
                        if trainer is not None:
                            if hasattr(trainer, "val_loader") and trainer.val_loader is not None:
                                val_loader = trainer.val_loader
                                logger.info("SWACallback: Using val_loader from trainer for BN stats update.")
                            elif hasattr(trainer, "test_loader") and trainer.test_loader is not None:
                                val_loader = trainer.test_loader
                                logger.info("SWACallback: Using test_loader from trainer for BN stats update.")
                            elif hasattr(trainer, "train_loader") and trainer.train_loader is not None:
                                val_loader = trainer.train_loader
                                logger.info("SWACallback: Using train_loader from trainer for BN stats update.")
                        
                        # 2. Try to get from data_module if it exists
                        if val_loader is None:
                            data_module = logs.get("data_module")
                            if data_module is not None:
                                if hasattr(data_module, "val_dataloader") and callable(data_module.val_dataloader):
                                    try:
                                        val_loader = data_module.val_dataloader()
                                        logger.info("SWACallback: Using val_dataloader from data_module for BN stats update.")
                                    except Exception as e:
                                        logger.warning(f"Error getting val_dataloader from data_module: {e}")
                                
                                elif hasattr(data_module, "test_dataloader") and callable(data_module.test_dataloader):
                                    try:
                                        val_loader = data_module.test_dataloader()
                                        logger.info("SWACallback: Using test_dataloader from data_module for BN stats update.")
                                    except Exception as e:
                                        logger.warning(f"Error getting test_dataloader from data_module: {e}")
                                
                                elif hasattr(data_module, "train_dataloader") and callable(data_module.train_dataloader):
                                    try:
                                        val_loader = data_module.train_dataloader()
                                        logger.info("SWACallback: Using train_dataloader from data_module for BN stats update.")
                                    except Exception as e:
                                        logger.warning(f"Error getting train_dataloader from data_module: {e}")
                        
                        # 3. Look in the model object if it might have dataloaders
                        if val_loader is None and self._model is not None:
                            if hasattr(self._model, "val_loader"):
                                val_loader = self._model.val_loader
                                logger.info("SWACallback: Using val_loader from model for BN stats update.")
                            elif hasattr(self._model, "test_loader"):
                                val_loader = self._model.test_loader
                                logger.info("SWACallback: Using test_loader from model for BN stats update.")
                            elif hasattr(self._model, "train_loader"):
                                val_loader = self._model.train_loader
                                logger.info("SWACallback: Using train_loader from model for BN stats update.")
                            
                        # If still no loader found, skip BN stats update
                        if val_loader is None:
                            logger.warning(
                                "SWACallback: No val_loader, test_loader, or train_loader found in any context. BN stats update skipped."
                            )
                            return
                    else:
                        logger.info(
                            "SWACallback: Using train_loader for BN stats update as fallback."
                        )
            else:
                logger.info("SWACallback: Using val_loader for BN stats update.")

            if self.device is None:
                logger.warning(
                    "SWACallback: Cannot update BN stats - 'device' not found."
                )
                return

            logger.info(
                f"Epoch {epoch + 1}: Updating SWA model Batch Normalization statistics using val_loader..."
            )

            try:
                # Ensure the SWA model itself is on the correct device
                if self.device:
                    self.swa_model.to(self.device)

                # Update BN stats using the validation loader with a wrapper to handle dict-type batches
                def dataloader_wrapper(loader):
                    for batch in loader:
                        # If batch is a dict, extract the image tensor
                        if isinstance(batch, dict):
                            if "image" in batch:
                                yield batch["image"]
                            elif "input" in batch:
                                yield batch["input"]
                            elif "x" in batch:
                                yield batch["x"]
                            else:
                                # Skip batch if we can't find the input
                                logger.warning(
                                    "Unable to extract input tensor from batch dict"
                                )
                                continue
                        # If batch is a tuple/list (inputs, targets, ...), yield the first element
                        elif isinstance(batch, (tuple, list)) and len(batch) > 0:
                            yield batch[0]
                        else:
                            # Pass through if it's not a dict or tuple
                            yield batch

                # Update BN stats using wrapped loader
                try:
                    torch.optim.swa_utils.update_bn(
                        dataloader_wrapper(val_loader),
                        self.swa_model,
                        device=self.device,
                    )
                    logger.info("  SWA BN statistics updated.")
                except Exception as e:
                    logger.error(f"  Failed to update SWA BN with wrapper: {e}")
                    # Fallback approach: try a more manual BN update
                    try:
                        # Set to train mode for BN statistics collection
                        self.swa_model.train()
                        # Process a few batches manually
                        with torch.no_grad():
                            batch_count = 0
                            for batch in val_loader:
                                # Process input based on batch type
                                if isinstance(batch, dict):
                                    if "image" in batch:
                                        inputs = batch["image"].to(self.device)
                                    elif "input" in batch:
                                        inputs = batch["input"].to(self.device)
                                    elif "x" in batch:
                                        inputs = batch["x"].to(self.device)
                                    else:
                                        continue
                                elif (
                                    isinstance(batch, (tuple, list)) and len(batch) > 0
                                ):
                                    inputs = batch[0].to(self.device)
                                else:
                                    continue

                                # Forward pass to update BN stats
                                self.swa_model(inputs)
                                batch_count += 1
                                if (
                                    batch_count >= 10
                                ):  # Process a reasonable number of batches
                                    break
                        logger.info(
                            "  SWA BN statistics updated using manual approach."
                        )
                    except Exception as inner_e:
                        logger.error(
                            f"  Failed to update SWA BN with manual approach: {inner_e}"
                        )

            except Exception as e:
                logger.error(
                    f"  Failed to update SWA BN statistics: {e}", exc_info=True
                )

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Log completion and potentially transfer weights."""
        if self._is_swa_active:
            logger.info("SWA phase completed.")

            # Save the SWA model if experiment directory is available
            exp_dir = None
            if self.experiment_dir is not None:
                exp_dir = self.experiment_dir
            elif logs and "experiment_dir" in logs:
                exp_dir = logs["experiment_dir"]
            elif hasattr(self, "_model") and hasattr(self._model, "experiment_dir"):
                exp_dir = self._model.experiment_dir

            if exp_dir:
                try:
                    swa_dir = Path(exp_dir) / "swa"
                    swa_dir.mkdir(parents=True, exist_ok=True)
                    swa_path = swa_dir / "swa_model.pth"

                    torch.save(
                        {
                            "model_state_dict": self.swa_model.module.state_dict(),
                            "swa_n_averaged": getattr(self.swa_model, "n_averaged", 0),
                        },
                        swa_path,
                    )

                    logger.info(f"Saved SWA model to {swa_path}")
                except Exception as e:
                    logger.error(f"Failed to save SWA model: {e}", exc_info=True)

            # Optionally copy SWA weights back to the original model if requested
            copy_to_main = logs.get("swa_copy_to_main", False) if logs else False
            if copy_to_main and self._model and self.swa_model:
                try:
                    logger.info("Copying final SWA weights back to original model.")
                    # Ensure both models are on the same device first
                    if self.device:
                        self.swa_model.to(self.device)
                        self._model.to(self.device)
                    # Copy weights
                    self._model.load_state_dict(self.swa_model.module.state_dict())
                    logger.info("Successfully copied SWA weights to main model.")
                except Exception as e:
                    logger.error(
                        f"Failed to copy SWA weights to main model: {e}", exc_info=True
                    )
