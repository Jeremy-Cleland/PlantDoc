# Path: <your_project>/core/training/train.py  <- Adjust path as needed
"""
PyTorch trainer with support for progressive resizing, callbacks, and device optimizations,
using IncrementalMetricsCalculator. Compatible with OmegaConf configuration.
"""

import inspect
import numbers
import os
import random
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf  # Import OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# --- TODO: UPDATE THESE IMPORT PATHS ---
# Imports for progressive resizing (using get_transforms)
from core.data.transforms import AlbumentationsWrapper, get_transforms

# Import the Incremental Metrics Calculator
from core.evaluation.metrics import IncrementalMetricsCalculator
from utils.logger import get_logger
from utils.mps_utils import (
    MPSProfiler,
    deep_clean_memory,
    empty_cache,
    is_mps_available,
    log_memory_stats,
    set_manual_seed,
    set_memory_limit,  # Added based on user code
    synchronize,
)

# Core training components
from .callbacks.base import Callback
from .callbacks.utils import get_callbacks
from .loss import get_loss_fn
from .optimizers import get_optimizer
from .schedulers import get_scheduler_with_callback

# --- END TODO ---

logger = get_logger(__name__)


class Trainer:
    """
    Trainer class handling training loops, validation, callbacks, mixed precision,
    MPS optimizations, and progressive resizing. Uses IncrementalMetricsCalculator.

    Args:
        model: PyTorch model to train.
        cfg: OmegaConf DictConfig object for training configuration.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        experiment_dir: Path object for the experiment output directory.
        device: Device string ('cpu', 'cuda', 'mps').
        callbacks: Optional list of pre-initialized callbacks.
    """

    def __init__(
        self,
        model: nn.Module,
        # --- Expecting OmegaConf DictConfig ---
        cfg: DictConfig,
        # -------------------------------------
        train_loader: DataLoader,
        val_loader: DataLoader,
        experiment_dir: Path,
        device: str = "cpu",
        callbacks: Optional[List[Callback]] = None,
    ):
        # --- Use OmegaConf resolver if needed, otherwise direct access ---
        self.cfg = cfg
        # ---------------------------------------------------------------
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.experiment_dir = experiment_dir
        self.device = torch.device(device)
        self.stop_training = False

        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # --- Configuration Extraction (using OmegaConf attribute access or OmegaConf.select) ---
        # Example using direct attribute access (common with Hydra)
        try:
            train_cfg = cfg.training
            optimizer_cfg = cfg.optimizer
            loss_cfg = cfg.loss
            scheduler_cfg = cfg.scheduler
            callback_cfg = cfg.callbacks
            hardware_cfg = cfg.hardware
            transfer_cfg = cfg.transfer_learning
            prog_resize_cfg = cfg.progressive_resizing

            self.epochs = int(train_cfg.epochs)  # Assuming epochs is required
            self.use_mixed_precision = bool(
                hardware_cfg.get("precision", "float32") == "float16"
            )  # Check precision key
            self.seed = int(cfg.seed)  # Assuming seed is top-level

            # --- Get num_classes needed for metrics ---
            self.num_classes = getattr(model, "num_classes", None)
            if self.num_classes is None:
                self.num_classes = OmegaConf.select(
                    cfg, "model.num_classes", default=None
                )
                if self.num_classes is None:
                    logger.warning(
                        "Could not determine num_classes from model or config. Defaulting to 10 for metrics."
                    )
                    self.num_classes = 10
            # --- Get optional class_names ---
            self.class_names = getattr(model, "class_names", None)
            if self.class_names is None:
                self.class_names = OmegaConf.select(
                    cfg, "data.class_names", default=None
                )

        except KeyError as e:
            logger.error(
                f"Missing required configuration key via attribute access: {e}"
            )
            raise ValueError(f"Missing required configuration key: {e}")
        except Exception as e:
            logger.error(f"Error accessing configuration: {e}", exc_info=True)
            raise

        # --- Device and Reproducibility ---
        self.device_name = self.device.type
        self.use_mps_optimizations = self.device_name == "mps" and is_mps_available()
        self.mps_config = (
            OmegaConf.select(cfg, "hardware.mps", default={})
            if self.use_mps_optimizations
            else {}
        )

        torch.manual_seed(self.seed)
        if self.device_name == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.use_mps_optimizations:
            # Use OmegaConf select for safer access
            if OmegaConf.select(
                self.mps_config, "reproducibility.set_mps_seed", default=False
            ):
                set_manual_seed(self.seed)
                logger.info(f"Set MPS manual seed to {self.seed}")

            memory_limit = OmegaConf.select(
                self.mps_config, "memory.limit_fraction", default=0.0
            )
            if memory_limit > 0:
                try:
                    set_memory_limit(memory_limit)
                    logger.info(f"Set MPS memory limit fraction to {memory_limit}")
                except Exception as e:
                    logger.error(f"Failed to set MPS memory limit: {e}")

        # --- Model Preparation ---
        self.model.to(self.device)
        if self.use_mps_optimizations and OmegaConf.select(
            self.mps_config, "model.use_channels_last", default=False
        ):
            logger.info("Converting model to channels_last format for MPS.")
            self.model = self.model.to(memory_format=torch.channels_last)

        # --- Progressive Resizing Setup ---
        self.use_progressive_resizing = OmegaConf.select(
            cfg, "progressive_resizing.enabled", default=False
        )
        self.original_train_dataset = getattr(train_loader, "dataset", None)
        self.original_val_dataset = getattr(val_loader, "dataset", None)
        self.image_sizes = []
        self.size_epoch_boundaries = []
        # self.transform_params = {} # No longer needed if get_transforms reads cfg directly

        if self.use_progressive_resizing:
            if not self.original_train_dataset or not self.original_val_dataset:
                logger.warning(
                    "Progressive resizing enabled but cannot get Dataset from DataLoader. Disabling."
                )
                self.use_progressive_resizing = False
            else:
                # Use OmegaConf select for safe access
                self.image_sizes = list(
                    OmegaConf.select(
                        cfg, "progressive_resizing.sizes", default=[[224, 224]]
                    )
                )
                epochs_per_size_cfg = OmegaConf.select(
                    cfg, "progressive_resizing.epochs_per_size", default=[]
                )
                legacy_epochs_cfg = OmegaConf.select(
                    cfg, "progressive_resizing.epochs", default=[]
                )  # Legacy

                if epochs_per_size_cfg:
                    epochs_per_size = list(epochs_per_size_cfg)
                elif legacy_epochs_cfg:
                    epochs_per_size = list(legacy_epochs_cfg)
                else:  # Default logic
                    num_stages = len(self.image_sizes)
                    base_epochs = self.epochs // num_stages
                    remainder = self.epochs % num_stages
                    epochs_per_size = [base_epochs] * num_stages
                    epochs_per_size[-1] += remainder

                # Adjust length mismatch
                if len(epochs_per_size) < len(self.image_sizes):
                    epochs_per_size.extend(
                        [epochs_per_size[-1]]
                        * (len(self.image_sizes) - len(epochs_per_size))
                    )
                elif len(epochs_per_size) > len(self.image_sizes):
                    epochs_per_size = epochs_per_size[: len(self.image_sizes)]

                if sum(epochs_per_size) != self.epochs:
                    logger.warning(
                        f"Progressive resizing epoch sum {sum(epochs_per_size)} != total epochs {self.epochs}."
                    )

                self.size_epoch_boundaries = [0]
                for num_ep in epochs_per_size:
                    self.size_epoch_boundaries.append(
                        self.size_epoch_boundaries[-1] + num_ep
                    )

                logger.info(
                    f"Progressive resizing enabled: {len(self.image_sizes)} stages"
                )
                for i, (size, num_ep) in enumerate(
                    zip(self.image_sizes, epochs_per_size)
                ):
                    logger.info(
                        f"  Stage {i + 1}: size={size}, epochs={num_ep} (Ends after epoch {self.size_epoch_boundaries[i + 1]})"
                    )
                # No need to call _extract_transform_params if get_transforms reads cfg

        # --- Training Components ---
        # Convert OmegaConf DictConfig to a regular Python dictionary
        if isinstance(optimizer_cfg, DictConfig):
            optimizer_dict = OmegaConf.to_container(optimizer_cfg, resolve=True)
        else:
            optimizer_dict = optimizer_cfg

        self.optimizer = get_optimizer(optimizer_dict, self.model)

        # Convert loss config if needed
        if isinstance(loss_cfg, DictConfig):
            loss_dict = OmegaConf.to_container(loss_cfg, resolve=True)
        else:
            loss_dict = loss_cfg

        # Loss config prep happens before get_loss_fn
        self._prepare_loss_config(loss_dict)  # Pass the specific sub-config
        self.criterion = get_loss_fn(loss_dict)

        # --- Scheduler & Callbacks ---
        try:
            self.steps_per_epoch = len(self.train_loader)
            if self.steps_per_epoch == 0:
                raise ValueError("train_loader zero length")
        except (TypeError, ValueError) as e:
            logger.warning(f"Could not determine steps_per_epoch: {e}. Fallback=1.")
            self.steps_per_epoch = 1

        # Convert scheduler config if needed
        if isinstance(scheduler_cfg, DictConfig):
            scheduler_dict = OmegaConf.to_container(scheduler_cfg, resolve=True)
        else:
            scheduler_dict = scheduler_cfg

        self.scheduler, scheduler_callback = get_scheduler_with_callback(
            scheduler_dict, self.optimizer, self.epochs, self.steps_per_epoch
        )

        self.callbacks = []
        if callbacks is not None:
            self.callbacks.extend(callbacks)
        # Pass the full OmegaConf object to get_callbacks if it expects it
        standard_callbacks = get_callbacks(
            config=cfg,
            scheduler=None,  # Pass None explicitly for scheduler
            experiment_dir=str(self.experiment_dir),
            test_data=None,  # Pass None for test_data
            skip_gradcam=True,  # Skip GradCAM callback in Trainer init
        )  # Pass experiment_dir as a string
        self.callbacks.extend(standard_callbacks)
        if scheduler_callback is not None:
            if not any(
                isinstance(cb, type(scheduler_callback)) for cb in self.callbacks
            ):
                self.callbacks.append(scheduler_callback)

        self.callbacks.sort(key=lambda cb: getattr(cb, "priority", 100))
        self._pass_context_to_callbacks()

        # --- Mixed Precision Setup ---
        self._setup_mixed_precision()  # Uses self.use_mixed_precision derived from config

        # --- Transfer Learning State ---
        self.initial_frozen_epochs = int(
            OmegaConf.select(cfg, "transfer_learning.initial_frozen_epochs", default=0)
        )
        self.is_fine_tuning = False
        if self.initial_frozen_epochs > 0:
            logger.info(
                f"Transfer learning: Backbone frozen for first {self.initial_frozen_epochs} epochs."
            )
            if hasattr(self.model, "freeze_backbone"):
                self.model.freeze_backbone()
            else:
                logger.warning(
                    "Model lacks 'freeze_backbone'. Transfer freeze skipped."
                )
                self.initial_frozen_epochs = 0

        # --- Training State ---
        self.epoch = 0
        self.best_monitor_metric_val = float("inf")
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.history = defaultdict(list)

        logger.info("Trainer initialization complete.")

    # --- Helper Methods ---
    def _prepare_loss_config(self, loss_cfg: Union[DictConfig, Dict[str, Any]]):
        """Inject context into loss config (can be DictConfig or dict)."""
        # Use OmegaConf.update for DictConfig, standard update for dict
        is_omegaconf = isinstance(loss_cfg, DictConfig)

        if "num_classes" not in loss_cfg and self.num_classes is not None:
            if is_omegaconf:
                OmegaConf.update(loss_cfg, "num_classes", self.num_classes, merge=False)
            else:
                loss_cfg["num_classes"] = self.num_classes

        if "feature_dim" not in loss_cfg:
            feature_dim = getattr(self.model, "feature_dim", None)
            if feature_dim is not None:
                if is_omegaconf:
                    OmegaConf.update(loss_cfg, "feature_dim", feature_dim, merge=False)
                else:
                    loss_cfg["feature_dim"] = feature_dim

        if "device" not in loss_cfg:
            if is_omegaconf:
                OmegaConf.update(loss_cfg, "device", str(self.device), merge=False)
            else:
                loss_cfg["device"] = str(self.device)

    def _pass_context_to_callbacks(self):
        context = {
            "model": self.model,
            "optimizer": self.optimizer,
            "criterion": self.criterion,
            "scheduler": self.scheduler,
            "train_loader": self.train_loader,
            "val_loader": self.val_loader,
            "device": self.device,
            "config": self.cfg,
            "experiment_dir": self.experiment_dir,
            "total_epochs": self.epochs,
        }
        for cb in self.callbacks:
            if hasattr(cb, "set_trainer_context"):
                cb.set_trainer_context(context)
            else:
                for key, value in context.items():
                    if hasattr(cb, key) and getattr(cb, key) is None:
                        setattr(cb, key, value)

    def _setup_mixed_precision(self):
        self.scaler = None
        if self.use_mixed_precision:
            if self.device_name == "cuda":
                self.scaler = torch.cuda.amp.GradScaler()
                self.autocast_context = lambda: torch.cuda.amp.autocast(
                    dtype=torch.float16
                )
                logger.info("Using CUDA mixed precision (float16) with GradScaler.")
            elif self.device_name == "mps":
                self.autocast_context = lambda: torch.amp.autocast(
                    device_type="mps", dtype=torch.float16
                )
                logger.info("Using MPS mixed precision (float16).")
            else:
                self.use_mixed_precision = False
                self.autocast_context = nullcontext
                logger.warning(
                    "Mixed precision requested but device is CPU. Disabling."
                )
        else:
            self.autocast_context = nullcontext
            logger.info(f"Using full precision (float32) on {self.device_name}.")

    def _get_batch_data(
        self, batch: Any
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            inputs, targets = batch[0], batch[1]
        elif isinstance(batch, dict) and "image" in batch and "label" in batch:
            inputs, targets = batch["image"], batch["label"]
        else:
            logger.error(f"Unexpected batch format: {type(batch)}.")
            return None, None
        if not isinstance(inputs, torch.Tensor) or not isinstance(
            targets, torch.Tensor
        ):
            logger.error(
                f"Expected tensors in batch, got {type(inputs)} and {type(targets)}."
            )
            return None, None
        return inputs, targets

    def _calculate_loss(
        self,
        outputs: Any,
        targets: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        try:
            is_func = callable(self.criterion) and not hasattr(
                self.criterion, "forward"
            )
            loss_callable = self.criterion if is_func else self.criterion.forward
            sig = inspect.signature(loss_callable)
            accepts_features = "features" in sig.parameters
        except (ValueError, TypeError):
            accepts_features = False
        if accepts_features and features is not None:
            loss = self.criterion(outputs, targets, features=features)
        else:
            loss = self.criterion(outputs, targets)
        return loss

    def _get_train_begin_logs(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "optimizer": self.optimizer,
            "criterion": self.criterion,
            "scheduler": self.scheduler,
            "train_loader": self.train_loader,
            "val_loader": self.val_loader,
            "epochs": self.epochs,
            "device": self.device,
            "callbacks": self.callbacks,
            "config": self.cfg,
            "total_epochs": self.epochs,
            "batch_size": getattr(self.train_loader, "batch_size", "Unknown"),
        }

    def _handle_transfer_learning_transition(self, current_epoch: int):
        if (
            current_epoch == self.initial_frozen_epochs
            and self.initial_frozen_epochs > 0
        ):
            logger.info(
                f"Epoch {current_epoch + 1}: Unfreezing backbone for fine-tuning."
            )
            if self.use_mps_optimizations:
                deep_clean_memory()
            if hasattr(self.model, "unfreeze_backbone"):
                self.model.unfreeze_backbone()
                self.is_fine_tuning = True
                ft_lr_factor = OmegaConf.select(
                    self.cfg, "transfer_learning.finetune_lr_factor", default=0.1
                )
                if ft_lr_factor < 1.0:
                    logger.info(
                        f"Reducing optimizer LR by factor {ft_lr_factor} for fine-tuning."
                    )
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = max(param_group["lr"] * ft_lr_factor, 1e-7)
            else:
                logger.warning(
                    "Model lacks 'unfreeze_backbone' method. Fine-tuning transition skipped."
                )

    def _prepare_epoch_end_logs(self, train_metrics: Dict, val_metrics: Dict) -> Dict:
        epoch_logs = {**train_metrics, **val_metrics, "epoch": self.epoch}
        epoch_logs["model"] = self.model
        epoch_logs["optimizer"] = self.optimizer
        epoch_logs["is_fine_tuning"] = self.is_fine_tuning
        epoch_logs["is_validation"] = True
        if "val_outputs" in val_metrics:
            epoch_logs["val_outputs"] = val_metrics["val_outputs"]
        if "val_targets" in val_metrics:
            epoch_logs["val_targets"] = val_metrics["val_targets"]
        for k, v in epoch_logs.items():
            if isinstance(v, numbers.Number):
                scalar_v = v.item() if hasattr(v, "item") else float(v)
                self.history[k].append(scalar_v)
            elif isinstance(v, torch.Tensor) and v.numel() == 1:
                self.history[k].append(v.item())
        current_lr = self.optimizer.param_groups[0]["lr"]
        self.history["learning_rate"].append(current_lr)
        epoch_logs["learning_rate"] = current_lr
        return epoch_logs

    def _handle_train_end(
        self, start_time: float, restore_best: bool, monitor_metric: str
    ):
        # Restore logic is now handled by EarlyStopping callback if enabled.
        # This method just logs final summary and calls on_train_end for callbacks.
        total_time = time.time() - start_time
        final_logs = {
            "total_time": total_time,
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "best_epoch": self.best_epoch,
            "best_monitor_metric_value": self.best_monitor_metric_val,
            "model": self.model,
            "optimizer": self.optimizer,
            "history": dict(self.history),
            "config": self.cfg,
        }
        for callback in self.callbacks:
            callback.on_train_end(final_logs)  # Callbacks might restore weights here

        if self.use_mps_optimizations:
            if OmegaConf.select(self.mps_config, "memory.monitor", default=False):
                log_memory_stats("Final")
            deep_clean_memory()
        hours, rem = divmod(total_time, 3600)
        mins, secs = divmod(rem, 60)
        logger.info(
            f"===== Training Finished in {int(hours):02d}h {int(mins):02d}m {secs:.2f}s ====="
        )
        if self.best_epoch > 0:
            logger.info(
                f"Best Epoch: {self.best_epoch}, Best {monitor_metric.replace('_', ' ').title()}: {self.best_monitor_metric_val:.4f} (Loss: {self.best_val_loss:.4f}, Acc: {self.best_val_acc:.4f})"
            )
        else:
            logger.info(
                "No best epoch recorded (check validation metric and early stopping)."
            )

    # --- Progressive Resizing Helpers ---
    # Note: _extract_transform_params is removed as get_transforms should read cfg directly
    def _update_transforms_for_size(self, img_size: Union[int, List[int]]) -> None:
        """Updates dataset transforms for a new image size using get_transforms."""
        if (
            not self.use_progressive_resizing
            or not self.original_train_dataset
            or not self.original_val_dataset
        ):
            return
        try:
            # --- TODO: Ensure these imports point to your transforms.py ---
            from core.data.transforms import AlbumentationsWrapper, get_transforms
            # --- END TODO ---
        except ImportError:
            logger.error(
                "Cannot update transforms: `get_transforms` or `AlbumentationsWrapper` not found."
            )
            return

        size = (
            img_size[0]
            if isinstance(img_size, (list, tuple)) and img_size
            else (img_size if isinstance(img_size, int) else 224)
        )
        new_size_tuple = [size, size]
        logger.info(f"Progressive Resizing: Updating transforms to size {size}x{size}")

        # --- Modify OmegaConf temporarily ---
        # This requires careful handling of the config structure
        # Assumes structure like: cfg.preprocessing.resize, cfg.preprocessing.center_crop
        original_resize = None
        original_crop = None
        try:
            # Use OmegaConf.select to safely get current values, allowing defaults if missing initially
            original_resize = OmegaConf.select(
                self.cfg, "preprocessing.resize", default=None
            )
            original_crop = OmegaConf.select(
                self.cfg, "preprocessing.center_crop", default=None
            )

            # --- Use OmegaConf context manager for temporary modification ---
            # This is safer than direct modification if cfg is used elsewhere concurrently
            # It requires OmegaConf version supporting this context manager.
            # If not available, direct assignment + try/finally is the alternative.

            # Check if preprocessing node exists, create if not (might be needed)
            if "preprocessing" not in self.cfg:
                OmegaConf.update(self.cfg, "preprocessing", {}, merge=True)

            with OmegaConf.set_struct(
                self.cfg, False
            ):  # Temporarily allow adding fields if needed
                OmegaConf.update(
                    self.cfg, "preprocessing.resize", new_size_tuple, merge=False
                )
                OmegaConf.update(
                    self.cfg, "preprocessing.center_crop", new_size_tuple, merge=False
                )

            logger.debug(f"Temporarily updated config resize/crop to {new_size_tuple}")

            # Create new transforms using the temporarily modified config
            new_train_transform_alb = get_transforms(self.cfg, split="train")
            new_val_transform_alb = get_transforms(self.cfg, split="val")

            # Wrap transforms
            new_train_transform = AlbumentationsWrapper(new_train_transform_alb)
            new_val_transform = AlbumentationsWrapper(new_val_transform_alb)

        except Exception as e:
            logger.error(
                f"Error creating new transforms for size {size}: {e}", exc_info=True
            )
            # Ensure config is restored even if transform creation fails
            if original_resize is not None:
                OmegaConf.update(
                    self.cfg, "preprocessing.resize", original_resize, merge=False
                )
            if original_crop is not None:
                OmegaConf.update(
                    self.cfg, "preprocessing.center_crop", original_crop, merge=False
                )
            return  # Cannot proceed without transforms
        finally:
            # --- Restore original config values ---
            # This block executes even if errors occurred during transform creation
            with OmegaConf.set_struct(self.cfg, False):  # Allow modification back
                if original_resize is not None:
                    OmegaConf.update(
                        self.cfg, "preprocessing.resize", original_resize, merge=False
                    )
                if original_crop is not None:
                    OmegaConf.update(
                        self.cfg,
                        "preprocessing.center_crop",
                        original_crop,
                        merge=False,
                    )
            logger.debug("Restored original config resize/crop values.")
        # Restore struct setting if it was originally True (optional)
        # OmegaConf.set_struct(self.cfg, True)

        # Function to apply the transform
        def set_transform(loader, transform):
            dataset = loader.dataset
            original_dataset = (
                dataset.dataset if hasattr(dataset, "dataset") else dataset
            )  # Handle Subset
            if hasattr(original_dataset, "transform"):
                original_dataset.transform = transform
                logger.debug(f"Updated transform on {type(original_dataset).__name__}")
                return True
            logger.warning(
                f"Could not find 'transform' attribute on dataset {type(original_dataset).__name__}."
            )
            return False

        # Apply the new wrapped transforms
        updated_train = set_transform(self.train_loader, new_train_transform)
        updated_val = set_transform(self.val_loader, new_val_transform)

        if updated_train and updated_val:
            logger.info(f"Successfully updated transforms for size {size}x{size}.")
        else:
            logger.error("Failed to update transforms on one or both datasets.")

    # --- Main Epoch Loops (Using IncrementalMetricsCalculator) ---
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        epoch_loss = 0.0
        epoch_start_time = time.time()
        metrics_calculator = IncrementalMetricsCalculator(
            num_classes=self.num_classes, class_names=self.class_names
        )
        phase = "Fine-tuning" if self.is_fine_tuning else "Initial"
        pbar_desc = f"Epoch {self.epoch + 1}/{self.epochs} [{phase}] Train"
        pbar = tqdm(
            enumerate(self.train_loader),
            total=self.steps_per_epoch,
            desc=pbar_desc,
            leave=False,
        )
        epoch_logs = {"epoch": self.epoch}
        [cb.on_epoch_begin(self.epoch, epoch_logs) for cb in self.callbacks]
        profiler = MPSProfiler(
            enabled=self.use_mps_optimizations
            and OmegaConf.select(self.mps_config, "profiling.enabled", default=False)
        )

        with profiler:
            for batch_idx, batch in pbar:
                inputs, targets = self._get_batch_data(batch)
                if inputs is None:
                    continue
                batch_size = inputs.size(0)
                batch_logs = {"batch": batch_idx, "size": batch_size}
                [cb.on_batch_begin(batch_idx, batch_logs) for cb in self.callbacks]
                inputs, targets = (
                    inputs.to(self.device, non_blocking=True),
                    targets.to(self.device, non_blocking=True),
                )
                self.optimizer.zero_grad(set_to_none=True)

                with self.autocast_context():
                    outputs = self.model(inputs)
                    features = None
                    if isinstance(outputs, tuple) and len(outputs) == 2:
                        outputs, features = outputs
                    loss = self._calculate_loss(outputs, targets, features)

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                loss_item = loss.item()
                epoch_loss += loss_item * batch_size
                if outputs.ndim >= 2:
                    preds = torch.argmax(outputs.detach(), dim=1)
                    metrics_calculator.update(targets.cpu(), preds.cpu())
                pbar.set_postfix(loss=f"{loss_item:.4f}")
                batch_end_logs = {
                    "batch": batch_idx,
                    "size": batch_size,
                    "loss": loss_item,
                    "is_validation": False,
                    "outputs": outputs,
                    "targets": targets,
                    "inputs": inputs,
                    "optimizer": self.optimizer,
                    "model": self.model,
                }
                [cb.on_batch_end(batch_idx, batch_end_logs) for cb in self.callbacks]
                if (
                    self.use_mps_optimizations
                    and batch_idx
                    % OmegaConf.select(
                        self.mps_config, "memory.clear_cache_freq", default=10
                    )
                    == 0
                ):
                    empty_cache()

        if self.use_mps_optimizations and OmegaConf.select(
            self.mps_config, "memory.enable_sync_for_timing", default=False
        ):
            synchronize()
        epoch_time = time.time() - epoch_start_time
        metrics_result = metrics_calculator.compute()
        perf_metrics = metrics_result["overall"]
        num_samples = metrics_calculator.total_samples
        avg_epoch_loss = epoch_loss / num_samples if num_samples > 0 else 0.0
        final_metrics = {
            "loss": avg_epoch_loss,
            "accuracy": metrics_result["accuracy"],
            "precision": perf_metrics["precision"],
            "recall": perf_metrics["recall"],
            "f1": perf_metrics["f1"],
            "time": epoch_time,
            "phase": phase,
        }
        logger.info(
            f"Epoch {self.epoch + 1} [{phase}] Train: Loss={avg_epoch_loss:.4f}, Acc={metrics_result['accuracy']:.4f}, Prec={perf_metrics['precision']:.4f}, Rec={perf_metrics['recall']:.4f}, F1={perf_metrics['f1']:.4f} ({epoch_time:.2f}s)"
        )
        if self.use_mps_optimizations and OmegaConf.select(
            self.mps_config, "memory.monitor", default=False
        ):
            log_memory_stats(f"Epoch {self.epoch + 1} train end")
        return final_metrics

    @torch.no_grad()
    def validate_epoch(self) -> Dict[str, Any]:
        self.model.eval()
        epoch_loss = 0.0
        epoch_start_time = time.time()
        all_outputs_list, all_targets_list = [], []  # For callbacks
        metrics_calculator = IncrementalMetricsCalculator(
            num_classes=self.num_classes, class_names=self.class_names
        )
        phase = "Fine-tuning" if self.is_fine_tuning else "Initial"
        pbar_desc = f"Epoch {self.epoch + 1}/{self.epochs} [{phase}] Val"
        pbar = tqdm(
            enumerate(self.val_loader),
            total=len(self.val_loader),
            desc=pbar_desc,
            leave=False,
        )
        if self.use_mps_optimizations and OmegaConf.select(
            self.mps_config, "memory.monitor", default=False
        ):
            log_memory_stats(f"Epoch {self.epoch + 1} val start")

        for batch_idx, batch in pbar:
            inputs, targets = self._get_batch_data(batch)
            if inputs is None:
                continue
            inputs, targets = (
                inputs.to(self.device, non_blocking=True),
                targets.to(self.device, non_blocking=True),
            )
            if self.use_mps_optimizations and batch_idx % 10 == 0:
                empty_cache()  # Default frequency

            with self.autocast_context():
                outputs = self.model(inputs)
                features = None
                if isinstance(outputs, tuple) and len(outputs) == 2:
                    outputs, features = outputs
                loss = self._calculate_loss(outputs, targets, features)

            loss_item = loss.item()
            epoch_loss += loss_item * inputs.size(0)
            if outputs.ndim >= 2:
                preds = torch.argmax(outputs.detach(), dim=1)
                metrics_calculator.update(targets.cpu(), preds.cpu())
                all_outputs_list.append(outputs.cpu())
                all_targets_list.append(targets.cpu())  # Collect for callbacks
            pbar.set_postfix(loss=f"{loss_item:.4f}")

        if self.use_mps_optimizations and OmegaConf.select(
            self.mps_config, "memory.enable_sync_for_timing", default=False
        ):
            synchronize()
        epoch_time = time.time() - epoch_start_time
        metrics_result = metrics_calculator.compute()
        perf_metrics = metrics_result["overall"]
        num_samples = metrics_calculator.total_samples
        avg_epoch_loss = epoch_loss / num_samples if num_samples > 0 else 0.0
        final_metrics = {
            "val_loss": avg_epoch_loss,
            "val_accuracy": metrics_result["accuracy"],
            "val_precision": perf_metrics["precision"],
            "val_recall": perf_metrics["recall"],
            "val_f1": perf_metrics["f1"],
            "val_time": epoch_time,
        }
        logger.info(
            f"Epoch {self.epoch + 1} [{phase}] Val:   Loss={avg_epoch_loss:.4f}, Acc={metrics_result['accuracy']:.4f}, Prec={perf_metrics['precision']:.4f}, Rec={perf_metrics['recall']:.4f}, F1={perf_metrics['f1']:.4f} ({epoch_time:.2f}s)"
        )

        full_val_outputs, full_val_targets = None, None
        if all_outputs_list:
            full_val_outputs = torch.cat(all_outputs_list, dim=0)
        if all_targets_list:
            full_val_targets = torch.cat(all_targets_list, dim=0)
        if self.use_mps_optimizations:
            if OmegaConf.select(self.mps_config, "memory.monitor", default=False):
                log_memory_stats(f"Epoch {self.epoch + 1} val end")
            if OmegaConf.select(
                self.mps_config, "memory.deep_clean_after_val", default=True
            ):
                deep_clean_memory()
        final_metrics["val_outputs"] = full_val_outputs
        final_metrics["val_targets"] = full_val_targets
        return final_metrics

    def fit(self) -> Dict[str, Any]:
        """Executes the main training loop."""
        if self.use_mps_optimizations:
            deep_clean_memory()
            log_memory_stats("Initial - Before Training")
        start_time = time.time()
        logger.info(f"===== Starting Training for {self.epochs} Epochs =====")
        logger.info(f"Device: {self.device_name}")
        if self.use_mixed_precision:
            logger.info(f"Mixed Precision: Enabled ({self.device_name} mode)")
        else:
            logger.info("Mixed Precision: Disabled (Using float32)")

        primary_monitor_metric, monitor_mode, restore_best_weights = (
            "val_loss",
            "min",
            False,
        )
        for cb in self.callbacks:
            if hasattr(cb, "monitor"):
                primary_monitor_metric, monitor_mode = (
                    cb.monitor,
                    getattr(cb, "mode", "min"),
                )
                restore_best_weights = getattr(cb, "restore_best_weights", False)
                logger.info(
                    f"Using '{primary_monitor_metric}' (mode: {monitor_mode}) from {type(cb).__name__} for best epoch logic."
                )
                if restore_best_weights:
                    logger.info(
                        "Best weights will be restored at the end if EarlyStopping enables it."
                    )
                break
        self.best_monitor_metric_val = (
            float("inf") if monitor_mode == "min" else float("-inf")
        )

        current_size_idx = -1
        if self.use_progressive_resizing and len(self.image_sizes) > 0:
            logger.info("Progressive Resizing: Applying initial transforms...")
            self._update_transforms_for_size(self.image_sizes[0])
            current_size_idx = 0
        elif self.use_progressive_resizing:
            logger.error(
                "Progressive resizing enabled but no sizes configured. Disabling."
            )
            self.use_progressive_resizing = False

        train_begin_logs = self._get_train_begin_logs()
        [cb.on_train_begin(train_begin_logs) for cb in self.callbacks]

        for epoch in range(self.epochs):
            self.epoch = epoch

            if self.use_progressive_resizing:  # Trigger resize check
                target_idx = -1
                for i in range(len(self.size_epoch_boundaries) - 1):
                    if (
                        self.size_epoch_boundaries[i]
                        <= epoch
                        < self.size_epoch_boundaries[i + 1]
                    ):
                        target_idx = i
                        break
                if target_idx != -1 and target_idx != current_size_idx:
                    current_size_idx = target_idx
                    new_size = self.image_sizes[current_size_idx]
                    logger.info(
                        f"Progressive Resizing: Transitioning to size {new_size} for epoch {epoch + 1}"
                    )
                    self._update_transforms_for_size(new_size)  # Call update method

            self._handle_transfer_learning_transition(epoch)
            train_metrics = self.train_epoch()
            val_metrics = self.validate_epoch()
            epoch_logs = self._prepare_epoch_end_logs(train_metrics, val_metrics)

            # Update best metric tracking
            current_monitor_val = epoch_logs.get(primary_monitor_metric)
            if current_monitor_val is not None:
                is_better = (
                    (current_monitor_val < self.best_monitor_metric_val)
                    if monitor_mode == "min"
                    else (current_monitor_val > self.best_monitor_metric_val)
                )
                if (
                    self.best_monitor_metric_val == float("inf")
                    or self.best_monitor_metric_val == float("-inf")
                    or is_better
                ):
                    self.best_monitor_metric_val = current_monitor_val
                    self.best_val_loss = epoch_logs.get("val_loss", self.best_val_loss)
                    self.best_val_acc = epoch_logs.get(
                        "val_accuracy", self.best_val_acc
                    )
                    self.best_epoch = epoch + 1
                    epoch_logs["is_best"] = True
                    logger.debug(
                        f"Epoch {epoch + 1}: New best model detected based on {primary_monitor_metric}={current_monitor_val:.4f}."
                    )
                else:
                    epoch_logs["is_best"] = False  # Ensure key exists

            # Callbacks & Check Stop
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, epoch_logs)
                if getattr(callback, "stop_training", False):
                    self.stop_training = True
                    logger.info(
                        f"Stop training requested by {type(callback).__name__} at epoch {epoch + 1}."
                    )
                    break
            if self.stop_training:
                break

        self._handle_train_end(start_time, restore_best_weights, primary_monitor_metric)
        return {
            "model": self.model,
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "best_epoch": self.best_epoch,
            "total_time": time.time() - start_time,
            "history": dict(self.history),
        }


# --- Convenience Function ---
def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    cfg: DictConfig,
    device=None,
    callbacks=None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Train a PyTorch model with the given configuration.

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        cfg: OmegaConf configuration object
        device: Optional device to use ('cpu', 'cuda', 'mps')
        callbacks: Optional list of callbacks
        **kwargs: Additional arguments to pass to the Trainer

    Returns:
        Dictionary of training results
    """
    # Determine device if not specified
    if device is None:
        device = (
            "mps"
            if is_mps_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info(f"No device specified, using: {device}")

    # Create experiment directory from the config
    experiment_dir = Path(cfg.paths.experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Add metrics_dir to config if not already present
    if not hasattr(cfg.paths, "metrics_dir"):
        cfg.paths.metrics_dir = str(experiment_dir / "metrics")

    # If logs_dir is not in config, add it
    if not hasattr(cfg.paths, "log_dir"):
        cfg.paths.log_dir = str(experiment_dir / "logs")

    # Ensure the metrics and logs directories exist
    Path(cfg.paths.metrics_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.log_dir).mkdir(parents=True, exist_ok=True)

    # Create callbacks if none provided
    if callbacks is None:
        # Get lr scheduler if needed
        scheduler = None
        if hasattr(cfg, "scheduler") and cfg.scheduler.name:
            # We'll let the Trainer handle optimizer creation
            # Just get callbacks without the scheduler
            pass

        # Set up GradCAM test data if needed
        test_samples = None
        if (
            hasattr(cfg, "callbacks")
            and hasattr(cfg.callbacks, "gradcam")
            and cfg.callbacks.gradcam.enabled
        ):
            # Extract class names from val_loader dataset
            class_names = None
            if hasattr(val_loader.dataset, "classes"):
                class_names = val_loader.dataset.classes
            elif hasattr(val_loader.dataset, "class_names"):
                class_names = val_loader.dataset.class_names

            # Set class_names in config
            if class_names is not None:
                cfg.callbacks.gradcam.class_names = class_names
                logger.info(
                    f"Set up {len(class_names)} class names for GradCAM visualization"
                )
            else:
                logger.warning("Could not find class names for GradCAM visualization")

            # Extract test samples for GradCAM (don't store in config)
            logger.info(
                "Setting up test data for GradCAM visualization from validation data"
            )

            # Extract a few samples from the validation loader
            test_samples = []
            sample_count = min(
                cfg.callbacks.gradcam.get("n_samples", 5), len(val_loader.dataset)
            )

            # Get a few random indices
            import random

            indices = random.sample(range(len(val_loader.dataset)), sample_count)

            # Extract those samples
            for idx in indices:
                sample = val_loader.dataset[idx]
                if isinstance(sample, (list, tuple)) and len(sample) >= 2:
                    test_samples.append((sample[0], sample[1]))
                elif (
                    isinstance(sample, dict) and "image" in sample and "label" in sample
                ):
                    # Handle dictionary format samples
                    test_samples.append((sample["image"], sample["label"]))
                else:
                    logger.warning(
                        f"Unexpected sample format for GradCAM: {type(sample)}"
                    )

            logger.info(
                f"Set up {len(test_samples)} test samples for GradCAM visualization"
            )

        # Setup callbacks - pass actual metrics_dir path to ensure it's used
        callbacks = get_callbacks(
            config=cfg,
            scheduler=None,  # No scheduler here
            experiment_dir=str(experiment_dir),
            test_data=test_samples,  # Pass test_samples directly instead of storing in config
            skip_gradcam=False,  # Don't skip GradCAM in train_model
        )

    # Create and run trainer
    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        experiment_dir=experiment_dir,
        device=device,
        callbacks=callbacks,
        **kwargs,
    )

    # Run the training process
    results = trainer.fit()
    return results
