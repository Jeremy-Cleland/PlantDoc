# Path: <your_project>/core/training/train.py  <- Adjust path as needed
"""
PyTorch trainer with support for callbacks and device optimizations,
particularly suited for MPS (Apple Silicon).

Includes additional features such as:
- Transfer learning with frozen/unfrozen backbone control
- Automatic batch handling
- IncrementalMetricsCalculator for efficient metrics tracking
- Advanced callbacks system
- MPS optimizations. Uses IncrementalMetricsCalculator.
"""

import inspect
import logging
import numbers
import random
import time
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Import the Incremental Metrics Calculator
from core.evaluation.metrics import IncrementalMetricsCalculator
from utils.logger import configure_logging, get_logger
from utils.mps_utils import (
    MPSProfiler,
    deep_clean_memory,
    empty_cache,
    is_mps_available,
    log_memory_stats,
    set_manual_seed,
    set_memory_limit,
    synchronize,
)

# Core training components
from .callbacks.base import Callback
from .callbacks.utils import get_callbacks
from .loss import get_loss_fn
from .optimizers import get_optimizer
from .schedulers import get_scheduler_with_callback

logger = get_logger(__name__)


class Trainer:
    """
    Advanced PyTorch model training class with support for:

    - Automatic device detection (CUDA, MPS, CPU)
    - MPS (Apple Silicon) optimizations
    - Mixed precision training
    - Callbacks system
    - Transfer learning with frozen/unfrozen stages

    Handles checkpoint saving, metrics calculation, and logging.

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
            raise ValueError(f"Missing required configuration key: {e}") from e

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
            scheduler=self.scheduler,  # Pass the created scheduler
            experiment_dir=str(self.experiment_dir),
            test_data=None,  # Pass None for test_data
            skip_gradcam=True,  # Skip GradCAM callback in Trainer init
        )
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

        # Initialize metrics calculator as a class member
        self.metrics_calculator = IncrementalMetricsCalculator(
            num_classes=self.num_classes, class_names=self.class_names
        )

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
            "trainer": self,  # Add the trainer itself to the context
        }
        for cb in self.callbacks:
            if hasattr(cb, "set_trainer_context"):
                cb.set_trainer_context(context)
            else:
                for key, value in context.items():
                    # Always set config regardless of current value to ensure it's passed
                    if key == "config" or hasattr(cb, key) and getattr(cb, key) is None:
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

        # Flush all logs to ensure they're written to disk
        self._flush_logs()

        return epoch_logs

    def _flush_logs(self):
        """Force flush all logs to ensure they're written to disk immediately."""
        # Get all loggers and flush their handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if hasattr(handler, "flush"):
                handler.flush()

        # Also flush our main logger
        for handler in logger.handlers:
            if hasattr(handler, "flush"):
                handler.flush()

    def _handle_train_end(
        self, start_time: float, restore_best: bool, monitor_metric: str
    ):
        # Restore logic is now handled by EarlyStopping callback if enabled.
        # This method just logs final summary and calls on_train_end for callbacks.
        total_time = time.time() - start_time

        # Get class metrics and confusion matrix
        class_metrics = None
        confusion_matrix = None

        try:
            # Get confusion matrix
            confusion_matrix = self.metrics_calculator.get_confusion_matrix()

            # Get per-class metrics if class names are available
            if self.class_names and hasattr(
                self.metrics_calculator, "get_classification_report"
            ):
                try:
                    class_f1, class_precision, class_recall = [], [], []
                    report = self.metrics_calculator.get_classification_report()

                    for i, class_name in enumerate(self.class_names):
                        if str(i) in report:
                            class_precision.append(report[str(i)]["precision"])
                            class_recall.append(report[str(i)]["recall"])
                            class_f1.append(report[str(i)]["f1-score"])
                        elif class_name in report:
                            class_precision.append(report[class_name]["precision"])
                            class_recall.append(report[class_name]["recall"])
                            class_f1.append(report[class_name]["f1-score"])

                    if class_f1 and len(class_f1) == len(self.class_names):
                        class_metrics = {
                            "f1": class_f1,
                            "precision": class_precision,
                            "recall": class_recall,
                        }
                        logger.info(
                            f"Including metrics for {len(class_metrics['f1'])} classes in final logs"
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to get classification report in _handle_train_end: {e}"
                    )
        except Exception as e:
            logger.warning(f"Error preparing metrics for callbacks: {e}")

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
            "class_metrics": class_metrics,
            "confusion_matrix": (
                confusion_matrix.tolist() if confusion_matrix is not None else None
            ),
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

    # --- Main Epoch Loops (Using IncrementalMetricsCalculator) ---
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        epoch_loss = 0.0
        epoch_start_time = time.time()
        self.metrics_calculator.reset()  # Reset the metrics calculator for this epoch
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
                batch_logs = {
                    "batch": batch_idx,
                    "size": batch_size,
                    "is_validation": False,
                }
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
                    self.metrics_calculator.update(targets.cpu(), preds.cpu())
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
        metrics_result = self.metrics_calculator.compute()
        perf_metrics = metrics_result["overall"]
        num_samples = self.metrics_calculator.total_samples
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
        self.metrics_calculator.reset()  # Reset for validation
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
                self.metrics_calculator.update(targets.cpu(), preds.cpu())
                all_outputs_list.append(outputs.cpu())
                all_targets_list.append(targets.cpu())  # Collect for callbacks
            pbar.set_postfix(loss=f"{loss_item:.4f}")

        if self.use_mps_optimizations and OmegaConf.select(
            self.mps_config, "memory.enable_sync_for_timing", default=False
        ):
            synchronize()
        epoch_time = time.time() - epoch_start_time
        metrics_result = self.metrics_calculator.compute()
        perf_metrics = metrics_result["overall"]
        num_samples = self.metrics_calculator.total_samples
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

        train_begin_logs = self._get_train_begin_logs()
        [cb.on_train_begin(train_begin_logs) for cb in self.callbacks]

        for epoch in range(self.epochs):
            self.epoch = epoch

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

        # End of training - final calculations and cleanup
        try:
            # Calculate final metrics
            class_metrics = {}
            confusion_matrix = None

            # Get confusion matrix
            confusion_matrix = self.metrics_calculator.get_confusion_matrix()

            # Get per-class metrics
            class_f1, class_precision, class_recall = [], [], []

            if hasattr(self.metrics_calculator, "get_classification_report"):
                try:
                    report = self.metrics_calculator.get_classification_report()
                    for i, class_name in enumerate(self.class_names):
                        if str(i) in report:
                            class_precision.append(report[str(i)]["precision"])
                            class_recall.append(report[str(i)]["recall"])
                            class_f1.append(report[str(i)]["f1-score"])
                        elif class_name in report:
                            class_precision.append(report[class_name]["precision"])
                            class_recall.append(report[class_name]["recall"])
                            class_f1.append(report[class_name]["f1-score"])
                except Exception as e:
                    logger.warning(f"Failed to get classification report: {e}")

            if class_f1 and len(class_f1) == len(self.class_names):
                class_metrics = {
                    "f1": class_f1,
                    "precision": class_precision,
                    "recall": class_recall,
                }

            # Log that we've calculated class metrics
            if class_metrics:
                logger.info(
                    f"Successfully calculated metrics for {len(class_metrics['f1'])} classes"
                )
            else:
                logger.warning("Could not calculate per-class metrics")

        except Exception as e:
            logger.error(f"Error calculating final metrics: {e}", exc_info=True)

        return {
            "model": self.model,
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "best_epoch": self.best_epoch,
            "total_time": time.time() - start_time,
            "history": dict(self.history),
            "class_metrics": class_metrics if class_metrics else None,
            "confusion_matrix": (
                confusion_matrix.tolist() if confusion_matrix is not None else None
            ),
        }


# --- Convenience Function ---
def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    cfg: DictConfig,
    device=None,
    callbacks=None,
    experiment_dir=None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Train a model using the provided data loaders and configuration.

    Args:
        model: Model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        cfg: Configuration object
        device: Device to train on
        callbacks: Optional list of callbacks
        experiment_dir: Directory for experiment output. If None, will try to get from cfg.paths.experiment_dir
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

    # Create experiment directory from the provided path or config
    if experiment_dir is None:
        # Fallback to config if provided
        if hasattr(cfg.paths, "experiment_dir"):
            experiment_dir = Path(cfg.paths.experiment_dir)
        else:
            raise ValueError("experiment_dir not provided and not found in config")
    else:
        experiment_dir = Path(experiment_dir)

    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Ensure experiment_dir is in cfg.paths for other components
    if not hasattr(cfg.paths, "experiment_dir"):
        cfg.paths.experiment_dir = str(experiment_dir)

    # Add metrics_dir to config if not already present - ensure it's inside experiment_dir
    if not hasattr(cfg.paths, "metrics_dir"):
        cfg.paths.metrics_dir = str(experiment_dir / "metrics")
    # Create metrics directory
    Path(cfg.paths.metrics_dir).mkdir(parents=True, exist_ok=True)

    # If log_dir is not in config, add it - ensure it's inside experiment_dir
    if not hasattr(cfg.paths, "log_dir"):
        cfg.paths.log_dir = str(experiment_dir / "logs")
    # Create logs directory
    Path(cfg.paths.log_dir).mkdir(parents=True, exist_ok=True)

    # Create a log file in experiment_dir/logs to ensure it exists
    log_file_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    cfg.logging.log_file = log_file_name
    cfg.logging.log_to_file = True

    # Ensure experiment dir is used for logging
    cfg.paths.log_dir = str(experiment_dir / "logs")
    Path(cfg.paths.log_dir).mkdir(parents=True, exist_ok=True)

    # Delete any existing log file with the same name to ensure a fresh start
    log_path = Path(cfg.paths.log_dir) / log_file_name
    if log_path.exists():
        try:
            log_path.unlink()
            logger.info(f"Removed existing log file: {log_path}")
        except Exception as e:
            logger.warning(f"Could not remove existing log file {log_path}: {e}")

    # Also create a timestamp-free symlink to the latest log for convenience
    latest_log_path = Path(cfg.paths.log_dir) / "latest_train.log"
    try:
        if latest_log_path.exists():
            latest_log_path.unlink()
        latest_log_path.symlink_to(log_file_name)
        logger.info(f"Created symlink to latest log: {latest_log_path}")
    except Exception as e:
        logger.warning(f"Could not create symlink to latest log: {e}")

    # Configure logging with updated settings
    configure_logging(cfg)
    logger.info(f"Logging configured to save in: {cfg.paths.log_dir}/{log_file_name}")
    logger.info(f"Experiment directory: {experiment_dir}")

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

    # Copy config to experiment directory
    if experiment_dir:
        import shutil

        config_path = Path(cfg._config_filename)
        if config_path.exists():
            shutil.copy2(config_path, Path(experiment_dir) / "config.yaml")
            logger.info(f"Copied config to {experiment_dir}/config.yaml")

    # Run the training process
    results = trainer.fit()
    return results
