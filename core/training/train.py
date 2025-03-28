# Path: plantdoc/core/training/train.py
"""
PyTorch trainer for plant disease classification.
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
from plantdoc.utils.logging import get_logger

# Assuming MPS utils are in plantdoc utils
from plantdoc.utils.mps_utils import (
    MPSProfiler,
    deep_clean_memory,
    empty_cache,
    is_mps_available,
    log_memory_stats,
    set_manual_seed,
    synchronize,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Assuming these modules are organized correctly in plantdoc
# Note: Metrics calculator might need to be created or imported if used
# from plantdoc.core.evaluation.metrics import IncrementalMetricsCalculator
from .callbacks.base import Callback  # Import base from new location
from .callbacks.utils import get_callbacks  # Import factory from new location
from .loss import CenterLoss, get_loss_fn  # Import from local module
from .optimizers import get_optimizer  # Import from local module
from .schedulers import get_scheduler_with_callback  # Import from local module

logger = get_logger(__name__)


class Trainer:
    """
    Trainer class for plant disease classification models.

    Args:
        model: PyTorch model to train.
        cfg: Training configuration dictionary (expects keys like 'epochs',
             'optimizer', 'loss', 'scheduler', 'callbacks', 'hardware', etc.).
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        experiment_dir: Path object for the experiment output directory.
        device: Device string ('cpu', 'cuda', 'mps').
        callbacks: Optional list of pre-initialized callbacks. If None, they
                   will be created using `get_callbacks` from the config.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        experiment_dir: Path,
        device: str = "cpu",
        callbacks: Optional[List[Callback]] = None,
    ):
        self.model = model
        self.cfg = cfg # Store the full config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.experiment_dir = experiment_dir # Expecting Path object
        self.device = torch.device(device) # Store as torch.device
        self.stop_training = False

        # Ensure experiment directory exists
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # --- Configuration Extraction ---
        train_cfg = cfg.get('training', cfg) # Look for 'training' sub-dict, else use root
        optimizer_cfg = cfg.get('optimizer', {})
        loss_cfg = cfg.get('loss', {})
        scheduler_cfg = cfg.get('scheduler', {})
        callback_cfg = cfg.get('callbacks', {})
        hardware_cfg = cfg.get('hardware', {})
        transfer_cfg = cfg.get('transfer_learning', {})
        prog_resize_cfg = cfg.get('progressive_resizing', {})
        try:
            self.epochs = int(train_cfg.get('epochs', 10)) # Default epochs = 10
            self.use_mixed_precision = bool(train_cfg.get('use_mixed_precision', False))
            self.seed = int(cfg.get('seed', 42))
        except KeyError as e:
            logger.error(f"Missing required training configuration key: {e}")
            raise ValueError(f"Missing required training configuration key: {e}")
        except (TypeError, ValueError) as e:
             logger.error(f"Invalid type for core training config: {e}")
             raise

        # --- Device and Reproducibility ---
        self.device_name = self.device.type
        self.use_mps_optimizations = self.device_name == "mps" and is_mps_available()
        self.mps_config = hardware_cfg.get('mps', {}) if self.use_mps_optimizations else {}

        torch.manual_seed(self.seed)
        if self.device_name == 'cuda': torch.cuda.manual_seed_all(self.seed)
        # Note: Seeding numpy and random might be better done globally once at startup
        np.random.seed(self.seed)
        random.seed(self.seed)
        if self.use_mps_optimizations and self.mps_config.get("reproducibility", {}).get("set_mps_seed", False):
            set_manual_seed(self.seed)
            logger.info(f"Set MPS manual seed to {self.seed}")

        # --- Model Preparation ---
        self.model.to(self.device)
        if self.use_mps_optimizations and self.mps_config.get("model", {}).get("use_channels_last", False):
            logger.info("Converting model to channels_last format for MPS.")
            self.model = self.model.to(memory_format=torch.channels_last)

        # --- Training Components ---
        self.optimizer = get_optimizer(optimizer_cfg, self.model)

        # Inject context into loss config if needed
        self._prepare_loss_config(loss_cfg)
        self.criterion = get_loss_fn(loss_cfg)

        # --- Scheduler & Callbacks ---
        try:
            self.steps_per_epoch = len(self.train_loader)
            if self.steps_per_epoch == 0: raise ValueError("train_loader has zero length.")
        except (TypeError, ValueError) as e:
            logger.error(f"Could not determine steps_per_epoch: {e}. Fallback to 1.")
            self.steps_per_epoch = 1

        self.scheduler, scheduler_callback = get_scheduler_with_callback(
            scheduler_cfg, self.optimizer, self.epochs, self.steps_per_epoch
        )

        self.callbacks = []
        if callbacks is not None: self.callbacks.extend(callbacks)
        self.callbacks.extend(get_callbacks(callback_cfg, self.experiment_dir))
        if scheduler_callback is not None:
             # Avoid adding duplicate scheduler callbacks
             if not any(isinstance(cb, type(scheduler_callback)) for cb in self.callbacks):
                 self.callbacks.append(scheduler_callback)

        self.callbacks.sort(key=lambda cb: getattr(cb, "priority", 100)) # Lower runs first
        self._pass_context_to_callbacks()

        # --- Mixed Precision ---
        self._setup_mixed_precision()

        # --- Transfer Learning State ---
        self.initial_frozen_epochs = int(transfer_cfg.get('initial_frozen_epochs', 0))
        self.is_fine_tuning = False
        if self.initial_frozen_epochs > 0:
            logger.info(f"Transfer learning: Backbone frozen for first {self.initial_frozen_epochs} epochs.")
            if hasattr(self.model, 'freeze_backbone'): self.model.freeze_backbone()
            else: logger.warning("Model lacks 'freeze_backbone' method.")

        # --- Progressive Resizing (Placeholder) ---
        self.use_progressive_resizing = prog_resize_cfg.get('enabled', False)
        if self.use_progressive_resizing:
             logger.warning("Progressive Resizing logic not fully implemented in Trainer. Handle in DataModule.")
             self.prog_resize_config = prog_resize_cfg

        # --- Training State ---
        self.epoch = 0 # Use 0-based internally
        self.best_monitor_metric_val = float("inf") # Placeholder, updated based on callback
        self.best_epoch = 0
        self.history = defaultdict(list)

        logger.info("Trainer initialization complete.")


    def _prepare_loss_config(self, loss_cfg: Dict[str, Any]):
        """Inject necessary info into loss config before instantiation."""
        # Example: Pass num_classes, feature_dim for CenterLoss, device
        if 'num_classes' not in loss_cfg:
            num_classes = getattr(self.model, 'num_classes', None) or self.cfg.get('model',{}).get('num_classes', None)
            if num_classes: loss_cfg['num_classes'] = num_classes
        if 'feature_dim' not in loss_cfg:
             # Try to infer feature dim if model provides it
             feature_dim = getattr(self.model, 'feature_dim', None) # Or backbone_dim
             if feature_dim: loss_cfg['feature_dim'] = feature_dim
        if 'device' not in loss_cfg:
            loss_cfg['device'] = str(self.device) # Pass device string


    def _pass_context_to_callbacks(self):
        """Pass essential runtime objects to callbacks after initialization."""
        context = {
            'model': self.model,
            'optimizer': self.optimizer,
            'criterion': self.criterion,
            'scheduler': self.scheduler,
            'train_loader': self.train_loader,
            'val_loader': self.val_loader,
            'device': self.device,
            'config': self.cfg, # Pass full config if needed
            'experiment_dir': self.experiment_dir,
            'total_epochs': self.epochs,
        }
        for cb in self.callbacks:
            # Check if callback has a specific method or attribute to set context
            if hasattr(cb, 'set_trainer_context'):
                cb.set_trainer_context(context)
            # Fallback: Set individual attributes if they exist and are None
            for key, value in context.items():
                 if hasattr(cb, key) and getattr(cb, key) is None:
                      setattr(cb, key, value)
                      logger.debug(f"Set context '{key}' for callback {type(cb).__name__}")


    def _setup_mixed_precision(self):
        """Configure autocast context and scaler based on device and config."""
        self.scaler = None
        if self.use_mixed_precision:
            if self.device_name == "cuda":
                self.scaler = torch.cuda.amp.GradScaler()
                self.autocast_context = lambda: torch.cuda.amp.autocast(dtype=torch.float16)
                logger.info("Using CUDA mixed precision (float16) with GradScaler.")
            elif self.device_name == "mps":
                # MPS autocast uses torch.amp.autocast
                self.autocast_context = lambda: torch.amp.autocast(device_type="mps", dtype=torch.float16)
                logger.info("Using MPS mixed precision (float16).")
            else: # CPU
                self.use_mixed_precision = False
                self.autocast_context = nullcontext
                logger.warning("Mixed precision requested but device is CPU. Disabling.")
        else:
            self.autocast_context = nullcontext
            logger.info(f"Using full precision (float32) on {self.device_name}.")

    def _get_batch_data(self, batch: Any) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Extracts inputs and targets from various batch formats."""
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            inputs, targets = batch[0], batch[1]
        elif isinstance(batch, dict) and 'image' in batch and 'label' in batch:
            inputs, targets = batch['image'], batch['label']
        else:
            logger.error(f"Unexpected batch format: {type(batch)}. Cannot extract input/target.")
            return None, None

        if not isinstance(inputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
            logger.error(f"Expected tensors in batch, got {type(inputs)} and {type(targets)}.")
            return None, None

        return inputs, targets

    def _calculate_loss(self, outputs: Any, targets: torch.Tensor, features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Calculates loss, passing features if the criterion accepts them."""
        try:
            # Check if criterion is a function or an object with forward method
            is_func = callable(self.criterion) and not hasattr(self.criterion, "forward")
            loss_callable = self.criterion if is_func else self.criterion.forward
            sig = inspect.signature(loss_callable)
            accepts_features = "features" in sig.parameters
        except (ValueError, TypeError): # Handle built-ins or other issues
            accepts_features = False
            # Explicit check for CenterLoss added
            is_center_loss = isinstance(self.criterion, CenterLoss) or \
                             (hasattr(self.criterion, 'loss_components') and any(isinstance(lc[0], CenterLoss) for lc in self.criterion.loss_components))


        if accepts_features and features is not None:
            loss = self.criterion(outputs, targets, features=features)
        elif is_center_loss and features is not None: # Explicit check if signature introspection failed
             # CenterLoss often expects specific args
             if isinstance(self.criterion, CenterLoss):
                 loss = self.criterion(features=features, labels=targets)
             else: # Assume combined loss needs features passed down
                 loss = self.criterion(outputs, targets, features=features) # Hope combined loss handles it
        else:
            loss = self.criterion(outputs, targets) # Standard call

        return loss


    def train_epoch(self) -> Dict[str, float]:
        """Trains the model for one epoch."""
        self.model.train()
        train_loss = 0.0
        epoch_start_time = time.time()
        all_preds_list = []
        all_targets_list = []

        phase = "Fine-tuning" if self.is_fine_tuning else "Initial training"
        pbar = tqdm(enumerate(self.train_loader), total=self.steps_per_epoch, desc=f"Epoch {self.epoch+1}/{self.epochs} [{phase}] Train", leave=False)

        epoch_logs = {"epoch": self.epoch}
        for callback in self.callbacks: callback.on_epoch_begin(self.epoch, epoch_logs)

        profiler = MPSProfiler(enabled=self.mps_config.get("profiling", {}).get("enabled", False)) if self.use_mps_optimizations else nullcontext()

        with profiler:
            for batch_idx, batch in pbar:
                inputs, targets = self._get_batch_data(batch)
                if inputs is None: continue

                batch_size = inputs.size(0)
                batch_logs = {"batch": batch_idx, "size": batch_size}
                for callback in self.callbacks: callback.on_batch_begin(batch_idx, batch_logs)

                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True) # More efficient zeroing

                with self.autocast_context():
                    outputs = self.model(inputs)
                    features = None
                    if isinstance(outputs, tuple) and len(outputs) == 2: outputs, features = outputs
                    elif hasattr(self.model, 'get_features'): features = self.model.get_features(inputs)

                    loss = self._calculate_loss(outputs, targets, features)

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                loss_item = loss.item()
                train_loss += loss_item * batch_size

                if outputs.ndim >= 2:
                    preds = torch.argmax(outputs.detach(), dim=1)
                    all_preds_list.append(preds.cpu())
                    all_targets_list.append(targets.cpu())

                pbar.set_postfix(loss=f"{loss_item:.4f}")

                batch_end_logs = {
                    "batch": batch_idx, "size": batch_size, "loss": loss_item,
                    "is_validation": False, "outputs": outputs, "targets": targets,
                    "inputs": inputs, "optimizer": self.optimizer, "model": self.model
                }
                for callback in self.callbacks: callback.on_batch_end(batch_idx, batch_end_logs)

                if self.use_mps_optimizations and batch_idx % self.mps_config.get("memory", {}).get("clear_cache_freq", 10) == 0:
                    empty_cache()

        # --- Epoch End Calculation ---
        if self.use_mps_optimizations and self.mps_config.get("memory", {}).get("enable_sync_for_timing", False): synchronize()

        epoch_time = time.time() - epoch_start_time
        num_samples = sum(len(t) for t in all_targets_list)
        avg_train_loss = train_loss / num_samples if num_samples > 0 else 0.0

        # Calculate overall accuracy, precision, recall, F1
        final_metrics = {"loss": avg_train_loss, "time": epoch_time, "phase": phase}
        if all_targets_list:
            all_preds = torch.cat(all_preds_list).numpy()
            all_targets = torch.cat(all_targets_list).numpy()
            # --- Use scikit-learn for robust metrics ---
            try:
                from sklearn.metrics import (
                    accuracy_score,
                    precision_recall_fscore_support,
                )
                accuracy = accuracy_score(all_targets, all_preds)
                precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted', zero_division=0)
                final_metrics.update({
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                })
                logger.info(f"Epoch {self.epoch+1} Train: Loss={avg_train_loss:.4f}, Acc={accuracy:.4f}, "
                            f"Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f} ({epoch_time:.2f}s)")
            except ImportError:
                logger.warning("scikit-learn not found. Calculating only accuracy.")
                accuracy = (all_preds == all_targets).mean()
                final_metrics["accuracy"] = accuracy
                logger.info(f"Epoch {self.epoch+1} Train: Loss={avg_train_loss:.4f}, Acc={accuracy:.4f} ({epoch_time:.2f}s)")
            except Exception as e:
                 logger.error(f"Error calculating train metrics: {e}")
        else:
             logger.info(f"Epoch {self.epoch+1} Train: Loss={avg_train_loss:.4f} ({epoch_time:.2f}s) (No samples processed for metrics)")


        if self.use_mps_optimizations and self.mps_config.get("memory", {}).get("monitor", False):
            log_memory_stats(f"Epoch {self.epoch+1} train end")

        return final_metrics


    @torch.no_grad()
    def validate_epoch(self) -> Dict[str, Any]: # Return includes outputs/targets now
        """Evaluates the model on the validation set for one epoch."""
        self.model.eval()
        val_loss = 0.0
        epoch_start_time = time.time()
        all_outputs_list = []
        all_targets_list = []
        all_preds_list = [] # Store predictions as well

        phase = "Fine-tuning" if self.is_fine_tuning else "Initial training"
        pbar = tqdm(enumerate(self.val_loader), total=len(self.val_loader), desc=f"Epoch {self.epoch+1}/{self.epochs} [{phase}] Val", leave=False)

        if self.use_mps_optimizations and self.mps_config.get("memory", {}).get("monitor", False):
            log_memory_stats(f"Epoch {self.epoch+1} val start")

        for batch_idx, batch in pbar:
            inputs, targets = self._get_batch_data(batch)
            if inputs is None: continue

            batch_size = inputs.size(0)
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            if self.use_mps_optimizations and (self.is_fine_tuning or batch_idx % 10 == 0): empty_cache()

            with self.autocast_context():
                outputs = self.model(inputs)
                features = None
                if isinstance(outputs, tuple) and len(outputs) == 2: outputs, features = outputs
                elif hasattr(self.model, 'get_features'): features = self.model.get_features(inputs)

                loss = self._calculate_loss(outputs, targets, features)

            loss_item = loss.item()
            val_loss += loss_item * batch_size

            if outputs.ndim >= 2:
                preds = torch.argmax(outputs.detach(), dim=1)
                all_preds_list.append(preds.cpu())
                all_targets_list.append(targets.cpu())
                all_outputs_list.append(outputs.cpu()) # Store logits/outputs

            pbar.set_postfix(loss=f"{loss_item:.4f}")

        # --- Epoch End Calculation ---
        if self.use_mps_optimizations and self.mps_config.get("memory", {}).get("enable_sync_for_timing", False): synchronize()

        epoch_time = time.time() - epoch_start_time
        num_samples = sum(len(t) for t in all_targets_list)
        avg_val_loss = val_loss / num_samples if num_samples > 0 else 0.0

        # Calculate overall accuracy, precision, recall, F1
        final_metrics = {"val_loss": avg_val_loss, "val_time": epoch_time}
        full_val_outputs = None
        full_val_targets = None

        if all_targets_list:
            all_preds = torch.cat(all_preds_list).numpy()
            all_targets = torch.cat(all_targets_list).numpy()
            full_val_outputs = torch.cat(all_outputs_list, dim=0) # For callbacks
            full_val_targets = torch.from_numpy(all_targets)     # For callbacks

            try:
                from sklearn.metrics import (
                    accuracy_score,
                    precision_recall_fscore_support,
                )
                accuracy = accuracy_score(all_targets, all_preds)
                precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted', zero_division=0)
                final_metrics.update({
                    "val_accuracy": accuracy,
                    "val_precision": precision,
                    "val_recall": recall,
                    "val_f1": f1,
                })
                logger.info(f"Epoch {self.epoch+1} Val:   Loss={avg_val_loss:.4f}, Acc={accuracy:.4f}, "
                            f"Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f} ({epoch_time:.2f}s)")
            except ImportError:
                 accuracy = (all_preds == all_targets).mean()
                 final_metrics["val_accuracy"] = accuracy
                 logger.info(f"Epoch {self.epoch+1} Val:   Loss={avg_val_loss:.4f}, Acc={accuracy:.4f} ({epoch_time:.2f}s)")
            except Exception as e:
                 logger.error(f"Error calculating validation metrics: {e}")
        else:
             logger.info(f"Epoch {self.epoch+1} Val:   Loss={avg_val_loss:.4f} ({epoch_time:.2f}s) (No samples processed)")


        if self.use_mps_optimizations:
            if self.mps_config.get("memory", {}).get("monitor", False): log_memory_stats(f"Epoch {self.epoch+1} val end")
            if self.mps_config.get("memory", {}).get("deep_clean_after_val", True): deep_clean_memory()

        # Add full outputs/targets for callbacks that might need them (like TrainingReport)
        final_metrics["val_outputs"] = full_val_outputs
        final_metrics["val_targets"] = full_val_targets

        return final_metrics


    def train(self) -> Dict[str, Any]:
        """Executes the main training loop."""
        if self.use_mps_optimizations: deep_clean_memory()
        start_time = time.time()
        logger.info(f"===== Starting Training for {self.epochs} Epochs =====")
        logger.info(f"Device: {self.device_name}")
        if self.use_mixed_precision: logger.info(f"Mixed Precision: Enabled ({self.device_name} mode)")
        else: logger.info("Mixed Precision: Disabled (Using float32)")

        # Get primary monitor metric from callbacks (e.g., EarlyStopping or ModelCheckpoint)
        primary_monitor_metric = "val_loss" # Default
        monitor_mode = "min"
        restore_best_weights = False
        for cb in self.callbacks:
            if hasattr(cb, 'monitor'):
                primary_monitor_metric = cb.monitor
                if hasattr(cb, 'mode'): monitor_mode = cb.mode
                if hasattr(cb, 'restore_best_weights'): restore_best_weights = cb.restore_best_weights
                logger.info(f"Using '{primary_monitor_metric}' (mode: {monitor_mode}) from {type(cb).__name__} for determining best epoch.")
                break # Assume first callback with monitor is the primary one

        self.best_monitor_metric_val = float("inf") if monitor_mode == "min" else float("-inf")


        train_begin_logs = self._get_train_begin_logs()
        for callback in self.callbacks: callback.on_train_begin(train_begin_logs)

        best_model_state_on_cpu = None

        for epoch in range(self.epochs):
            self.epoch = epoch # 0-based internal epoch

            self._handle_transfer_learning_transition(epoch)
            # Add progressive resizing call here if implemented

            train_metrics = self.train_epoch()
            val_metrics = self.validate_epoch()

            epoch_logs = self._prepare_epoch_end_logs(train_metrics, val_metrics)

            # --- Check for Best Model ---
            current_monitor_val = epoch_logs.get(primary_monitor_metric)
            if current_monitor_val is not None:
                is_current_best = False
                is_better = (current_monitor_val < self.best_monitor_metric_val) if monitor_mode == 'min' else (current_monitor_val > self.best_monitor_metric_val)

                if self.best_monitor_metric_val == float('inf') or self.best_monitor_metric_val == float('-inf') or is_better:
                     self.best_monitor_metric_val = current_monitor_val
                     # Store related metrics from the best epoch
                     self.best_val_loss = epoch_logs.get('val_loss', self.best_val_loss)
                     self.best_val_acc = epoch_logs.get('val_accuracy', self.best_val_acc)
                     self.best_epoch = epoch + 1 # 1-based for reporting
                     is_current_best = True
                     logger.debug(f"Epoch {epoch+1}: New best model found ({primary_monitor_metric}={current_monitor_val:.4f}).")
                     if restore_best_weights:
                          logger.debug("Saving best model state to CPU memory.")
                          best_model_state_on_cpu = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                epoch_logs['is_best'] = is_current_best

            # --- Callbacks & Check Stop ---
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, epoch_logs)
                if getattr(callback, 'stop_training', False):
                    self.stop_training = True
                    logger.info(f"Stop training requested by {type(callback).__name__} at epoch {epoch+1}.")
                    break
            if self.stop_training: break

        # --- Training End ---
        self._handle_train_end(start_time, restore_best_weights, best_model_state_on_cpu, primary_monitor_metric)

        return {
            "model": self.model,
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "best_epoch": self.best_epoch,
            "total_time": time.time() - start_time,
            "history": dict(self.history),
        }

    def _get_train_begin_logs(self) -> Dict[str, Any]:
        """Prepare logs dictionary for on_train_begin callbacks."""
        return {
            "model": self.model, "optimizer": self.optimizer, "criterion": self.criterion,
            "scheduler": self.scheduler, "train_loader": self.train_loader, "val_loader": self.val_loader,
            "epochs": self.epochs, "device": self.device, "callbacks": self.callbacks,
            "config": self.cfg, "total_epochs": self.epochs,
            "batch_size": getattr(self.train_loader, 'batch_size', 'Unknown'),
        }

    def _handle_transfer_learning_transition(self, current_epoch: int):
        """Unfreeze backbone if transition epoch is reached."""
        if current_epoch == self.initial_frozen_epochs and self.initial_frozen_epochs > 0:
            logger.info(f"Epoch {current_epoch+1}: Unfreezing backbone for fine-tuning.")
            if self.use_mps_optimizations: deep_clean_memory()
            if hasattr(self.model, 'unfreeze_backbone'):
                self.model.unfreeze_backbone()
                self.is_fine_tuning = True
                ft_lr_factor = self.cfg.get('transfer_learning', {}).get('finetune_lr_factor', 0.1)
                if ft_lr_factor < 1.0:
                    logger.info(f"Reducing optimizer LR by factor {ft_lr_factor} for fine-tuning.")
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = max(param_group['lr'] * ft_lr_factor, 1e-7)
            else:
                 logger.warning("Model lacks 'unfreeze_backbone' method.")


    def _prepare_epoch_end_logs(self, train_metrics: Dict, val_metrics: Dict) -> Dict:
        """Combine metrics and add context for on_epoch_end callbacks."""
        epoch_logs = {**train_metrics, **val_metrics, "epoch": self.epoch}
        epoch_logs["model"] = self.model
        epoch_logs["optimizer"] = self.optimizer
        epoch_logs["is_fine_tuning"] = self.is_fine_tuning
        epoch_logs['is_validation'] = True # Indicate validation phase metrics are included

        # Add full validation outputs/targets if generated
        if "val_outputs" in val_metrics: epoch_logs["val_outputs"] = val_metrics["val_outputs"]
        if "val_targets" in val_metrics: epoch_logs["val_targets"] = val_metrics["val_targets"]

        # Update history (only scalar numbers)
        for k, v in epoch_logs.items():
            if isinstance(v, numbers.Number):
                # Convert numpy types just in case
                scalar_v = v.item() if hasattr(v, 'item') else float(v)
                self.history[k].append(scalar_v)
            elif isinstance(v, torch.Tensor) and v.numel() == 1:
                self.history[k].append(v.item())
        return epoch_logs


    def _handle_train_end(self, start_time: float, restore_best: bool, best_state: Optional[dict], monitor_metric: str):
        """Finalize training: restore weights, call callbacks, log summary."""
        if restore_best and best_state is not None:
             logger.info(f"Restoring model weights from best epoch {self.best_epoch} ({monitor_metric}={self.best_monitor_metric_val:.4f})")
             # Ensure state dict is loaded onto the correct device
             state_dict_on_device = {k: v.to(self.device) for k, v in best_state.items()}
             self.model.load_state_dict(state_dict_on_device)

        total_time = time.time() - start_time
        final_logs = {
            "total_time": total_time,
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "best_epoch": self.best_epoch,
            "best_monitor_metric_value": self.best_monitor_metric_val, # Add the best monitored value
            "model": self.model,
            "history": dict(self.history),
            "config": self.cfg,
        }
        for callback in self.callbacks: callback.on_train_end(final_logs)

        if self.use_mps_optimizations:
            if self.mps_config.get("memory", {}).get("monitor", False): log_memory_stats("Final")
            deep_clean_memory()

        hours, rem = divmod(total_time, 3600)
        mins, secs = divmod(rem, 60)
        logger.info(f"===== Training Finished in {int(hours):02d}h {int(mins):02d}m {secs:.2f}s =====")
        logger.info(f"Best Epoch: {self.best_epoch}, Best {monitor_metric.replace('_', ' ').title()}: {self.best_monitor_metric_val:.4f} "
                    f"(Loss: {self.best_val_loss:.4f}, Acc: {self.best_val_acc:.4f})")


# --- Convenience Function ---

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Dict[str, Any], # Expect full config here
    experiment_dir: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
    callbacks: Optional[List[Callback]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to initialize and run the Trainer.

    Args:
        model: PyTorch model instance.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        cfg: Full configuration dictionary (Hydra DictConfig or standard dict).
        experiment_dir: Optional path for experiment outputs. If None, uses default
                        from config or generates one.
        device: Optional device string ('cpu', 'cuda', 'mps'). If None, auto-detects.
        callbacks: Optional list of pre-initialized callbacks.

    Returns:
        Dictionary containing training results.
    """
    # Determine device
    if device is None:
        if is_mps_available(): device = "mps"
        elif torch.cuda.is_available(): device = "cuda"
        else: device = "cpu"

    # Determine experiment directory
    if experiment_dir is None:
         # Try getting from config, default if not found
         # Use OmegaConf select if cfg is DictConfig, else standard dict get
         if hasattr(cfg, 'select'): # Check if it's OmegaConf DictConfig
             exp_dir_str = cfg.select('paths.experiment_dir', default=f"outputs/training_run_{int(time.time())}")
         else:
             exp_dir_str = cfg.get('paths', {}).get('experiment_dir', f"outputs/training_run_{int(time.time())}")
         experiment_dir = Path(exp_dir_str)
    else:
        experiment_dir = Path(experiment_dir)

    # Create and run trainer
    trainer = Trainer(
        model=model,
        cfg=cfg, # Pass the full config
        train_loader=train_loader,
        val_loader=val_loader,
        experiment_dir=experiment_dir,
        device=device,
        callbacks=callbacks, # Pass pre-initialized callbacks if any
    )
    results = trainer.train()
    return results