# Table of Contents
- requirements.txt
- Makefile
- README.md
- .gitignore
- core/__init__.py
- core/training/__init__.py
- core/training/optimizers.py
- core/training/loss.py
- core/training/train.py
- core/training/schedulers.py
- core/training/callbacks/lr_scheduler.py
- core/training/callbacks/model_checkpoint.py
- core/training/callbacks/_init__.py
- core/training/callbacks/metrics_logger.py
- core/training/callbacks/early_stopping.py
- core/training/callbacks/base.py
- core/models/attention.py
- core/models/model_cbam18.py
- core/models/registry.py
- core/models/__init__.py
- core/models/base.py
- core/models/heads/residual.py
- core/models/heads/__init__.py
- core/models/backbones/__init__.py
- core/models/backbones/cbam_resnet18.py
- core/tuning/optuna_runner.py
- core/tuning/__init__.py
- core/tuning/search_space.py
- core/data/transforms.py
- core/data/prepare_data.py
- core/data/datasets.py
- core/data/__init__.py
- core/data/datamodule.py
- utils/metrics.py
- utils/paths.py
- utils/__init__.py
- utils/logger.py
- utils/visualization.py
- utils/seed.py
- cli/__init__.py
- cli/main.py
- configs/overrides/__init__.py
- configs/model/__init__.py
- configs/hydra/__init__.py
- reports/generate_report.py
- reports/generate_plots.py
- reports/templates/training_report.html
- reports/templates/__init__.py

## File: requirements.txt

- Extension: .txt
- Language: plaintext
- Size: 91 bytes
- Created: 2025-03-28 13:05:10
- Modified: 2025-03-28 13:05:10

### Code

```plaintext
typer
hydra-core
omegaconf
optuna
matplotlib
seaborn
torch
torchvision
jinja2
scikit-learn

```

## File: Makefile

- Extension: 
- Language: unknown
- Size: 321 bytes
- Created: 2025-03-28 13:05:10
- Modified: 2025-03-28 13:05:10

### Code

```unknown
EXPERIMENT ?= cbam_default
CLI=python cli/main.py
CONFIG=configs/config.yaml

train:
	$(CLI) train

eval:
	$(CLI) eval

tune:
	$(CLI) tune

report:
	python reports/generate_report.py --experiment $(EXPERIMENT)

plots:
	python reports/generate_plots.py --experiment $(EXPERIMENT)

clean:
	rm -rf outputs/* logs/* .cache/*

```

## File: README.md

- Extension: .md
- Language: markdown
- Size: 74 bytes
- Created: 2025-03-28 13:05:10
- Modified: 2025-03-28 13:05:10

### Code

```markdown
# CBAM Classification

Deep Learning project for multiclass CNN with CBAM.
```

## File: .gitignore

- Extension: 
- Language: unknown
- Size: 1656 bytes
- Created: 2025-03-28 15:12:43
- Modified: 2025-03-28 15:12:43

### Code

```unknown
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/
.python-version

# Project specific
# Data
data/raw/*
data/processed/*
data/interim/*
data/external/*
!data/raw/.gitkeep
!data/processed/.gitkeep
!data/interim/.gitkeep
!data/external/.gitkeep

# Model outputs
outputs/
checkpoints/
logs/
runs/
wandb/
artifacts/
lightning_logs/
*.ckpt
*.pth
*.pt
*.onnx
*.tflite
*.h5

# Hydra outputs
multirun/
.hydra/

# Media files (may be in datasets)
*.jpg
*.jpeg
*.png
*.gif
*.tif
*.tiff
*.bmp
*.ico
*.mp4
*.avi
*.mov

# Reports and large documents
reports/plots/
reports/figures/
*.pdf
*.pptx
*.xlsx
*.docx
*.csv
*.json
*.yaml
*.yml
!/configs/**/*.yaml
!/configs/**/*.yml

# Secrets
.env
.env.*
*.key
credentials.json
config.json

# OS specific
# macOS
.DS_Store
.AppleDouble
.LSOverride
._*

# Windows
Thumbs.db
ehthumbs.db
Desktop.ini
$RECYCLE.BIN/
*.lnk

# Linux
*~
.fuse_hidden*
.directory
.Trash-*
.nfs*

# IDE specific
# VSCode
.vscode/
*.code-workspace
.history/

# PyCharm
.idea/
*.iml
*.iws
*.ipr
*.iws
out/

# Spyder
.spyderproject
.spyproject

# Rope
.ropeproject

# Sublime Text
*.tmlanguage.cache
*.tmPreferences.cache
*.stTheme.cache
*.sublime-workspace
*.sublime-project

# Temporary files
tmp/
temp/
.tmp/
.temp/
```

## File: core/__init__.py

- Extension: .py
- Language: python
- Size: 0 bytes
- Created: 2025-03-28 15:12:54
- Modified: 2025-03-28 15:12:54

### Code

```python

```

## File: core/training/__init__.py

- Extension: .py
- Language: python
- Size: 893 bytes
- Created: 2025-03-28 14:45:06
- Modified: 2025-03-28 14:45:06

### Code

```python
"""
Core training module for PlantDoc.

Includes the Trainer class, loss functions, optimizers, schedulers,
and basic callbacks.
"""

# Main training components
from .train import Trainer, train_model
from .loss import WeightedCrossEntropyLoss, get_loss_fn
from .optimizers import get_optimizer
from .schedulers import get_scheduler, get_scheduler_with_callback

# Callbacks
from .callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateSchedulerCallback,
    MetricsLogger,
)

__all__ = [
    # Trainer
    "Trainer",
    "train_model",

    # Loss Functions
    "get_loss_fn",
    "WeightedCrossEntropyLoss",

    # Optimizers
    "get_optimizer",

    # Schedulers
    "get_scheduler",
    "get_scheduler_with_callback",

    # Callbacks
    "Callback",
    "EarlyStopping",
    "ModelCheckpoint",
    "LearningRateSchedulerCallback",
    "MetricsLogger",
]
```

## File: core/training/optimizers.py

- Extension: .py
- Language: python
- Size: 6011 bytes
- Created: 2025-03-28 14:45:29
- Modified: 2025-03-28 14:45:29

### Code

```python
"""
Optimizer configurations for model training.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW, Optimizer

from plantdoc.utils.logging import get_logger

logger = get_logger(__name__)


def get_param_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float = 0.0,
    differential_lr_factor: float = 0.1,  # Factor to reduce backbone LR
) -> List[Dict[str, Any]]:
    """
    Create parameter groups for the optimizer, optionally with differential learning rates
    for backbone and head.

    Args:
        model: The model to optimize. Assumes 'backbone' and 'head' attributes if
               differential learning rate is desired.
        base_lr: Base learning rate (typically applied to the head).
        weight_decay: Weight decay regularization applied to all groups unless overridden.
        differential_lr_factor: Factor to multiply base_lr by for the backbone. Default 0.1.

    Returns:
        List of parameter groups suitable for initializing an optimizer.
    """
    # Check if model has backbone and head attributes for differential learning rates
    has_backbone = hasattr(model, "backbone") and isinstance(model.backbone, nn.Module)
    has_head = hasattr(model, "head") and isinstance(model.head, nn.Module)

    if not has_backbone or not has_head:
        logger.info("Model does not have 'backbone' and 'head' attributes. Using single LR group.")
        # Simple case - just return all parameters in one group
        return [{"params": model.parameters(), "lr": base_lr, "weight_decay": weight_decay}]

    # Advanced case - separate learning rates for backbone and head
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.head.parameters())

    # Identify parameters that are neither in backbone nor head (if any)
    model_param_ids = {id(p) for p in model.parameters()}
    backbone_param_ids = {id(p) for p in backbone_params}
    head_param_ids = {id(p) for p in head_params}
    other_param_ids = model_param_ids - backbone_param_ids - head_param_ids
    other_params = [p for p in model.parameters() if id(p) in other_param_ids]

    backbone_lr = base_lr * differential_lr_factor
    head_lr = base_lr
    # Use base_lr for other parameters by default
    other_lr = base_lr

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay})
        logger.info(f"  Backbone group: LR={backbone_lr:.2e}, Weight Decay={weight_decay:.2e}")
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})
        logger.info(f"  Head group:     LR={head_lr:.2e}, Weight Decay={weight_decay:.2e}")
    if other_params:
        param_groups.append({"params": other_params, "lr": other_lr, "weight_decay": weight_decay})
        logger.warning(f"  Found {len(other_params)} parameters outside backbone/head. Using LR={other_lr:.2e}.")

    return param_groups


def get_optimizer(cfg: Dict[str, Any], model: nn.Module) -> Optimizer:
    """
    Create an optimizer based on the configuration.

    Args:
        cfg: Optimizer configuration dictionary (e.g., from config['optimizer']).
             Expected keys: name, lr, weight_decay, and optimizer-specific params.
        model: The model whose parameters need optimization.

    Returns:
        An initialized PyTorch Optimizer.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"Optimizer configuration must be a dictionary, got {type(cfg)}")

    optimizer_name = cfg.get("name", "adam").lower()  # Default to Adam
    lr = cfg.get("lr", 1e-3)  # Default learning rate
    weight_decay = cfg.get("weight_decay", 1e-5)  # Default weight decay

    # SGD parameters
    momentum = cfg.get("momentum", 0.9)
    nesterov = cfg.get("nesterov", False)

    # Adam/AdamW parameters
    beta1 = cfg.get("beta1", 0.9)
    beta2 = cfg.get("beta2", 0.999)
    eps = cfg.get("eps", 1e-8)
    betas = (beta1, beta2)

    # Differential LR configuration
    use_differential_lr = cfg.get("differential_lr", False)
    differential_lr_factor = cfg.get("differential_lr_factor", 0.1)

    logger.info(f"Creating '{optimizer_name}' optimizer.")
    logger.info(f"  Base LR: {lr:.2e}, Weight Decay: {weight_decay:.2e}")

    # Get parameter groups
    if use_differential_lr:
        logger.info(f"  Using differential LR with factor: {differential_lr_factor}")
        param_groups = get_param_groups(model, lr, weight_decay, differential_lr_factor)
        # Note: weight_decay is now set within each group, so set default to 0 for optimizer init
        optimizer_weight_decay = 0.0
    else:
        logger.info("  Using single LR group for all parameters.")
        param_groups = model.parameters()
        optimizer_weight_decay = weight_decay  # Apply WD directly in optimizer

    # Create the optimizer
    if optimizer_name == "sgd":
        optimizer = SGD(
            param_groups, 
            lr=lr, 
            momentum=momentum,
            weight_decay=optimizer_weight_decay, 
            nesterov=nesterov
        )
    elif optimizer_name == "adam":
        optimizer = Adam(
            param_groups, 
            lr=lr, 
            betas=betas, 
            eps=eps,
            weight_decay=optimizer_weight_decay
        )
    elif optimizer_name == "adamw":
        optimizer = AdamW(
            param_groups, 
            lr=lr, 
            betas=betas, 
            eps=eps,
            weight_decay=optimizer_weight_decay
        )
    else:
        logger.warning(f"Unsupported optimizer: '{optimizer_name}', falling back to Adam")
        optimizer = Adam(
            param_groups, 
            lr=lr, 
            betas=betas, 
            eps=eps,
            weight_decay=optimizer_weight_decay
        )

    logger.info(f"Optimizer '{type(optimizer).__name__}' created successfully.")
    return optimizer
```

## File: core/training/loss.py

- Extension: .py
- Language: python
- Size: 4323 bytes
- Created: 2025-03-28 14:45:24
- Modified: 2025-03-28 14:45:24

### Code

```python
# loss.py stub
"""
Loss functions for plant disease classification.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from plantdoc.utils.logging import get_logger

logger = get_logger(__name__)


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross Entropy Loss for handling class imbalance.

    Args:
        weight: Manual weight for each class, must be a tensor of size C.
        reduction: 'mean', 'sum' or 'none'.
        label_smoothing: Float in [0, 1]. Label smoothing coefficient.
    """
    def __init__(
        self, 
        weight: Optional[torch.Tensor] = None, 
        reduction: str = 'mean', 
        label_smoothing: float = 0.0
    ):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
        if weight is not None and not isinstance(weight, torch.Tensor):
            logger.warning("WeightedCrossEntropyLoss 'weight' should be a torch.Tensor. Converting list/tuple.")
            self.weight = torch.tensor(weight, dtype=torch.float)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate weighted cross entropy loss.

        Args:
            inputs: Predictions from model (logits) (B, C)
            targets: Ground truth class indices (B,)

        Returns:
            Loss tensor
        """
        # Ensure weight is on the correct device
        if self.weight is not None and self.weight.device != inputs.device:
            self.weight = self.weight.to(inputs.device)

        # Handle label smoothing: CE expects target indices if LS > 0
        if self.label_smoothing > 0 and targets.ndim == 2:
            logger.warning("Label smoothing applied but target is 2D. Converting target to indices using argmax.")
            targets = torch.argmax(targets, dim=1)  # Convert prob/one-hot to indices

        # Ensure targets are long integers
        if targets.dtype != torch.long and targets.ndim == 1:
            targets = targets.long()

        return F.cross_entropy(
            inputs,
            targets,
            weight=self.weight,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )


def get_loss_fn(cfg: Dict[str, Any]) -> nn.Module:
    """
    Get the appropriate loss function based on the configuration.

    Args:
        cfg: Loss configuration dictionary (e.g., from config['loss']).

    Returns:
        Loss function instance.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"Loss configuration must be a dictionary, got {type(cfg)}")

    loss_name = cfg.get("name", "cross_entropy").lower()
    logger.info(f"Initializing loss function: '{loss_name}'")

    # Extract common parameters
    reduction = cfg.get("reduction", "mean")
    label_smoothing = cfg.get("label_smoothing", 0.0)

    # Handle specific loss types
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(
            reduction=reduction,
            label_smoothing=label_smoothing
        )

    elif loss_name == "weighted_cross_entropy":
        weights_list = cfg.get("weights", None)
        weights_tensor = None
        if weights_list is not None:
            try:
                weights_tensor = torch.tensor(weights_list, dtype=torch.float)
                logger.info(f"  WeightedCrossEntropy params: using {len(weights_list)} class weights, "
                            f"reduction='{reduction}', label_smoothing={label_smoothing}")
            except Exception as e:
                logger.error(f"  Failed to convert weights to tensor: {e}. Using unweighted CE.")
                return nn.CrossEntropyLoss(
                    reduction=reduction,
                    label_smoothing=label_smoothing
                )

        return WeightedCrossEntropyLoss(
            weight=weights_tensor,
            reduction=reduction,
            label_smoothing=label_smoothing
        )

    else:
        logger.warning(f"Unsupported loss function: '{loss_name}', falling back to CrossEntropyLoss")
        return nn.CrossEntropyLoss(
            reduction=reduction,
            label_smoothing=label_smoothing
        )
```

## File: core/training/train.py

- Extension: .py
- Language: python
- Size: 33721 bytes
- Created: 2025-03-28 14:50:32
- Modified: 2025-03-28 14:50:32

### Code

```python
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
```

## File: core/training/schedulers.py

- Extension: .py
- Language: python
- Size: 4737 bytes
- Created: 2025-03-28 14:45:35
- Modified: 2025-03-28 14:45:35

### Code

```python
"""
Learning rate schedulers for training optimization.
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    ReduceLROnPlateau,
    StepLR,
)

from plantdoc.utils.logging import get_logger

# Import will be resolved later when LearningRateSchedulerCallback is defined
# from .callbacks.lr_scheduler import LearningRateSchedulerCallback

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
) -> Tuple[Optional[Union[LRScheduler, ReduceLROnPlateau]], Optional['LearningRateSchedulerCallback']]:
    """
    Create a learning rate scheduler and its corresponding callback.

    This function will be called after LearningRateSchedulerCallback is imported.

    Args:
        cfg: Scheduler configuration dictionary (e.g., from config['scheduler']).
        optimizer: Optimizer to schedule.
        num_epochs: Total number of training epochs (required for some schedulers).
        steps_per_epoch: Number of steps per epoch (required for some schedulers).

    Returns:
        Tuple of (scheduler_instance, scheduler_callback_instance) or (None, None).
    """
    # This import is here to prevent circular imports
    from .callbacks.lr_scheduler import LearningRateSchedulerCallback
    
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
```

## File: core/training/callbacks/lr_scheduler.py

- Extension: .py
- Language: python
- Size: 4541 bytes
- Created: 2025-03-28 14:45:53
- Modified: 2025-03-28 14:45:53

### Code

```python
"""
Learning rate scheduler callback for adjusting learning rates during training.
"""
from typing import Any, Dict, Optional, Union

import torch
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from .base import Callback
from plantdoc.utils.logging import get_logger

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
        log_changes: bool = True
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
            logger.warning("ReduceLROnPlateau scheduler can only be used with mode='epoch'. "
                          "Forcing mode='epoch'.")
            self.mode = "epoch"
            
        logger.info(f"Initialized LearningRateSchedulerCallback: "
                   f"scheduler={type(scheduler).__name__}, "
                   f"mode='{self.mode}', "
                   f"monitor='{self.monitor if is_plateau else 'N/A'}', "
                   f"log_changes={self.log_changes}")
        
        # Store initial learning rates
        self.initial_lrs = [group['lr'] for group in self.scheduler.optimizer.param_groups]
        
    def _log_lr(self, epoch: Optional[int] = None, batch: Optional[int] = None) -> None:
        """Log current learning rates."""
        current_lrs = [group['lr'] for group in self.scheduler.optimizer.param_groups]
        
        # Determine if LR has changed
        has_changed = any(abs(old - new) > 1e-10 for old, new in zip(self.initial_lrs, current_lrs))
        
        if has_changed or self.log_changes:
            prefix = f"Epoch {epoch+1}" if epoch is not None else f"Batch {batch+1}"
            
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
                logger.warning(f"LearningRateSchedulerCallback: "
                              f"Monitor '{self.monitor}' not found in logs. "
                              f"Scheduler step skipped.")
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
```

## File: core/training/callbacks/model_checkpoint.py

- Extension: .py
- Language: python
- Size: 10223 bytes
- Created: 2025-03-28 14:50:05
- Modified: 2025-03-28 14:50:05

### Code

```python
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
```

## File: core/training/callbacks/_init__.py

- Extension: .py
- Language: python
- Size: 501 bytes
- Created: 2025-03-28 14:45:12
- Modified: 2025-03-28 14:45:12

### Code

```python
"""
Training callbacks for PlantDoc.

Provides basic callbacks for early stopping, model checkpointing,
learning rate scheduling, and metrics logging.
"""

from .base import Callback
from .early_stopping import EarlyStopping
from .model_checkpoint import ModelCheckpoint
from .lr_scheduler import LearningRateSchedulerCallback
from .metrics_logger import MetricsLogger

__all__ = [
    "Callback",
    "EarlyStopping",
    "ModelCheckpoint",
    "LearningRateSchedulerCallback",
    "MetricsLogger",
]
```

## File: core/training/callbacks/metrics_logger.py

- Extension: .py
- Language: python
- Size: 5432 bytes
- Created: 2025-03-28 14:46:12
- Modified: 2025-03-28 14:46:12

### Code

```python
"""
Metrics logger callback for saving training metrics to disk.
"""
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base import Callback
from plantdoc.utils.logging import get_logger

logger = get_logger(__name__)


class MetricsLogger(Callback):
    """
    Callback to save training and validation metrics to disk.
    
    Args:
        metrics_dir: Directory to save metrics files
        filename: Base filename (extensions added based on format)
        save_format: Format to save metrics in ('json', 'jsonl', 'csv')
        overwrite: Whether to overwrite existing files
    """
    priority = 70  # Run after most other callbacks

    def __init__(
        self,
        metrics_dir: Union[str, Path],
        filename: str = "training_metrics",
        save_format: str = "json",
        overwrite: bool = True
    ):
        super().__init__()
        self.metrics_dir = Path(metrics_dir)
        self.filename = filename
        self.save_format = save_format.lower()
        self.overwrite = overwrite
        
        # Validate format
        if self.save_format not in ["json", "jsonl", "csv"]:
            raise ValueError(f"Format must be 'json', 'jsonl', or 'csv', got {self.save_format}")
            
        # Create metrics directory
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Initialize metrics history
        self.history = []
        self.has_saved = False
        
        logger.info(f"Initialized MetricsLogger: "
                   f"format='{self.save_format}', "
                   f"overwrite={self.overwrite}")

    def _get_filepath(self) -> Path:
        """Get the full path to the metrics file with appropriate extension."""
        if self.save_format == "json":
            return self.metrics_dir / f"{self.filename}.json"
        elif self.save_format == "jsonl":
            return self.metrics_dir / f"{self.filename}.jsonl"
        else:  # csv
            return self.metrics_dir / f"{self.filename}.csv"

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Save metrics at the end of each epoch."""
        logs = logs or {}
        
        # Filter logs to only include scalar values (numbers, booleans, strings)
        epoch_metrics = {"epoch": epoch + 1}  # 1-based epoch for user-friendly numbering
        for key, value in logs.items():
            if isinstance(value, (int, float, bool, str)):
                epoch_metrics[key] = value
                
        # Add to history
        self.history.append(epoch_metrics)
        
        # Save metrics
        self._save_metrics()

    def _save_metrics(self) -> None:
        """Save metrics to disk in the specified format."""
        filepath = self._get_filepath()
        
        # Check if file exists and we're not supposed to overwrite
        if filepath.exists() and not self.overwrite and self.has_saved:
            return
            
        try:
            if self.save_format == "json":
                self._save_json(filepath)
            elif self.save_format == "jsonl":
                self._save_jsonl(filepath)
            else:  # csv
                self._save_csv(filepath)
                
            self.has_saved = True
        except Exception as e:
            logger.error(f"Error saving metrics to {filepath}: {e}")

    def _save_json(self, filepath: Path) -> None:
        """Save metrics in JSON format."""
        with open(filepath, "w") as f:
            json.dump({"history": self.history}, f, indent=2)

    def _save_jsonl(self, filepath: Path) -> None:
        """Save metrics in JSONL format (one JSON object per line)."""
        mode = "w" if self.overwrite or not self.has_saved else "a"
        with open(filepath, mode) as f:
            for metrics in self.history:
                f.write(json.dumps(metrics) + "\n")

    def _save_csv(self, filepath: Path) -> None:
        """Save metrics in CSV format."""
        if not self.history:
            return
            
        # Get all field names from all epochs
        fieldnames = set()
        for metrics in self.history:
            fieldnames.update(metrics.keys())
        fieldnames = sorted(list(fieldnames))
        
        mode = "w" if self.overwrite or not self.has_saved else "a"
        with open(filepath, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # Write header only once
            if mode == "w":
                writer.writeheader()
                
            # Write rows
            for metrics in self.history:
                writer.writerow(metrics)

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Final save of metrics at the end of training."""
        logs = logs or {}
        
        # Add best values to the final metrics if available
        final_metrics = {}
        for key in ["best_val_loss", "best_val_acc", "best_epoch", "total_time"]:
            if key in logs:
                final_metrics[key] = logs[key]
                
        if final_metrics:
            # Add to history
            self.history.append({"final_metrics": True, **final_metrics})
            
            # Save metrics
            self._save_metrics()
            
            logger.info(f"Final metrics saved to {self._get_filepath()}")
```

## File: core/training/callbacks/early_stopping.py

- Extension: .py
- Language: python
- Size: 6432 bytes
- Created: 2025-03-28 14:45:41
- Modified: 2025-03-28 14:45:41

### Code

```python
"""
Early Stopping callback to halt training when a monitored metric stops improving.
"""
import numbers
from typing import Any, Dict, Optional

import torch

from .base import Callback
from plantdoc.utils.logging import get_logger

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
        restore_best_weights: bool = False
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
            logger.info(f"Initialized EarlyStopping: monitor='{self.monitor}', "
                       f"patience={self.patience}, mode='{self.mode}', "
                       f"min_delta={self.min_delta}, "
                       f"restore_best={self.restore_best_weights}")

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
            logger.warning(f"EarlyStopping: Metric '{self.monitor}' not found at epoch {epoch+1}. "
                          f"Skipping check.")
            return
            
        if not isinstance(current_value, numbers.Number):
            logger.warning(f"EarlyStopping: Metric '{self.monitor}' is not a number "
                          f"({type(current_value).__name__}). Skipping check.")
            return
            
        current_value = float(current_value)
        model = logs.get("model")
        
        # First epoch or initialization
        if self.best == float("inf") or self.best == float("-inf"):
            self.best = current_value
            if self.verbose:
                logger.debug(f"EarlyStopping: Initial best {self.monitor}: {self.best:.6f}")
                
            if self.restore_best_weights and model is not None:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                logger.debug("EarlyStopping: Saved initial best weights")
            return
            
        # Check if improved
        if self.is_better(current_value, self.best):
            if self.verbose:
                logger.info(f"Epoch {epoch+1}: EarlyStopping {self.monitor} improved from "
                           f"{self.best:.6f} to {current_value:.6f}")
                
            self.best = current_value
            self.wait = 0
            
            if self.restore_best_weights and model is not None:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                logger.debug(f"EarlyStopping: Saved new best weights at epoch {epoch+1}")
        else:
            self.wait += 1
            if self.verbose > 1:  # Extra verbosity
                logger.debug(f"Epoch {epoch+1}: EarlyStopping {self.monitor} did not improve "
                            f"({current_value:.6f}). Patience: {self.wait}/{self.patience}")
                
            if self.wait >= self.patience:
                self.stopped_epoch = epoch + 1
                self.stop_training = True
                if self.verbose:
                    logger.info(f"Epoch {epoch+1}: EarlyStopping patience reached. "
                               f"Stopping training. Best {self.monitor}: {self.best:.6f}")

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Handle end of training, including best weight restoration if needed."""
        if self.stopped_epoch > 0 and self.verbose:
            logger.info(f"Training stopped early by EarlyStopping at epoch {self.stopped_epoch}.")
        elif self.verbose:
            logger.info("Training completed without early stopping.")
            
        if self.restore_best_weights and self.best_weights is not None and logs is not None:
            model = logs.get("model")
            if model is not None:
                device = next(model.parameters()).device  # Get model's current device
                # Load state dict ensuring weights are moved to the correct device
                state_dict_on_device = {k: v.to(device) for k, v in self.best_weights.items()}
                model.load_state_dict(state_dict_on_device)
                logger.info(f"EarlyStopping: Restored model to best weights (best {self.monitor}: {self.best:.6f})")
            else:
                logger.warning("EarlyStopping: Could not restore best weights. Model not found in logs.")
```

## File: core/training/callbacks/base.py

- Extension: .py
- Language: python
- Size: 2022 bytes
- Created: 2025-03-28 14:45:19
- Modified: 2025-03-28 14:45:19

### Code

```python
"""
Base Callback class for customizing training loops.
"""

from typing import Any, Dict, Optional

from plantdoc.utils.logging import get_logger

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
```

## File: core/models/attention.py

- Extension: .py
- Language: python
- Size: 4095 bytes
- Created: 2025-03-28 13:40:47
- Modified: 2025-03-28 13:40:47

### Code

```python
# Path: PlantDoc/core/models/attention.py

"""
Convolutional Block Attention Module (CBAM).

This module combines channel and spatial attention mechanisms to enhance
the representation power of CNNs. Based on the paper:
"CBAM: Convolutional Block Attention Module" (https://arxiv.org/abs/1807.06521)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from plantdoc.utils.logging import get_logger

logger = get_logger(__name__)


class ChannelAttention(nn.Module):
    """
    Channel attention module for CBAM.

    This applies attention across channels using both max pooling and
    average pooling.

    Args:
        channels: Number of input channels
        reduction: Reduction ratio for the MLP
    """

    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        # Average pooling branch
        avg_out = self.mlp(self.avg_pool(x))

        # Max pooling branch
        max_out = self.mlp(self.max_pool(x))

        # Combine both features and apply sigmoid
        out = avg_out + max_out
        out = torch.sigmoid(out)

        return out


class SpatialAttention(nn.Module):
    """
    Spatial attention module for CBAM.

    This applies attention across the spatial dimensions using both
    channel-wise max pooling and average pooling.

    Args:
        kernel_size: Size of the convolution kernel, default 7
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        # Ensure kernel size is odd for padding='same' behavior
        assert kernel_size % 2 == 1, "Kernel size must be odd"

        # Spatial attention conv layer with both max and avg pooled features
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Average pooling
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Max pooling

        # Concatenate along channel dimension
        out = torch.cat([avg_out, max_out], dim=1)

        # Apply convolution and sigmoid
        out = self.conv(out)
        out = torch.sigmoid(out)

        return out


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    This combines channel attention and spatial attention sequentially.

    Args:
        channels: Number of input channels
        reduction: Reduction ratio for the channel attention
        spatial_kernel_size: Kernel size for spatial attention
        drop_path_prob: Probability for drop path regularization
    """

    def __init__(
        self, channels, reduction=16, spatial_kernel_size=7, drop_path_prob=0.0
    ):
        super(CBAM, self).__init__()

        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

        # Drop path (similar to drop connection in SE block)
        self.drop_path_prob = drop_path_prob
        self.drop_path = (
            nn.Dropout2d(drop_path_prob) if drop_path_prob > 0 else nn.Identity()
        )

        logger.info(
            f"Initialized CBAM with channels={channels}, reduction={reduction}, "
            f"spatial_kernel_size={spatial_kernel_size}, drop_path_prob={drop_path_prob}"
        )

    def forward(self, x):
        # Apply channel attention
        channel_att = self.channel_attention(x)
        x = x * channel_att

        # Apply spatial attention
        spatial_att = self.spatial_attention(x)
        x = x * spatial_att

        # Apply drop path
        if self.training and self.drop_path_prob > 0:
            x = self.drop_path(x)

        return x

```

## File: core/models/model_cbam18.py

- Extension: .py
- Language: python
- Size: 6341 bytes
- Created: 2025-03-28 13:40:47
- Modified: 2025-03-28 13:40:47

### Code

```python
# Path: PlantDoc/core/models/model_cbam18.py

"""
CBAM-Only ResNet18 model for plant disease classification.
"""

import torch
import torch.nn as nn
from plantdoc.core.models.backbones.cbam_resnet18 import CBAMResNet18Backbone
from plantdoc.core.models.base import BaseModel
from plantdoc.core.models.heads.residual import ResidualHead
from plantdoc.core.models.registry import register_model
from plantdoc.utils.logging import get_logger

logger = get_logger(__name__)


@register_model("cbam_resnet18")
class CBAMResNet18Model(BaseModel):
    """
    CBAM-Only ResNet18 model for plant disease classification with attention.

    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        freeze_backbone: Whether to freeze the backbone
        dropout_rate: Dropout rate for the head
        head_type: Type of head to use
        hidden_dim: Dimension of hidden layer in the head
        in_channels: Number of input channels
        input_size: Input image size
        reduction_ratio: Reduction ratio for CBAM
        regularization: Regularization parameters
        feature_fusion: Whether to use feature fusion from multiple layers
        use_residual_head: Whether to use residual head for classification
    """

    def __init__(
        self,
        num_classes=39,
        pretrained=True,
        freeze_backbone=False,
        dropout_rate=0.2,
        head_type="linear",
        hidden_dim=None,
        in_channels=3,
        input_size=(256, 256),
        reduction_ratio=16,
        regularization=None,
        feature_fusion=False,
        use_residual_head=False,
    ):
        # Store model-specific parameters
        self.reduction_ratio = reduction_ratio
        self.feature_fusion = feature_fusion
        self.use_residual_head = use_residual_head

        # Initialize with custom parameters
        super().__init__(
            backbone_name="cbam_resnet18",
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout_rate=dropout_rate,
            head_type=head_type,
            hidden_dim=hidden_dim,
            in_channels=in_channels,
            input_size=input_size,
            regularization=regularization,
        )

        logger.info(
            f"Initialized CBAMResNet18Model with feature_fusion={feature_fusion}, "
            f"use_residual_head={use_residual_head}, reduction_ratio={reduction_ratio}"
        )

    def _create_backbone(self):
        """
        Create CBAM-Only ResNet18 backbone.

        Returns:
            Tuple of (backbone, output_dimension)
        """
        backbone = CBAMResNet18Backbone(
            pretrained=self.pretrained,
            freeze_layers=self.frozen_backbone,
            reduction_ratio=self.reduction_ratio,
            regularization=self.regularization,
            feature_fusion=self.feature_fusion,
            in_channels=self.in_channels,
        )
        return backbone, backbone.output_dim

    def _create_head(self):
        """
        Create classification head.

        Returns:
            Classification head module
        """
        if self.use_residual_head:
            # Use residual head for better feature discrimination
            hidden_dim = self.hidden_dim or self.backbone_dim // 2
            return ResidualHead(
                in_features=self.backbone_dim,
                hidden_dim=hidden_dim,
                num_classes=self.num_classes,
                dropout_rate=self.dropout_rate,
            )
        else:
            # Use standard head from base model
            return super()._create_head()

    def gradual_unfreeze(self, stage):
        """
        Gradually unfreeze backbone layers for controlled fine-tuning.

        Args:
            stage: Stage of unfreezing (0-4)
        """
        if hasattr(self.backbone, "gradual_unfreeze"):
            self.backbone.gradual_unfreeze(stage)
            if stage > 0:
                self.frozen_backbone = False
            logger.info(f"Applied gradual unfreezing stage {stage} to CBAM-ResNet18")
        else:
            logger.warning("Backbone does not support gradual unfreezing")
            if stage > 0:
                self.unfreeze_backbone()

    def forward_features(self, x):
        """
        Extract features without classification.

        This is useful for feature visualization or center loss.

        Args:
            x: Input tensor

        Returns:
            Feature tensor before classification
        """
        # Get features from backbone
        features = self.backbone(x)
        # Flatten for feature extraction
        flat_features = torch.flatten(features, 1)
        return flat_features

    def get_gradcam_target_layer(self):
        """
        Get the target layer to use for GradCAM visualization.

        Returns:
            The target layer for GradCAM visualization
        """
        # Target the last convolutional layer before attention
        try:
            # Get the last convolutional layer in layer4
            if hasattr(self.backbone, "backbone") and hasattr(
                self.backbone.backbone, "layer4"
            ):
                # Try to get the last conv layer in the last block
                if hasattr(self.backbone.backbone.layer4[-1], "conv2"):
                    logger.info("Using layer4[-1].conv2 as GradCAM target layer")
                    return self.backbone.backbone.layer4[-1].conv2

            # Fallback to the entire layer4
            logger.info(
                "Using backbone.backbone.layer4 as GradCAM target layer (fallback)"
            )
            return self.backbone.backbone.layer4

        except (AttributeError, IndexError) as e:
            logger.warning(f"Error accessing expected layer in CBAM-ResNet18: {e}")

            # Ultimate fallback - any convolutional layer
            for name, module in self.backbone.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    logger.info(
                        f"Using {name} as GradCAM target layer (ultimate fallback)"
                    )
                    return module

            logger.error("Could not find any suitable target layer for GradCAM")
            return None

```

## File: core/models/registry.py

- Extension: .py
- Language: python
- Size: 1493 bytes
- Created: 2025-03-28 13:40:47
- Modified: 2025-03-28 13:40:47

### Code

```python
# Path: PlantDoc/core/models/registry.py

"""
Registry for model classes.
"""

from typing import Dict, Type

from plantdoc.core.models.base import BaseModel
from plantdoc.utils.logging import get_logger

logger = get_logger(__name__)

# Global registry of models
MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {}


def register_model(name, model_cls=None):
    """
    Register a model in the model registry.

    Can be used as a decorator:
    @register_model("name")
    class ModelClass: ...

    Or called directly:
    register_model("name", ModelClass)

    Args:
        name: A string identifier for the model
        model_cls: Optional model class for direct registration

    Returns:
        The model class or a decorator function
    """
    if model_cls is not None:
        # Direct registration
        MODEL_REGISTRY[name] = model_cls
        return model_cls

    # Decorator usage
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model_class(name: str) -> Type[BaseModel]:
    """
    Get a model class by name.

    Args:
        name: Name of the model

    Returns:
        Model class
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")

    return MODEL_REGISTRY[name]


def list_models() -> Dict[str, Type[BaseModel]]:
    """
    Get all registered models.

    Returns:
        Dictionary of model names to model classes
    """
    return MODEL_REGISTRY.copy()

```

## File: core/models/__init__.py

- Extension: .py
- Language: python
- Size: 368 bytes
- Created: 2025-03-28 14:06:19
- Modified: 2025-03-28 14:06:19

### Code

```python
"""
Plant disease classification models.
"""

from plantdoc.core.models.base import BaseModel
from plantdoc.core.models.cbam_resnet18 import CBAMResNet18Model
from plantdoc.core.models.registry import get_model_class, list_models, register_model

__all__ = [
    "CBAMResNet18Model",
    "register_model",
    "get_model_class",
    "list_models",

    "BaseModel",
]

```

## File: core/models/base.py

- Extension: .py
- Language: python
- Size: 3972 bytes
- Created: 2025-03-28 14:06:40
- Modified: 2025-03-28 14:06:40

### Code

```python
# Path: PlantDoc/core/models/base

"""
Base model for the plant disease classification models.
"""

import torch
import torch.nn as nn
from plantdoc.utils.logging import get_logger

logger = get_logger(__name__)


class BaseModel(nn.Module):
    """
    Base model for plant disease classification.

    This class provides a common interface for all models.

    Args:
        backbone_name: Name of the backbone to use
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        freeze_backbone: Whether to freeze the backbone
        dropout_rate: Dropout rate for the head
        head_type: Type of head to use
        hidden_dim: Dimension of hidden layer in the head
        in_channels: Number of input channels
        input_size: Input image size
        regularization: Regularization parameters
    """

    def __init__(
        self,
        backbone_name,
        num_classes=39,
        pretrained=True,
        freeze_backbone=False,
        dropout_rate=0.2,
        head_type="linear",
        hidden_dim=None,
        in_channels=3,
        input_size=(256, 256),
        regularization=None,
    ):
        super(BaseModel, self).__init__()

        # Store model parameters
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.frozen_backbone = freeze_backbone
        self.dropout_rate = dropout_rate
        self.head_type = head_type
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        self.input_size = input_size
        self.regularization = regularization or {}

        # Create backbone and get output dimension
        self.backbone, self.backbone_dim = self._create_backbone()

        # Create classification head
        self.head = self._create_head()

        logger.info(
            f"Initialized BaseModel with backbone={backbone_name}, "
            f"classes={num_classes}, pretrained={pretrained}, "
            f"frozen_backbone={freeze_backbone}"
        )

    def _create_backbone(self):
        """
        Create backbone.

        Returns:
            Tuple of (backbone, output_dimension)
        """
        raise NotImplementedError("Subclasses must implement _create_backbone")

    def _create_head(self):
        """
        Create classification head.

        Returns:
            Classification head module
        """
        if self.head_type == "linear":
            return nn.Sequential(
                nn.Flatten(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.backbone_dim, self.num_classes),
            )
        elif self.head_type == "mlp":
            hidden_dim = self.hidden_dim or self.backbone_dim // 2
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.backbone_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
                nn.Linear(hidden_dim, self.num_classes),
            )
        else:
            raise ValueError(f"Unknown head type: {self.head_type}")

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x: Input tensor

        Returns:
            Classification logits
        """
        features = self.backbone(x)
        output = self.head(features)
        return output

    def unfreeze_backbone(self):
        """
        Unfreeze backbone for fine-tuning.
        """
        if hasattr(self.backbone, "unfreeze_layers"):
            self.backbone.unfreeze_layers()
            logger.info(f"Unfroze {self.backbone_name} backbone for fine-tuning")
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True
            logger.info(f"Unfroze {self.backbone_name} backbone using generic method")
        self.frozen_backbone = False

```

## File: core/models/heads/residual.py

- Extension: .py
- Language: python
- Size: 2043 bytes
- Created: 2025-03-28 13:40:47
- Modified: 2025-03-28 13:40:47

### Code

```python
# Path: PlantDoc/core/models/heads/residual.py
"""
Residual head implementation for plant disease classification.
"""

import torch
import torch.nn as nn
from plantdoc.utils.logging import get_logger

logger = get_logger(__name__)


class ResidualHead(nn.Module):
    """
    Residual classification head with skip connections.

    This head uses residual connections and multiple layers for better
    feature discrimination.

    Args:
        in_features: Number of input features
        hidden_dim: Dimension of hidden layer
        num_classes: Number of output classes
        dropout_rate: Dropout rate
    """

    def __init__(self, in_features, hidden_dim, num_classes, dropout_rate=0.2):
        super(ResidualHead, self).__init__()

        # First branch: direct mapping
        self.direct = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout_rate), nn.Linear(in_features, num_classes)
        )

        # Second branch: MLP with residual connection
        self.residual = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # Weight for combining branches (learnable)
        self.weight = nn.Parameter(torch.tensor(0.5))

        logger.info(
            f"Initialized ResidualHead with in_features={in_features}, "
            f"hidden_dim={hidden_dim}, num_classes={num_classes}"
        )

    def forward(self, x):
        # Forward through both branches
        direct_out = self.direct(x)
        residual_out = self.residual(x)

        # Combine results with learnable weight
        w = torch.sigmoid(self.weight)  # bound between 0 and 1
        out = w * direct_out + (1 - w) * residual_out

        return out

```

## File: core/models/heads/__init__.py

- Extension: .py
- Language: python
- Size: 144 bytes
- Created: 2025-03-28 13:32:23
- Modified: 2025-03-28 13:32:23

### Code

```python
"""
Model heads for plant disease classification.
"""

from plantdoc.core.models.heads.residual import ResidualHead

__all__ = ["ResidualHead"]

```

## File: core/models/backbones/__init__.py

- Extension: .py
- Language: python
- Size: 173 bytes
- Created: 2025-03-28 13:32:23
- Modified: 2025-03-28 13:32:23

### Code

```python
"""
Model backbones for plant disease classification.
"""

from plantdoc.core.models.backbones.cbam_resnet18 import CBAMResNet18Backbone

__all__ = ["CBAMResNet18Backbone"]

```

## File: core/models/backbones/cbam_resnet18.py

- Extension: .py
- Language: python
- Size: 11164 bytes
- Created: 2025-03-28 13:32:23
- Modified: 2025-03-28 13:32:23

### Code

```python
# Path: core/models/backbones/cbam_resnet18.py
"""
CBAM-Only ResNet18 backbone for plant disease classification.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from plantdoc.core.models.backbones.blocks import BasicBlock, make_layer
from plantdoc.utils.logging import get_logger

logger = get_logger(__name__)


class CBAMResNet(nn.Module):
    """
    CBAM-Only ResNet model (without SE blocks).

    Args:
        block: Block class to use
        layers: Number of blocks in each layer
        num_classes: Number of output classes
        cbam_reduction: Reduction ratio for CBAM
        drop_path_prob: Probability for drop path regularization
        in_channels: Number of input channels
    """

    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        cbam_reduction=16,
        drop_path_prob=0.0,
        in_channels=3,
    ):
        super(CBAMResNet, self).__init__()

        self.inplanes = 64

        # Stem
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Apply increasing drop path probabilities in deeper layers
        stage_drop_paths = (
            [drop_path_prob * i / 3 for i in range(4)]
            if drop_path_prob > 0
            else [0] * 4
        )

        # Layers
        self.layer1 = make_layer(
            block,
            64,
            64,
            layers[0],
            cbam_reduction=cbam_reduction,
            drop_path_prob=stage_drop_paths[0],
        )

        self.layer2 = make_layer(
            block,
            64,
            128,
            layers[1],
            stride=2,
            cbam_reduction=cbam_reduction,
            drop_path_prob=stage_drop_paths[1],
        )

        self.layer3 = make_layer(
            block,
            128,
            256,
            layers[2],
            stride=2,
            cbam_reduction=cbam_reduction,
            drop_path_prob=stage_drop_paths[2],
        )

        self.layer4 = make_layer(
            block,
            256,
            512,
            layers[3],
            stride=2,
            cbam_reduction=cbam_reduction,
            drop_path_prob=stage_drop_paths[3],
        )

        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        logger.info(
            f"Initialized CBAMResNet with cbam_reduction={cbam_reduction}, "
            f"drop_path_prob={drop_path_prob}"
        )

    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Backbone
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def cbam_resnet18(pretrained=False, **kwargs):
    """
    Constructs a ResNet-18 model with CBAM attention.

    Args:
        pretrained: Whether to use pretrained weights
        **kwargs: Additional arguments to pass to the model

    Returns:
        CBAMResNet model
    """
    kwargs["cbam_reduction"] = kwargs.get("cbam_reduction", 16)

    model = CBAMResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    if pretrained:
        # Load standard ResNet weights and adapt to our model
        state_dict = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        ).state_dict()
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded pretrained weights for ResNet18")

    return model


class CBAMResNet18Backbone(nn.Module):
    """
    CBAM-Only ResNet-18 backbone.

    Args:
        pretrained: Whether to use pretrained weights
        freeze_layers: Whether to freeze the backbone
        reduction_ratio: Reduction ratio for CBAM
        regularization: Regularization parameters
        feature_fusion: Whether to use feature fusion from multiple layers
        in_channels: Number of input channels
    """

    def __init__(
        self,
        pretrained=True,
        freeze_layers=False,
        reduction_ratio=16,
        regularization=None,
        feature_fusion=False,
        in_channels=3,
    ):
        super(CBAMResNet18Backbone, self).__init__()

        # Process regularization parameters
        if regularization is None:
            regularization = {}

        self.drop_path_prob = regularization.get("drop_path_prob", 0.0)
        self.feature_fusion = feature_fusion

        # Create ResNet backbone with CBAM only
        self.backbone = cbam_resnet18(
            pretrained=pretrained,
            cbam_reduction=reduction_ratio,
            drop_path_prob=self.drop_path_prob,
            in_channels=in_channels,
        )

        # Remove the final fully connected layer
        self.backbone.fc = nn.Identity()

        # Output dimension for further layers
        self.output_dim = 512

        # Freeze layers if requested
        if freeze_layers:
            self.freeze_layers()

        # If using feature fusion, create adapters for each level of features
        if feature_fusion:
            # Define feature adapters for multi-scale feature fusion
            self.layer2_adapter = nn.Conv2d(
                128, self.output_dim, kernel_size=1, stride=1, padding=0, bias=False
            )
            self.layer3_adapter = nn.Conv2d(
                256, self.output_dim, kernel_size=1, stride=1, padding=0, bias=False
            )
            self.layer4_adapter = nn.Conv2d(
                512, self.output_dim, kernel_size=1, stride=1, padding=0, bias=False
            )

            # Feature fusion module
            self.fusion = nn.Sequential(
                nn.Conv2d(
                    self.output_dim * 3, self.output_dim, kernel_size=1, bias=False
                ),
                nn.BatchNorm2d(self.output_dim),
                nn.ReLU(inplace=True),
            )

            # Global average pooling to maintain output shape
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

            logger.info(f"Initialized feature fusion for multi-scale features")

        logger.info(f"Initialized CBAMResNet18Backbone with features={self.output_dim}")

    def freeze_layers(self):
        """Freeze all layers in the backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Froze all layers in CBAMResNet18Backbone")

    def unfreeze_layers(self):
        """Unfreeze all layers in the backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Unfroze all layers in CBAMResNet18Backbone")

    def gradual_unfreeze(self, stage=0):
        """
        Gradually unfreeze layers for fine-tuning.

        Args:
            stage: Stage of unfreezing (0-4)
                0: Keep all frozen
                1: Unfreeze layer4
                2: Unfreeze layer3 and layer4
                3: Unfreeze layer2, layer3, and layer4
                4: Unfreeze all layers
        """
        # Ensure all layers are frozen first
        self.freeze_layers()

        # Unfreeze based on stage
        if stage >= 1:
            # Unfreeze layer4
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True
            logger.info("Unfroze layer4")

        if stage >= 2:
            # Unfreeze layer3
            for param in self.backbone.layer3.parameters():
                param.requires_grad = True
            logger.info("Unfroze layer3")

        if stage >= 3:
            # Unfreeze layer2
            for param in self.backbone.layer2.parameters():
                param.requires_grad = True
            logger.info("Unfroze layer2")

        if stage >= 4:
            # Unfreeze layer1 and stem
            for param in self.backbone.layer1.parameters():
                param.requires_grad = True
            for param in self.backbone.conv1.parameters():
                param.requires_grad = True
            for param in self.backbone.bn1.parameters():
                param.requires_grad = True
            logger.info("Unfroze layer1 and stem")

        # Always unfreeze feature fusion modules if they exist
        if self.feature_fusion:
            if hasattr(self, "layer2_adapter"):
                for param in self.layer2_adapter.parameters():
                    param.requires_grad = True
            if hasattr(self, "layer3_adapter"):
                for param in self.layer3_adapter.parameters():
                    param.requires_grad = True
            if hasattr(self, "layer4_adapter"):
                for param in self.layer4_adapter.parameters():
                    param.requires_grad = True
            if hasattr(self, "fusion"):
                for param in self.fusion.parameters():
                    param.requires_grad = True
            logger.info("Unfroze feature fusion modules")

    def forward(self, x):
        """
        Forward pass through the backbone.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Feature tensor of shape (B, output_dim, 1, 1)
        """
        # Get stem features
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        # Layer 1
        x = self.backbone.layer1(x)

        # Layer 2
        x = self.backbone.layer2(x)
        if self.feature_fusion:
            layer2_features = self.layer2_adapter(x)
            # Upsample to match the size of layer4 features
            layer2_features = nn.functional.interpolate(
                layer2_features, scale_factor=0.25, mode="bilinear", align_corners=False
            )

        # Layer 3
        x = self.backbone.layer3(x)
        if self.feature_fusion:
            layer3_features = self.layer3_adapter(x)
            # Upsample to match the size of layer4 features
            layer3_features = nn.functional.interpolate(
                layer3_features, scale_factor=0.5, mode="bilinear", align_corners=False
            )

        # Layer 4
        x = self.backbone.layer4(x)
        if self.feature_fusion:
            layer4_features = self.layer4_adapter(x)

            # Fuse features
            x = torch.cat([layer2_features, layer3_features, layer4_features], dim=1)
            x = self.fusion(x)
            x = self.global_pool(x)
        else:
            x = self.backbone.avgpool(x)

        return x

```

## File: core/tuning/optuna_runner.py

- Extension: .py
- Language: python
- Size: 9090 bytes
- Created: 2025-03-28 15:00:16
- Modified: 2025-03-28 15:00:16

### Code

```python
# optuna_runner.py stub
# core/tuning/optuna_runner.py
import time
import optuna
from omegaconf import DictConfig, OmegaConf
import hydra
from pathlib import Path

from plantdoc.core.data import PlantDiseaseDataModule
from plantdoc.core.models import get_model_class
from plantdoc.core.training import train_model
from plantdoc.utils.logging import get_logger
from plantdoc.utils.seed import set_seed

# Import suggestion functions
from .search_space import (
    suggest_optimizer_params,
    suggest_model_params,
    suggest_augmentation_params,
    # import others...
)

logger = get_logger(__name__)

# Global variable to hold the base config, set by tune_model
BASE_CFG = None

def objective(trial: optuna.trial.Trial) -> float:
    """Optuna objective function."""
    global BASE_CFG
    if BASE_CFG is None:
         raise ValueError("Base config not set. Call tune_model first.")

    # Start with a copy of the base config
    cfg = OmegaConf.structured(OmegaConf.to_yaml(BASE_CFG)) # Deep copy

    # ---- Apply Suggested Parameters ----
    try:
        cfg = suggest_optimizer_params(trial, cfg)
        cfg = suggest_model_params(trial, cfg)
        cfg = suggest_augmentation_params(trial, cfg)
        # Add calls to other suggestion functions (scheduler, loss, etc.)
        # Example: suggest loader params
        # cfg.loader.batch_size = trial.suggest_categorical("loader.batch_size", [32, 64, 128])

        # --- Setup Trial Directory ---
        # Create a unique directory for this trial based on Optuna's trial number
        trial_num = trial.number
        # Assume base experiment dir is set in cfg.paths.experiment_dir
        # This might come from hydra sweep dir in the main tune command
        base_exp_dir = Path(cfg.paths.experiment_dir)
        trial_dir = base_exp_dir / f"trial_{trial_num:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        cfg.paths.experiment_dir = str(trial_dir) # Update config with trial-specific path
        # Potentially update other paths based on this trial_dir if needed
        cfg.paths.checkpoint_dir = str(trial_dir / "checkpoints")
        cfg.paths.reports_dir = str(trial_dir / "reports")
        cfg.paths.logs_dir = str(trial_dir / "logs")

        # Set seed for reproducibility within the trial (can use trial number)
        set_seed(cfg.seed + trial_num)

        logger.info(f"--- Starting Optuna Trial {trial_num} ---")
        logger.info(f"Trial directory: {trial_dir}")
        # Log suggested parameters
        logger.info(f"Suggested params: {trial.params}")

        # ---- Run Training ----
        # 1. DataModule
        datamodule = PlantDiseaseDataModule(cfg=cfg)
        datamodule.prepare_data()
        datamodule.setup(stage='fit') # Need class info before model init

        # Ensure num_classes is set in config if not already present
        if 'num_classes' not in cfg.model:
             cfg.model.num_classes = datamodule.get_num_classes()

        # 2. Model
        ModelClass = get_model_class(cfg.model.name)
        model = ModelClass(**cfg.model) # Pass model config dict

        # 3. Training
        # Use the train_model convenience function
        # It will create the Trainer internally
        results = train_model(
            model=model,
            train_loader=datamodule.train_dataloader(),
            val_loader=datamodule.val_dataloader(),
            cfg=cfg, # Pass the modified config for this trial
            experiment_dir=trial_dir, # Pass trial-specific dir
        )

        # ---- Get Metric to Optimize ----
        # Fetch the metric defined in the main config (e.g., 'val_accuracy' or 'val_loss')
        monitor_metric = cfg.training.get('monitor_metric', 'val_loss') # Default to val_loss
        metric_value = results.get('best_' + monitor_metric, None)

        if metric_value is None:
            # Try fetching from history if best_ isn't populated correctly
            history = results.get('history', {})
            if monitor_metric in history and history[monitor_metric]:
                 monitor_mode = cfg.training.get('monitor_mode', 'min')
                 if monitor_mode == 'min':
                      metric_value = min(history[monitor_metric])
                 else:
                      metric_value = max(history[monitor_metric])
            else:
                 logger.warning(f"Monitor metric '{monitor_metric}' not found in results or history for trial {trial_num}. Returning inf/neginf.")
                 # Return a bad value to penalize this trial
                 return float('inf') if cfg.training.get('monitor_mode', 'min') == 'min' else float('-inf')

        logger.info(f"--- Optuna Trial {trial_num} Finished --- Metric ({monitor_metric}): {metric_value:.6f}")
        return metric_value

    except optuna.exceptions.TrialPruned as e:
         logger.info(f"Trial {trial_num} pruned.")
         raise e
    except Exception as e:
        logger.error(f"Error in Optuna trial {trial_num}: {e}", exc_info=True)
        # Return a bad value to indicate failure
        return float('inf') if cfg.training.get('monitor_mode', 'min') == 'min' else float('-inf')


# This function will be called by cli/main.py:tune
def tune_model(cfg: DictConfig):
    """Sets up and runs the Optuna study."""
    global BASE_CFG
    BASE_CFG = cfg # Store the base config passed from Hydra

    # --- Optuna Study Setup ---
    study_name = cfg.get("optuna", {}).get("study_name", f"{cfg.model.name}-tuning")
    n_trials = cfg.get("optuna", {}).get("n_trials", 50)
    direction = cfg.get("optuna", {}).get("direction", "maximize" if "acc" in cfg.training.monitor_metric else "minimize")
    storage_name = cfg.get("optuna", {}).get("storage", None) # e.g., "sqlite:///optuna_studies.db"
    pruner_config = cfg.get("optuna", {}).get("pruner", {"type": "median", "n_warmup_steps": 5})

    # Setup Pruner
    pruner = None
    if pruner_config.get("type") == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=pruner_config.get("n_startup_trials", 5),
            n_warmup_steps=pruner_config.get("n_warmup_steps", 5),
            interval_steps=pruner_config.get("interval_steps", 1)
        )
    elif pruner_config.get("type") == "hyperband":
         pruner = optuna.pruners.HyperbandPruner(
             min_resource=pruner_config.get("min_resource", 1),
             max_resource=pruner_config.get("max_resource", cfg.training.epochs), # Use epochs as resource
             reduction_factor=pruner_config.get("reduction_factor", 3)
         )
    # Add other pruners if needed

    logger.info(f"Starting Optuna study '{study_name}' for {n_trials} trials.")
    logger.info(f"Optimization direction: {direction}")
    if storage_name: logger.info(f"Using storage: {storage_name}")
    if pruner: logger.info(f"Using pruner: {type(pruner).__name__}")

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=storage_name,
        pruner=pruner,
        load_if_exists=True # Allow resuming studies
    )

    # --- Run Optimization ---
    start_time = time.time()
    study.optimize(objective, n_trials=n_trials, timeout=cfg.get("optuna", {}).get("timeout", None))
    duration = time.time() - start_time

    # --- Log Results ---
    logger.info(f"Optuna study finished in {duration:.2f} seconds.")
    logger.info(f"Number of finished trials: {len(study.trials)}")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"  Value ({cfg.training.monitor_metric}): {study.best_value:.6f}")
    logger.info("  Params: ")
    for key, value in study.best_params.items():
        logger.info(f"    {key}: {value}")

    # Save best params to a file in the main experiment dir (where tune was launched)
    best_params_path = Path(cfg.paths.experiment_dir) / "best_hyperparameters.yaml"
    try:
        OmegaConf.save(config=study.best_params, f=best_params_path)
        logger.info(f"Best hyperparameters saved to {best_params_path}")
    except Exception as e:
         logger.error(f"Could not save best hyperparameters: {e}")

    # Optional: Further analysis/plots using optuna.visualization
    if cfg.get("optuna", {}).get("create_plots", True):
         try:
             if optuna.visualization.is_available():
                 viz_dir = Path(cfg.paths.experiment_dir) / "optuna_plots"
                 viz_dir.mkdir(exist_ok=True)
                 optuna.visualization.plot_optimization_history(study).write_html(viz_dir / "optimization_history.html")
                 optuna.visualization.plot_param_importances(study).write_html(viz_dir / "param_importances.html")
                 # Add more plots like plot_slice, plot_contour if needed
                 logger.info(f"Optuna visualization plots saved to {viz_dir}")
             else:
                  logger.warning("Optuna visualization library not available. Skipping plots.")
         except Exception as e:
              logger.error(f"Failed to generate Optuna plots: {e}")

    return study # Return study object if needed
```

## File: core/tuning/__init__.py

- Extension: .py
- Language: python
- Size: 232 bytes
- Created: 2025-03-28 15:07:28
- Modified: 2025-03-28 15:07:28

### Code

```python

from plantdoc.core.tuning.optuna_runner import objective, tune_model
from plantdoc.core.tuning.search_space import (
    suggest_augmentation_params,
    # import others...
    suggest_model_params,
    suggest_optimizer_params,
)

```

## File: core/tuning/search_space.py

- Extension: .py
- Language: python
- Size: 2859 bytes
- Created: 2025-03-28 15:00:05
- Modified: 2025-03-28 15:00:05

### Code

```python
# search_space.py stub
# core/tuning/search_space.py
import optuna
from omegaconf import DictConfig

def suggest_optimizer_params(trial: optuna.trial.Trial, cfg: DictConfig) -> DictConfig:
    """Suggests optimizer hyperparameters."""
    # Example: Suggest learning rate based on optimizer type
    optimizer_name = cfg.optimizer.name

    if optimizer_name in ["adam", "adamw"]:
        lr = trial.suggest_float("optimizer.lr", 1e-5, 1e-2, log=True)
        beta1 = trial.suggest_float("optimizer.beta1", 0.8, 0.99)
        beta2 = trial.suggest_float("optimizer.beta2", 0.9, 0.9999)
        weight_decay = trial.suggest_float("optimizer.weight_decay", 1e-6, 1e-3, log=True)
        # Update config directly (or return a dict to merge)
        cfg.optimizer.lr = lr
        cfg.optimizer.beta1 = beta1
        cfg.optimizer.beta2 = beta2
        cfg.optimizer.weight_decay = weight_decay # Adjust if using differential LR
    elif optimizer_name == "sgd":
        lr = trial.suggest_float("optimizer.lr", 1e-4, 1e-1, log=True)
        momentum = trial.suggest_float("optimizer.momentum", 0.7, 0.99)
        weight_decay = trial.suggest_float("optimizer.weight_decay", 1e-6, 1e-3, log=True)
        cfg.optimizer.lr = lr
        cfg.optimizer.momentum = momentum
        cfg.optimizer.weight_decay = weight_decay

    # Example: Suggest differential LR settings
    cfg.optimizer.differential_lr = trial.suggest_categorical("optimizer.differential_lr", [True, False])
    if cfg.optimizer.differential_lr:
         cfg.optimizer.differential_lr_factor = trial.suggest_float("optimizer.differential_lr_factor", 0.01, 0.5, log=True)

    return cfg

def suggest_model_params(trial: optuna.trial.Trial, cfg: DictConfig) -> DictConfig:
    """Suggests model hyperparameters."""
    cfg.model.dropout_rate = trial.suggest_float("model.dropout_rate", 0.1, 0.5)
    # Example: Tune CBAM reduction ratio if desired
    # cfg.model.reduction_ratio = trial.suggest_categorical("model.reduction_ratio", [8, 16, 32])
    # Example: Tune head hidden dim if using MLP/Residual head
    if cfg.model.head_type in ["mlp", "residual"]:
        cfg.model.hidden_dim = trial.suggest_categorical("model.hidden_dim", [128, 256, 512])

    return cfg

def suggest_augmentation_params(trial: optuna.trial.Trial, cfg: DictConfig) -> DictConfig:
     """Suggests augmentation hyperparameters."""
     # Example: Tune cutout probability
     if hasattr(cfg.augmentation.train, 'cutout'):
          cfg.augmentation.train.cutout.p = trial.suggest_float("augmentation.train.cutout.p", 0.1, 0.7)
     # Example: Tune rotation limit
     if hasattr(cfg.augmentation.train, 'random_rotate'):
          cfg.augmentation.train.random_rotate = trial.suggest_int("augmentation.train.random_rotate", 10, 45)

     return cfg

# Add more functions as needed for scheduler, loss, data params etc.
```

## File: core/data/transforms.py

- Extension: .py
- Language: python
- Size: 8850 bytes
- Created: 2025-03-28 14:28:00
- Modified: 2025-03-28 14:28:00

### Code

```python
# Path: plantdoc/core/data/transforms.py
# Description: Image transformations using Albumentations library

from typing import Dict, List, Tuple, Union

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig
from plantdoc.utils.logging import get_logger

logger = get_logger(__name__)

def get_transforms(cfg: DictConfig, split: str = "train") -> A.Compose:
    """
    Get image transformations based on configuration and dataset split.

    Args:
        cfg: Hydra configuration object. Expected keys:
             preprocessing:
                 resize: [height, width] for initial resize.
                 center_crop: [height, width] for center cropping.
                 normalize:
                     mean: List of mean values.
                     std: List of std values.
             augmentation:
                 train: (Optional) Contains augmentation parameters for the training split.
                     horizontal_flip: bool
                     vertical_flip: bool
                     random_rotate: int (limit)
                     random_resized_crop:
                         scale: [min_scale, max_scale]
                         ratio: [min_ratio, max_ratio]
                     color_jitter:
                         brightness, contrast, saturation, hue: float limits
                     random_brightness_contrast:
                         brightness_limit, contrast_limit, p: float
                     shift_scale_rotate:
                         shift_limit, scale_limit, rotate_limit, p: float
                     cutout: (CoarseDropout)
                         num_holes, max_h_size, max_w_size, p: int/float
        split: Dataset split ('train', 'val', or 'test').

    Returns:
        An Albumentations composition of transformations.
    """
    try:
        # Basic preprocessing parameters (common to all splits)
        height, width = cfg.preprocessing.resize
        crop_height, crop_width = cfg.preprocessing.center_crop
        mean = cfg.preprocessing.normalize.mean
        std = cfg.preprocessing.normalize.std

        logger.debug(f"Transform params - Resize: ({height}, {width}), Crop: ({crop_height}, {crop_width})")
        logger.debug(f"Transform params - Mean: {mean}, Std: {std}")

        transforms_list = []

        if split == "train" and hasattr(cfg, 'augmentation') and hasattr(cfg.augmentation, 'train'):
            # Training transformations with augmentations
            aug_cfg = cfg.augmentation.train
            logger.info("Applying training augmentations.")

            # RandomResizedCrop is often a primary augmentation
            if hasattr(aug_cfg, 'random_resized_crop'):
                transforms_list.append(
                    A.RandomResizedCrop(
                        height=crop_height, width=crop_width, # Crop to final desired size
                        scale=aug_cfg.random_resized_crop.scale,
                        ratio=aug_cfg.random_resized_crop.ratio,
                        p=1.0, # Usually always applied
                        always_apply=True # Ensure it's always considered
                    )
                )
            else:
                 # Fallback if RandomResizedCrop is not defined: Resize then RandomCrop or CenterCrop
                 transforms_list.append(A.Resize(height=height, width=width))
                 # Add RandomCrop if specified, otherwise CenterCrop later
                 if hasattr(aug_cfg, 'random_crop'):
                     transforms_list.append(A.RandomCrop(height=crop_height, width=crop_width))
                 # If neither RandomResizedCrop nor RandomCrop, CenterCrop will be applied later

            if hasattr(aug_cfg, 'horizontal_flip') and aug_cfg.horizontal_flip:
                transforms_list.append(A.HorizontalFlip(p=0.5))
            if hasattr(aug_cfg, 'vertical_flip') and aug_cfg.vertical_flip:
                transforms_list.append(A.VerticalFlip(p=0.5))
            if hasattr(aug_cfg, 'random_rotate'):
                transforms_list.append(A.Rotate(limit=aug_cfg.random_rotate, p=0.7))
            if hasattr(aug_cfg, 'color_jitter'):
                transforms_list.append(A.ColorJitter(
                    brightness=aug_cfg.color_jitter.brightness,
                    contrast=aug_cfg.color_jitter.contrast,
                    saturation=aug_cfg.color_jitter.saturation,
                    hue=aug_cfg.color_jitter.hue,
                    p=0.7,
                ))
            if hasattr(aug_cfg, 'random_brightness_contrast'):
                 transforms_list.append(A.RandomBrightnessContrast(
                    brightness_limit=aug_cfg.random_brightness_contrast.brightness_limit,
                    contrast_limit=aug_cfg.random_brightness_contrast.contrast_limit,
                    p=aug_cfg.random_brightness_contrast.p,
                ))
            if hasattr(aug_cfg, 'shift_scale_rotate'):
                transforms_list.append(A.ShiftScaleRotate(
                    shift_limit=aug_cfg.shift_scale_rotate.shift_limit,
                    scale_limit=aug_cfg.shift_scale_rotate.scale_limit,
                    rotate_limit=aug_cfg.shift_scale_rotate.rotate_limit,
                    p=aug_cfg.shift_scale_rotate.p,
                ))
            if hasattr(aug_cfg, 'cutout'): # Corresponds to CoarseDropout
                transforms_list.append(A.CoarseDropout(
                    max_holes=aug_cfg.cutout.num_holes,
                    max_height=aug_cfg.cutout.max_h_size,
                    max_width=aug_cfg.cutout.max_w_size,
                    min_holes=1, min_height=8, min_width=8, # Reasonable defaults
                    fill_value=0, # Fill with black
                    p=aug_cfg.cutout.p,
                ))

            # Ensure CenterCrop is applied if RandomResized/RandomCrop wasn't
            if not any(isinstance(t, (A.RandomResizedCrop, A.RandomCrop)) for t in transforms_list):
                 transforms_list.append(A.CenterCrop(height=crop_height, width=crop_width))

        else:
            # Validation and test transformations (deterministic)
            logger.info(f"Applying {split} transformations (Resize -> CenterCrop).")
            transforms_list.extend([
                A.Resize(height=height, width=width),
                A.CenterCrop(height=crop_height, width=crop_width),
            ])

        # Normalization and Tensor conversion (applied to all splits)
        transforms_list.extend([
            A.Normalize(mean=mean, std=std),
            ToTensorV2(), # Converts image to PyTorch tensor (HWC -> CHW) and scales to [0, 1] if needed (already done by Normalize)
        ])

        return A.Compose(transforms_list)

    except KeyError as e:
        logger.error(f"Missing key in configuration for transforms: {e}")
        raise
    except Exception as e:
        logger.error(f"Error creating transforms for split '{split}': {e}")
        raise


class AlbumentationsWrapper:
    """
    Wrapper for Albumentations transforms to seamlessly handle PIL images
    as input, which is common with standard PyTorch datasets like ImageFolder.
    """
    def __init__(self, transform: A.Compose):
        """
        Args:
            transform: An Albumentations Compose object.
        """
        self.transform = transform
        logger.debug("AlbumentationsWrapper initialized.")

    def __call__(self, img: Image.Image) -> torch.Tensor:
        """
        Apply transformations to a PIL image.

        Args:
            img: Input PIL Image.

        Returns:
            Transformed image as a PyTorch tensor.
        """
        try:
            # Convert PIL image to numpy array (HWC format)
            img_np = np.array(img)
            if img_np.ndim == 2: # Handle grayscale images if necessary
                logger.warning("Input image is grayscale, converting to RGB for transform.")
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            elif img_np.shape[2] == 4: # Handle RGBA images
                 logger.warning("Input image has 4 channels (RGBA?), converting to RGB.")
                 img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)

            # Apply Albumentations transformations
            transformed = self.transform(image=img_np)
            # Return the transformed image tensor
            return transformed["image"]
        except Exception as e:
             logger.error(f"Error applying transform in AlbumentationsWrapper: {e}")
             # Return a dummy tensor or re-raise, depending on desired error handling
             # For robustness in dataloader, might return a dummy:
             # return torch.zeros(3, 224, 224) # Adjust size as needed
             raise # Re-raise to make the error visible during debugging
```

## File: core/data/prepare_data.py

- Extension: .py
- Language: python
- Size: 45278 bytes
- Created: 2025-03-28 14:31:36
- Modified: 2025-03-28 14:31:36

### Code

```python
# Path: plantdoc/core/data/prepare_data.py


# Path: plantdoc/core/data/prepare_data.py
# Description: Consolidates dataset validation, analysis, and visualization tasks.

import json
import os
import random
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig
from PIL import Image, UnidentifiedImageError
from plantdoc.core.data.transforms import get_transforms

# Assuming these utilities and transforms are available in your project structure
from plantdoc.utils.logging import get_logger
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Import basic plotting functions if they are separate, otherwise define inline or import
# Example: from plantdoc.utils.visualization import basic_plots, advanced_plots
# For simplicity here, plotting functions are included below where needed.

logger = get_logger(__name__)

# --- Helper Functions (from validation script) ---

def fix_filename(filename: str) -> str:
    """Fix problematic filenames by removing spaces and special characters."""
    clean_name = filename.replace(" ", "_")
    special_chars = r"()[]{}!@#$%^&*+="
    for char in special_chars:
        clean_name = clean_name.replace(char, "")
    while "__" in clean_name:
        clean_name = clean_name.replace("__", "_")
    return clean_name

def check_folder_name(folder_name: str) -> Tuple[bool, str, List[str]]:
    """Check folder name best practices."""
    issues = []
    suggested_name = folder_name
    has_issues = False
    if "," in folder_name:
        has_issues = True
        issues.append("Contains comma")
        suggested_name = suggested_name.replace(",", "")
    if " " in folder_name:
        has_issues = True
        issues.append("Contains spaces")
        suggested_name = suggested_name.replace(" ", "_")
    special_chars = set("!@#$%^&*()+={}[]|\\:;\"'<>?/")
    found_special = [c for c in folder_name if c in special_chars]
    if found_special:
        has_issues = True
        issues.append(f"Contains special characters: {''.join(found_special)}")
        for char in found_special:
            suggested_name = suggested_name.replace(char, "")
    if "__" in folder_name:
        has_issues = True
        issues.append("Contains consecutive underscores")
        while "__" in suggested_name:
            suggested_name = suggested_name.replace("__", "_")
    return has_issues, suggested_name, issues

def validate_image(file_path: Path) -> bool:
    """Verify that an image file can be opened and is valid."""
    try:
        with Image.open(file_path) as img:
            img.verify() # Verify image header and structure
        # Optionally, try to fully load to catch more subtle issues
        # with Image.open(file_path) as img:
        #     img.load()
        return True
    except (UnidentifiedImageError, OSError, IOError, SyntaxError) as e:
        logger.debug(f"Invalid image {file_path}: {e}")
        return False
    except Exception as e:
        logger.warning(f"Unexpected error validating image {file_path}: {e}")
        return False # Treat unexpected errors as invalid

def rename_file_with_unique_name(file_path: Path, new_file_path: Path) -> Path:
    """Rename a file, adding timestamp if the target exists."""
    if new_file_path.exists() and file_path.resolve() != new_file_path.resolve():
        timestamp = int(time.time() * 1000)
        new_file_path = new_file_path.with_name(f"{new_file_path.stem}_{timestamp}{new_file_path.suffix}")
    return new_file_path

# --- Validation Core Function ---

def run_validation(cfg: DictConfig) -> Dict[str, Any]:
    """
    Scan the dataset directory, report/fix issues based on config.

    Args:
        cfg: Hydra configuration object.

    Returns:
        Dictionary containing the scan results.
    """
    prep_cfg = cfg.prepare_data
    data_dir = Path(cfg.paths.raw_dir) # Validate the raw data
    output_dir = Path(prep_cfg.output_dir) / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    dry_run = prep_cfg.dry_run
    fix_extensions = prep_cfg.fix_extensions
    verify_images = prep_cfg.verify_images
    fix_folders = prep_cfg.fix_folders

    logger.info(f"Starting dataset validation in: {data_dir}")
    if dry_run:
        logger.info("DRY RUN MODE: No changes will be made.")

    stats = {
        "total_files": 0, "renamed_files": 0, "problematic_files": 0,
        "class_stats": {}, "extension_stats": Counter(),
        "folder_issues": 0, "folder_suggestions": {},
    }
    valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')

    if not data_dir.is_dir():
        logger.error(f"Data directory not found: {data_dir}")
        return {"error": f"Data directory not found: {data_dir}"}

    class_dirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    class_dirs.sort(key=lambda x: x.name)
    logger.info(f"Found {len(class_dirs)} potential class directories.")

    # --- Folder Name Check ---
    logger.info("Checking folder names...")
    folder_renames = {}
    current_class_dirs = list(class_dirs) # Copy to modify list if renaming
    for i, folder_path in enumerate(class_dirs):
        folder_name = folder_path.name
        has_issues, suggested_name, issues = check_folder_name(folder_name)
        if has_issues:
            stats["folder_issues"] += 1
            stats["folder_suggestions"][folder_name] = {
                "suggested_name": suggested_name, "issues": issues
            }
            if fix_folders and not dry_run:
                new_path = data_dir / suggested_name
                try:
                    if new_path.exists():
                        timestamp = int(time.time() * 1000)
                        suggested_name = f"{suggested_name}_{timestamp}"
                        new_path = data_dir / suggested_name
                    logger.info(f"Renaming folder: {folder_path} -> {new_path}")
                    folder_path.rename(new_path)
                    folder_renames[folder_name] = suggested_name
                    current_class_dirs[i] = new_path # Update path in the list for file iteration
                except Exception as e:
                    logger.error(f"Error renaming folder {folder_name} to {suggested_name}: {e}")
            else:
                 logger.warning(f"Folder '{folder_name}' has issues: {issues}. Suggested: '{suggested_name}'")

    # --- File Processing ---
    logger.info("Processing files within class directories...")
    for class_dir_path in tqdm(current_class_dirs, desc="Processing classes"):
        class_name = class_dir_path.name
        stats["class_stats"][class_name] = {"total": 0, "renamed": 0, "problematic": 0}
        class_file_count = 0
        class_renamed_count = 0
        class_problem_count = 0

        try:
            for file_path in class_dir_path.iterdir():
                if file_path.is_file():
                    stats["total_files"] += 1
                    class_file_count += 1
                    filename = file_path.name
                    ext = file_path.suffix.lower()
                    stats["extension_stats"][ext] += 1

                    is_valid_ext = ext in valid_extensions
                    if not is_valid_ext:
                        logger.debug(f"Skipping non-image file: {file_path}")
                        continue # Skip non-image files entirely

                    # Image Verification (Do this BEFORE potential rename)
                    is_problematic = False
                    if verify_images:
                        if not validate_image(file_path):
                            logger.warning(f"Problematic image detected: {file_path}")
                            stats["problematic_files"] += 1
                            class_problem_count += 1
                            is_problematic = True
                            # Decide whether to skip fixing problematic files
                            # continue # Uncomment to skip fixing problematic files

                    # File Renaming Logic
                    has_spaces_or_special = any(c in filename for c in " ()[]{}!@#$%^&*+=")
                    needs_ext_fix = ext != ".jpg"
                    needs_rename = has_spaces_or_special or needs_ext_fix

                    if needs_rename and fix_extensions:
                        if is_problematic and not dry_run:
                            logger.warning(f"Skipping rename for problematic file: {file_path}")
                            continue

                        base_name = file_path.stem
                        if has_spaces_or_special:
                            base_name = fix_filename(base_name)

                        new_filename = f"{base_name}.jpg"
                        new_file_path = class_dir_path / new_filename
                        new_file_path = rename_file_with_unique_name(file_path, new_file_path)

                        if file_path != new_file_path:
                            if not dry_run:
                                try:
                                    # Use copy and remove for potentially safer operation across filesystems
                                    shutil.copy2(file_path, new_file_path)
                                    file_path.unlink()
                                    logger.debug(f"Renamed: {filename} -> {new_file_path.name}")
                                    stats["renamed_files"] += 1
                                    class_renamed_count += 1
                                except Exception as e:
                                    logger.error(f"Error renaming {file_path} to {new_file_path}: {e}")
                            else:
                                logger.info(f"[Dry Run] Would rename: {filename} -> {new_file_path.name}")
                                stats["renamed_files"] += 1
                                class_renamed_count += 1

        except Exception as e:
            logger.error(f"Error processing directory {class_dir_path}: {e}")
            continue # Skip to the next class directory

        stats["class_stats"][class_name]["total"] = class_file_count
        stats["class_stats"][class_name]["renamed"] = class_renamed_count
        stats["class_stats"][class_name]["problematic"] = class_problem_count

    # --- Reporting ---
    report_path = output_dir / "validation_report.json"
    logger.info(f"Saving validation report to {report_path}")
    try:
        with open(report_path, "w") as f:
            # Convert Counter to dict for JSON serialization
            stats["extension_stats"] = dict(stats["extension_stats"])
            json.dump(stats, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save validation report: {e}")

    logger.info("--- Validation Summary ---")
    logger.info(f"Total Classes Found: {len(current_class_dirs)}")
    logger.info(f"Total Files Scanned: {stats['total_files']}")
    logger.info(f"Files Flagged for Rename: {stats['renamed_files']}")
    logger.info(f"Problematic/Invalid Images: {stats['problematic_files']}")
    logger.info(f"Folders with Naming Issues: {stats['folder_issues']}")
    logger.info("Extension Counts:")
    for ext, count in sorted(stats["extension_stats"].items()):
         logger.info(f"  {ext}: {count}")
    if dry_run and (stats['renamed_files'] > 0 or (stats['folder_issues'] > 0 and fix_folders)):
        logger.info("Run with prepare_data.dry_run=False to apply fixes.")
    logger.info("--- Validation Complete ---")

    return stats


# --- Helper Functions (from analysis script) ---

def extract_image_metadata(image_path: Path) -> Optional[Dict[str, Any]]:
    """Extract metadata from an image file."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            aspect_ratio = width / height if height > 0 else 0
            img_format = img.format
            img_mode = img.mode
            try:
                file_size_kb = image_path.stat().st_size / 1024
            except Exception:
                 file_size_kb = -1 # Indicate error

            # Basic color info (requires loading image data)
            try:
                img_array = np.array(img.convert("RGB"))
                mean_rgb = np.mean(img_array, axis=(0, 1))
                std_rgb = np.std(img_array, axis=(0, 1))
                brightness = mean_rgb.mean() # Simple brightness estimate
            except Exception as e:
                 logger.debug(f"Could not get color info for {image_path}: {e}")
                 mean_rgb = [-1,-1,-1]
                 std_rgb = [-1,-1,-1]
                 brightness = -1

            return {
                "path": str(image_path), "class": image_path.parent.name,
                "width": width, "height": height, "aspect_ratio": aspect_ratio,
                "format": img_format, "mode": img_mode, "file_size_kb": file_size_kb,
                "mean_r": mean_rgb[0], "mean_g": mean_rgb[1], "mean_b": mean_rgb[2],
                "std_r": std_rgb[0], "std_g": std_rgb[1], "std_b": std_rgb[2],
                "brightness": brightness,
            }
    except Exception as e:
        logger.error(f"Error processing metadata for {image_path}: {e}")
        return None

def create_summary_report(
    df: pd.DataFrame, class_counts: Dict[str, int], file_extensions: Dict[str, int], output_dir: Path
) -> None:
    """Generate a summary report (Markdown) of dataset statistics."""
    report_path = output_dir / "analysis_summary_report.md"
    logger.info(f"Generating analysis summary report: {report_path}")
    total_images = sum(class_counts.values())

    with open(report_path, "w") as f:
        f.write("# Dataset Analysis Report\n\n")
        f.write("## Overview\n\n")
        f.write(f"- **Total Classes:** {len(class_counts)}\n")
        f.write(f"- **Total Images:** {total_images:,}\n")
        if not df.empty:
            f.write(f"- **Images Analyzed (Sample):** {len(df):,}\n")

        f.write("\n### File Formats (Overall)\n\n")
        total_ext = sum(file_extensions.values())
        for ext, count in sorted(file_extensions.items()):
            perc = count / total_ext * 100 if total_ext > 0 else 0
            f.write(f"- **{ext}:** {count:,} ({perc:.1f}%)\n")

        f.write("\n## Class Distribution\n\n")
        f.write("| Class | Image Count | Percentage |\n")
        f.write("|---|---|---|\n")
        for class_name, count in sorted(class_counts.items(), key=lambda item: item[1], reverse=True):
            perc = count / total_images * 100 if total_images > 0 else 0
            f.write(f"| {class_name} | {count:,} | {perc:.2f}% |\n")

        if not df.empty:
            f.write("\n## Image Properties (Sampled)\n\n")
            f.write("### Dimensions\n")
            f.write(f"- **Width:** Min={df['width'].min():,}, Median={df['width'].median():,.0f}, Max={df['width'].max():,}\n")
            f.write(f"- **Height:** Min={df['height'].min():,}, Median={df['height'].median():,.0f}, Max={df['height'].max():,}\n")
            f.write(f"- **Aspect Ratio:** Min={df['aspect_ratio'].min():.2f}, Median={df['aspect_ratio'].median():.2f}, Max={df['aspect_ratio'].max():.2f}\n")

            f.write("\n### File Size\n")
            f.write(f"- **Size (KB):** Min={df['file_size_kb'].min():.2f}, Median={df['file_size_kb'].median():.2f}, Max={df['file_size_kb'].max():.2f}\n")

            f.write("\n### Color & Brightness\n")
            f.write(f"- **Avg RGB:** ({df['mean_r'].mean():.1f}, {df['mean_g'].mean():.1f}, {df['mean_b'].mean():.1f})\n")
            f.write(f"- **Avg Brightness:** {df['brightness'].mean():.1f}\n")

    logger.info("Analysis summary report generated.")

# --- Analysis Core Function ---

def run_analysis(cfg: DictConfig) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Analyze the dataset, generate statistics, and optional plots.

    Args:
        cfg: Hydra configuration object.

    Returns:
        Tuple containing:
            - DataFrame with image metadata (or None if error).
            - Dictionary with dataset statistics.
    """
    prep_cfg = cfg.prepare_data
    data_dir = Path(cfg.paths.raw_dir) # Analyze the raw (potentially fixed) data
    output_dir = Path(prep_cfg.output_dir) / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_size = prep_cfg.sample_size
    random_seed = prep_cfg.random_seed
    create_plots = prep_cfg.create_plots
    plots_output_dir = output_dir / "plots"
    if create_plots:
        plots_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting dataset analysis for: {data_dir}")
    random.seed(random_seed)
    np.random.seed(random_seed)

    # --- Collect Basic Stats & File Paths ---
    class_counts = {}
    file_extensions = Counter()
    all_image_paths = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')

    try:
        class_dirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        class_dirs.sort(key=lambda x: x.name)
        logger.info(f"Found {len(class_dirs)} classes for analysis.")

        for class_dir in tqdm(class_dirs, desc="Scanning for analysis"):
            count = 0
            for item in class_dir.iterdir():
                if item.is_file() and item.suffix.lower() in valid_extensions:
                    all_image_paths.append(item)
                    file_extensions[item.suffix.lower()] += 1
                    count += 1
            class_counts[class_dir.name] = count

    except Exception as e:
         logger.error(f"Error scanning directories for analysis: {e}")
         return None, {"error": "Failed to scan directories."}

    total_images = len(all_image_paths)
    logger.info(f"Found {total_images} total image files for potential analysis.")
    if total_images == 0:
        logger.warning("No images found to analyze.")
        return pd.DataFrame(), {"num_classes": len(class_counts), "total_images": 0}

    # --- Sample Images for Detailed Metadata Extraction ---
    if total_images > sample_size:
        logger.info(f"Sampling {sample_size} images for detailed analysis (seed={random_seed}).")
        sampled_paths = random.sample(all_image_paths, sample_size)
    else:
        logger.info(f"Analyzing all {total_images} images.")
        sampled_paths = all_image_paths

    # --- Extract Metadata ---
    metadata_list = []
    logger.info(f"Extracting metadata from {len(sampled_paths)} images...")
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_path = {executor.submit(extract_image_metadata, path): path for path in sampled_paths}
        for future in tqdm(as_completed(future_to_path), total=len(sampled_paths), desc="Extracting metadata"):
            result = future.result()
            if result:
                metadata_list.append(result)

    if not metadata_list:
         logger.warning("No metadata could be extracted from sampled images.")
         df = pd.DataFrame()
    else:
        df = pd.DataFrame(metadata_list)
        metadata_csv_path = output_dir / "image_metadata_sample.csv"
        try:
            df.to_csv(metadata_csv_path, index=False)
            logger.info(f"Saved sampled image metadata to {metadata_csv_path}")
        except Exception as e:
            logger.error(f"Failed to save metadata CSV: {e}")


    # --- Calculate Overall Statistics ---
    dataset_stats = {
        "num_classes": len(class_counts),
        "total_images": total_images,
        "class_counts": class_counts,
        "file_extensions": dict(file_extensions),
        "sampled_images_analyzed": len(df),
        # Add stats derived from the sampled DataFrame
        "image_dimensions_stats (sample)": {
            "min_width": int(df["width"].min()) if not df.empty else None,
            "max_width": int(df["width"].max()) if not df.empty else None,
            "median_width": float(df["width"].median()) if not df.empty else None,
            "min_height": int(df["height"].min()) if not df.empty else None,
            "max_height": int(df["height"].max()) if not df.empty else None,
            "median_height": float(df["height"].median()) if not df.empty else None,
        },
         "aspect_ratio_stats (sample)": {
            "min": float(df["aspect_ratio"].min()) if not df.empty else None,
            "max": float(df["aspect_ratio"].max()) if not df.empty else None,
            "median": float(df["aspect_ratio"].median()) if not df.empty else None,
        },
        "file_size_stats (sample)": {
            "min_kb": float(df["file_size_kb"].min()) if not df.empty else None,
            "max_kb": float(df["file_size_kb"].max()) if not df.empty else None,
            "median_kb": float(df["file_size_kb"].median()) if not df.empty else None,
        },
        "color_info_stats (sample)": {
            "avg_mean_rgb": [df['mean_r'].mean(), df['mean_g'].mean(), df['mean_b'].mean()] if not df.empty else None,
            "avg_std_rgb": [df['std_r'].mean(), df['std_g'].mean(), df['std_b'].mean()] if not df.empty else None,
            "avg_brightness": float(df["brightness"].mean()) if not df.empty else None,
        },
    }

    # Save statistics
    stats_path = output_dir / "dataset_analysis_stats.json"
    logger.info(f"Saving analysis statistics to {stats_path}")
    try:
        with open(stats_path, "w") as f:
            # Convert numpy types for JSON
            sanitized_stats = json.loads(json.dumps(dataset_stats, default=lambda x: int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else str(x)))
            json.dump(sanitized_stats, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save analysis stats JSON: {e}")

    # Generate Markdown Report
    create_summary_report(df, class_counts, file_extensions, output_dir)

    # --- Generate Plots (if enabled) ---
    if create_plots and not df.empty:
        logger.info(f"Generating analysis plots in {plots_output_dir}")
        try:
            # Plot Class Distribution
            plt.figure(figsize=(max(12, len(class_counts) * 0.5), 8))
            sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()), palette="viridis")
            plt.xticks(rotation=90, fontsize=10)
            plt.title("Class Distribution")
            plt.ylabel("Number of Images")
            plt.tight_layout()
            plt.savefig(plots_output_dir / "class_distribution.png", dpi=150)
            plt.close()

            # Plot Image Dimensions (Width vs Height)
            plt.figure(figsize=(10, 8))
            sns.scatterplot(data=df, x="width", y="height", hue="class", alpha=0.6, legend=False, s=20)
            plt.title("Image Dimensions (Width vs Height) - Sampled")
            plt.xlabel("Width (pixels)")
            plt.ylabel("Height (pixels)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_output_dir / "dimensions_scatter.png", dpi=150)
            plt.close()

            # Plot Aspect Ratio Distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(df["aspect_ratio"], kde=True, bins=50)
            plt.title("Aspect Ratio Distribution - Sampled")
            plt.xlabel("Aspect Ratio (Width / Height)")
            plt.tight_layout()
            plt.savefig(plots_output_dir / "aspect_ratio_hist.png", dpi=150)
            plt.close()

            # Plot File Size Distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(df["file_size_kb"], kde=True, bins=50)
            plt.title("File Size Distribution (KB) - Sampled")
            plt.xlabel("File Size (KB)")
            plt.tight_layout()
            plt.savefig(plots_output_dir / "file_size_hist.png", dpi=150)
            plt.close()

            logger.info("Analysis plots generated.")

        except Exception as e:
            logger.error(f"Error generating analysis plots: {e}", exc_info=True)

    logger.info("--- Dataset Analysis Complete ---")
    return df, dataset_stats


# --- Helper Functions (from visualization script) ---

def extract_features_for_viz(image_path: Path, target_size=(224, 224)) -> Optional[Tuple[List[float], str]]:
    """Extract simple features from an image for visualization (t-SNE, clustering)."""
    try:
        img = cv2.imread(str(image_path))
        if img is None: return None, None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)

        features = []
        # Color stats
        for i in range(3):
            features.extend([np.mean(img[:, :, i]), np.std(img[:, :, i])])
        # Gradient stats
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        features.extend([np.mean(magnitude), np.std(magnitude)])
        # Simple histogram
        hist = cv2.calcHist([gray], [0], None, [8], [0, 256])
        features.extend( (hist.flatten() / (hist.sum() + 1e-6)).tolist() ) # Normalize safely

        return features, image_path.parent.name
    except Exception as e:
        logger.error(f"Error extracting features for viz from {image_path}: {e}")
        return None, None

def sample_images_for_viz(data_dir: Path, n_per_class: int = 30, max_classes: Optional[int] = None, random_seed: int = 42) -> Tuple[List[Path], List[str]]:
    """Sample images for visualization tasks like t-SNE."""
    random.seed(random_seed)
    all_image_paths = []
    labels_list = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')

    class_dirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    class_dirs.sort(key=lambda x: x.name)

    if max_classes and len(class_dirs) > max_classes:
        logger.info(f"Sampling {max_classes} classes out of {len(class_dirs)} for visualization.")
        class_dirs = random.sample(class_dirs, max_classes)

    logger.info(f"Sampling up to {n_per_class} images per class for visualization...")
    for class_dir in class_dirs:
        image_files = [item for item in class_dir.iterdir() if item.is_file() and item.suffix.lower() in valid_extensions]
        sampled_images = random.sample(image_files, min(n_per_class, len(image_files)))
        all_image_paths.extend(sampled_images)
        labels_list.extend([class_dir.name] * len(sampled_images))

    logger.info(f"Sampled {len(all_image_paths)} images from {len(class_dirs)} classes for visualization.")
    return all_image_paths, labels_list


# --- Visualization Core Function ---

def run_visualization(cfg: DictConfig):
    """
    Create advanced visualizations for the dataset.

    Args:
        cfg: Hydra configuration object.
    """
    prep_cfg = cfg.prepare_data
    data_dir = Path(cfg.paths.raw_dir)
    output_dir = Path(prep_cfg.output_dir) / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sample size for t-SNE/clustering often different from metadata analysis
    # Use a specific config or reuse sample_size
    n_per_class_viz = prep_cfg.get("n_per_class_viz", 30)
    max_classes_viz = prep_cfg.get("max_classes_viz", None) # Limit classes for viz if needed
    random_seed = prep_cfg.random_seed

    logger.info(f"Starting dataset visualization generation in: {output_dir}")
    logger.info(f"Sampling: {n_per_class_viz} images/class, Max classes: {max_classes_viz}, Seed: {random_seed}")

    # --- Sample Images and Extract Features ---
    sampled_paths, sampled_labels = sample_images_for_viz(
        data_dir, n_per_class=n_per_class_viz, max_classes=max_classes_viz, random_seed=random_seed
    )

    if not sampled_paths:
        logger.warning("No images sampled for visualization. Skipping feature extraction and related plots.")
        features = None
        labels = None
    else:
        logger.info("Extracting features for visualization...")
        features_list = []
        labels_list = []
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_path = {executor.submit(extract_features_for_viz, path): path for path in sampled_paths}
            for future in tqdm(as_completed(future_to_path), total=len(sampled_paths), desc="Extracting viz features"):
                feature_set, class_name = future.result()
                if feature_set is not None:
                    features_list.append(feature_set)
                    labels_list.append(class_name)

        if not features_list:
             logger.warning("Feature extraction for visualization failed. Skipping related plots.")
             features = None
             labels = None
        else:
            features = np.array(features_list)
            labels = np.array(labels_list)
            logger.info(f"Extracted features matrix shape: {features.shape}")

    # --- Generate Visualizations ---
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("paper", font_scale=1.5)

    # 1. Image Grid
    try:
        logger.info("Generating image grid...")
        n_grid_classes = min(len(np.unique(sampled_labels)) if labels is not None else 10, 10)
        create_image_grid_viz(data_dir, output_dir, n_per_class=5, max_classes=n_grid_classes)
    except Exception as e:
        logger.error(f"Failed to create image grid: {e}", exc_info=True)

    # 2. t-SNE Plot (if features extracted)
    if features is not None and labels is not None:
        try:
             logger.info("Generating t-SNE plot...")
             create_tsne_viz(features, labels, output_dir, random_seed)
        except Exception as e:
             logger.error(f"Failed to create t-SNE plot: {e}", exc_info=True)

        # 3. Hierarchical Clustering (if features extracted)
        try:
            logger.info("Generating hierarchical clustering plot...")
            create_hierarchical_clustering_viz(features, labels, output_dir)
        except Exception as e:
            logger.error(f"Failed to create hierarchical clustering plot: {e}", exc_info=True)

        # 4. Similarity Matrix (if features extracted)
        try:
            logger.info("Generating class similarity matrix...")
            create_similarity_matrix_viz(features, labels, output_dir)
        except Exception as e:
            logger.error(f"Failed to create similarity matrix: {e}", exc_info=True)

    # 5. Augmentation Visualization
    try:
        logger.info("Generating augmentation visualization...")
        # Need the augmentation config part from main cfg
        create_augmentation_viz(cfg, data_dir, output_dir, num_samples=3, random_seed=random_seed)
    except Exception as e:
        logger.error(f"Failed to create augmentation visualization: {e}", exc_info=True)

    logger.info(f"--- Visualization Complete --- Plots saved in {output_dir}")


# --- Visualization Plotting Functions (Internal Helpers for run_visualization) ---

def create_image_grid_viz(data_dir: Path, output_dir: Path, n_per_class: int = 5, max_classes: int = 10):
    """Internal helper to create image grid plot."""
    class_dirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    class_dirs.sort(key=lambda x: x.name)
    if len(class_dirs) > max_classes: class_dirs = random.sample(class_dirs, max_classes)
    if not class_dirs: return

    fig = plt.figure(figsize=(n_per_class * 2.5, len(class_dirs) * 2.5))
    valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')

    for i, class_dir in enumerate(class_dirs):
        image_files = [item for item in class_dir.iterdir() if item.is_file() and item.suffix.lower() in valid_extensions]
        selected_images = random.sample(image_files, min(n_per_class, len(image_files)))

        for j, img_path in enumerate(selected_images):
            ax = fig.add_subplot(len(class_dirs), n_per_class, i * n_per_class + j + 1)
            try:
                img = Image.open(img_path)
                ax.imshow(np.array(img))
            except Exception as e:
                 logger.warning(f"Could not load image {img_path} for grid: {e}")
                 ax.text(0.5, 0.5, 'Error', horizontalalignment='center', verticalalignment='center')
            ax.axis("off")
            if j == 0:
                 simple_class = class_dir.name.replace("___", " - ").replace("_", " ")[:25]
                 ax.set_ylabel(simple_class, fontsize=10, rotation=0, labelpad=40, va='center', ha='right')

    plt.suptitle("Sample Images by Class", fontsize=16, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(output_dir / "viz_image_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

def create_tsne_viz(features: np.ndarray, labels: np.ndarray, output_dir: Path, random_seed: int):
    """Internal helper to create t-SNE plot."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # Optional PCA pre-reduction
    n_pca = min(50, X_scaled.shape[0] - 1, X_scaled.shape[1])
    if n_pca > 2:
        logger.debug(f"Applying PCA with {n_pca} components before t-SNE.")
        pca = PCA(n_components=n_pca, random_state=random_seed)
        X_pca = pca.fit_transform(X_scaled)
    else:
        X_pca = X_scaled # Skip PCA if too few samples/features

    logger.info("Running t-SNE...")
    perplexity_val = min(30, X_pca.shape[0] - 1)
    if perplexity_val <= 1:
        logger.warning(f"Perplexity ({perplexity_val}) too low for t-SNE, skipping plot.")
        return

    tsne = TSNE(n_components=2, random_state=random_seed, perplexity=perplexity_val, n_iter=300)
    X_tsne = tsne.fit_transform(X_pca)

    df_tsne = pd.DataFrame({"x": X_tsne[:, 0], "y": X_tsne[:, 1], "class": labels})
    unique_classes = sorted(df_tsne["class"].unique())

    plt.figure(figsize=(14, 10))
    colors = sns.color_palette("viridis", len(unique_classes))
    class_to_color = dict(zip(unique_classes, colors))

    for class_name, group in df_tsne.groupby("class"):
        simple_name = class_name.replace("___", "-").replace("_", " ")[:30]
        plt.scatter(group["x"], group["y"], label=simple_name, color=class_to_color[class_name], alpha=0.7, s=50)

    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.title("t-SNE Visualization of Image Features (Sampled)", fontsize=16)
    plt.xlabel("t-SNE Feature 1")
    plt.ylabel("t-SNE Feature 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust for legend
    plt.savefig(output_dir / "viz_tsne.png", dpi=150, bbox_inches="tight")
    plt.close()

def create_hierarchical_clustering_viz(features: np.ndarray, labels: np.ndarray, output_dir: Path):
    """Internal helper to create hierarchical clustering plot."""
    class_features = {}
    for i, label in enumerate(labels):
        class_features.setdefault(label, []).append(features[i])
    class_mean_features = {cls: np.mean(feats, axis=0) for cls, feats in class_features.items() if feats}
    if not class_mean_features: return

    class_names = list(class_mean_features.keys())
    feature_matrix = np.array([class_mean_features[cls] for cls in class_names])

    try:
        linked = linkage(feature_matrix, "ward")
    except Exception as e:
        logger.error(f"Hierarchical clustering linkage failed: {e}. Skipping plot.")
        return

    plt.figure(figsize=(12, max(8, len(class_names) * 0.3)))
    simple_class_names = [name.replace("___", "-").replace("_", " ")[:40] for name in class_names]
    dendrogram(linked, orientation="right", labels=simple_class_names, leaf_font_size=10)
    plt.title("Hierarchical Clustering of Class Mean Features", fontsize=16)
    plt.xlabel("Distance")
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "viz_hierarchical_clustering.png", dpi=150, bbox_inches="tight")
    plt.close()

def create_similarity_matrix_viz(features: np.ndarray, labels: np.ndarray, output_dir: Path):
    """Internal helper to create similarity matrix plot."""
    class_features = {}
    for i, label in enumerate(labels):
        class_features.setdefault(label, []).append(features[i])
    class_mean_features = {cls: np.mean(feats, axis=0) for cls, feats in class_features.items() if feats}
    if not class_mean_features: return

    class_names = list(class_mean_features.keys())
    feature_matrix = np.array([class_mean_features[cls] for cls in class_names])

    similarity_matrix = cosine_similarity(feature_matrix)
    simple_class_names = [name.replace("___", "-").replace("_", " ")[:30] for name in class_names]
    df_similarity = pd.DataFrame(similarity_matrix, index=simple_class_names, columns=simple_class_names)

    plt.figure(figsize=(max(12, len(class_names)*0.5), max(10, len(class_names)*0.5)))
    sns.heatmap(df_similarity, cmap="viridis", annot=False, fmt=".2f", linewidths=.5, square=True)
    plt.title("Class Similarity Matrix (Cosine Similarity of Mean Features)", fontsize=14)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "viz_similarity_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

def create_augmentation_viz(cfg: DictConfig, data_dir: Path, output_dir: Path, num_samples: int = 3, random_seed: int = 42):
    """Internal helper to visualize augmentations."""
    random.seed(random_seed)
    # Use get_transforms to get the configured training augmentations
    try:
        # Pass only relevant parts of config if needed, or full cfg if get_transforms handles it
        train_transforms = get_transforms(cfg, split='train')
        # Extract individual transforms for display (this might be tricky depending on Compose structure)
        # Or define a fixed set of augmentations to showcase here:
        showcase_augmentations = {
            "Original": A.Compose([A.Resize(256, 256)]), # Just resize
            "HorizontalFlip": A.Compose([A.Resize(256, 256), A.HorizontalFlip(p=1.0)]),
            "Rotate": A.Compose([A.Resize(256, 256), A.Rotate(limit=45, p=1.0)]),
            "BrightnessContrast": A.Compose([A.Resize(256, 256), A.RandomBrightnessContrast(p=1.0)]),
            "ShiftScaleRotate": A.Compose([A.Resize(256, 256), A.ShiftScaleRotate(p=1.0)]),
            "CoarseDropout": A.Compose([A.Resize(256, 256), A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=1.0)])
        }
        num_augments = len(showcase_augmentations)
    except Exception as e:
        logger.error(f"Could not get/define transforms for augmentation viz: {e}. Skipping.")
        return

    class_dirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not class_dirs: return
    selected_classes = random.sample(class_dirs, min(num_samples, len(class_dirs)))
    valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')

    fig, axes = plt.subplots(len(selected_classes), num_augments, figsize=(num_augments * 2.5, len(selected_classes) * 2.5))
    if len(selected_classes) == 1: axes = np.expand_dims(axes, axis=0) # Handle single sample case

    for i, class_dir in enumerate(selected_classes):
        image_files = [item for item in class_dir.iterdir() if item.is_file() and item.suffix.lower() in valid_extensions]
        if not image_files: continue
        img_path = random.choice(image_files)

        try:
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.warning(f"Could not load image {img_path} for augmentation viz: {e}")
            continue

        for j, (aug_name, aug_func) in enumerate(showcase_augmentations.items()):
            ax = axes[i, j]
            try:
                augmented = aug_func(image=image)["image"]
                ax.imshow(augmented)
            except Exception as e:
                logger.warning(f"Error applying {aug_name} to {img_path}: {e}")
                ax.imshow(image) # Show original on error
                ax.set_title(f"{aug_name}\n(Error)", fontsize=8, color='red')
            ax.axis("off")
            if i == 0: ax.set_title(aug_name, fontsize=10)
            if j == 0: ax.set_ylabel(class_dir.name[:20], fontsize=10, rotation=0, labelpad=40, va='center', ha='right')

    plt.suptitle("Sample Data Augmentations", fontsize=16, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    plt.savefig(output_dir / "viz_augmentations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --- Main Orchestration Function ---

def run_prepare_data(cfg: DictConfig):
    """
    Runs the full data preparation pipeline: validation, analysis, visualization.

    Args:
        cfg: Hydra configuration object.
    """
    start_time = time.time()
    logger.info("===== Starting Data Preparation Pipeline =====")

    prep_cfg = cfg.prepare_data
    main_output_dir = Path(prep_cfg.output_dir)
    main_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Main output directory: {main_output_dir}")

    # --- Step 1: Validation ---
    logger.info("--- Running Step 1: Dataset Validation ---")
    validation_results = run_validation(cfg)
    # Check for critical errors from validation if needed

    # --- Step 2: Analysis (Conditional) ---
    analysis_results_df = None
    analysis_stats = {}
    if prep_cfg.run_analysis_after_validation:
        logger.info("--- Running Step 2: Dataset Analysis ---")
        analysis_results_df, analysis_stats = run_analysis(cfg)
    else:
        logger.info("--- Skipping Step 2: Dataset Analysis (run_analysis_after_validation=False) ---")

    # --- Step 3: Visualization (Conditional) ---
    if prep_cfg.run_visualization_after_analysis:
        if analysis_stats.get("error"):
             logger.warning("Skipping visualization because analysis failed.")
        else:
            logger.info("--- Running Step 3: Dataset Visualization ---")
            run_visualization(cfg) # Pass df/stats if needed by viz funcs
    else:
        logger.info("--- Skipping Step 3: Dataset Visualization (run_visualization_after_analysis=False) ---")

    # --- Step 4: Combined Report (Optional) ---
    if prep_cfg.generate_combined_report:
        logger.info("--- Running Step 4: Generating Combined Report (Placeholder) ---")
        # TODO: Implement combined report generation
        # - Could involve merging the validation JSON, analysis JSON/Markdown,
        #   and adding links/references to the visualization PNGs.
        # - Example: Create a main README.md in main_output_dir summarizing all findings.
        combined_report_path = main_output_dir / "DATA_PREPARATION_REPORT.md"
        try:
            with open(combined_report_path, "w") as f:
                f.write("# Data Preparation Pipeline Report\n\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Data Source: {cfg.paths.raw_dir}\n\n")
                f.write("## Summary\n\n")
                if validation_results.get("error"): f.write("- Validation: FAILED\n")
                else: f.write("- Validation: COMPLETED\n")
                if prep_cfg.run_analysis_after_validation:
                    if analysis_stats.get("error"): f.write("- Analysis: FAILED\n")
                    else: f.write("- Analysis: COMPLETED\n")
                if prep_cfg.run_visualization_after_analysis and not analysis_stats.get("error"):
                     f.write("- Visualization: COMPLETED\n")

                f.write("\n## Details\n\n")
                f.write("- **Validation Report:** See `validation/validation_report.json`\n")
                if prep_cfg.run_analysis_after_validation and not analysis_stats.get("error"):
                     f.write("- **Analysis Stats:** See `analysis/dataset_analysis_stats.json`\n")
                     f.write("- **Analysis Summary:** See `analysis/analysis_summary_report.md`\n")
                     if prep_cfg.create_plots: f.write("- **Analysis Plots:** See `analysis/plots/`\n")
                if prep_cfg.run_visualization_after_analysis and not analysis_stats.get("error"):
                     f.write("- **Visualizations:** See `visualizations/`\n")
            logger.info(f"Generated combined report stub: {combined_report_path}")
        except Exception as e:
            logger.error(f"Failed to generate combined report: {e}")


    elapsed_time = time.time() - start_time
    logger.info(f"===== Data Preparation Pipeline Finished in {elapsed_time:.2f} seconds =====")


```

## File: core/data/datasets.py

- Extension: .py
- Language: python
- Size: 9107 bytes
- Created: 2025-03-28 14:28:05
- Modified: 2025-03-28 14:28:05

### Code

```python
# Path: plantdoc/core/data/dataset.py
# Description: Dataset class for plant disease classification

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from plantdoc.utils.logging import get_logger
from torch.utils.data import Dataset
from tqdm import tqdm

logger = get_logger(__name__)


class PlantDiseaseDataset(Dataset):
    """
    Dataset class for plant disease classification.

    Loads images from a directory structure where each subdirectory
    represents a class.
    """

    def __init__(
        self,
        data_dir: str,
        transform: Optional[callable] = None,
        split: str = "train",
        classes: Optional[List[str]] = None,
        image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'),
    ):
        """
        Initialize the PlantDiseaseDataset.

        Args:
            data_dir: Path to the dataset directory. For split='all', this should be
                      the root directory containing class subfolders. For specific splits
                      like 'train', 'val', 'test', this should point to the respective
                      split folder (e.g., 'data/raw/train').
            transform: Transformations callable (e.g., from Albumentations) to apply to the images.
            split: Dataset split identifier ('train', 'val', 'test', 'all'). Used mainly for logging.
            classes: Optional list of class names. If None, they will be inferred from the
                     subdirectory names in `data_dir`. Useful for ensuring consistent class ordering.
            image_extensions: Tuple of valid image file extensions (case-insensitive).
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        self.image_extensions = image_extensions
        self.samples: List[Tuple[str, int]] = [] # List of (image_path, class_idx)
        self.class_to_idx: Dict[str, int] = {}
        self.classes: List[str] = []

        if not self.data_dir.is_dir():
            logger.error(f"Dataset directory not found or is not a directory: {self.data_dir}")
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")

        # Determine class names and mapping
        self._find_classes(classes)

        # Load image paths and labels
        self._load_samples()

        if not self.samples:
            logger.warning(f"No valid image samples found in {self.data_dir} for split '{self.split}'.")
        else:
             logger.info(f"Initialized dataset for split '{self.split}' with {len(self.samples)} samples from {len(self.classes)} classes.")

    def _find_classes(self, provided_classes: Optional[List[str]]):
        """Find classes based on subdirectories or use provided list."""
        if provided_classes is not None:
            logger.info("Using provided class list.")
            self.classes = sorted(provided_classes) # Ensure consistent order
        else:
            logger.info(f"Inferring classes from subdirectories in {self.data_dir}.")
            self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])

        if not self.classes:
            raise ValueError(f"Could not find any class subdirectories in {self.data_dir}.")

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        logger.debug(f"Class to index mapping: {self.class_to_idx}")

    def _load_samples(self):
        """Load all valid image paths and their corresponding labels."""
        logger.info(f"Loading samples for split '{self.split}' from {self.data_dir}...")
        num_skipped = 0
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if not class_dir.is_dir():
                logger.warning(f"Class directory '{class_dir}' not found or is not a directory. Skipping.")
                continue

            class_idx = self.class_to_idx[class_name]
            for item in class_dir.iterdir():
                # Check if it's a file and has a valid extension
                if item.is_file() and item.suffix.lower() in self.image_extensions:
                    self.samples.append((str(item), class_idx))
                elif item.is_file():
                    num_skipped += 1
                    logger.debug(f"Skipping file with unsupported extension: {item}")

        if num_skipped > 0:
             logger.warning(f"Skipped {num_skipped} files with unsupported extensions.")
        logger.info(f"Found {len(self.samples)} potential image samples.")


    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int, str]]:
        """
        Get a sample from the dataset by index.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Dictionary containing:
                'image': Transformed image tensor.
                'label': Class index (integer).
                'path': Original path of the image file (string).
        """
        img_path, label = self.samples[idx]

        try:
            # Open image using PIL
            image = Image.open(img_path).convert("RGB")

            # Apply transformations if provided
            if self.transform:
                image_tensor = self.transform(image) # Assumes transform output is a tensor
            else:
                # Default minimal transformation if none provided (ToTensor)
                image_tensor = torch.from_numpy(np.array(image).transpose(2, 0, 1)).float() / 255.0

            return {"image": image_tensor, "label": label, "path": img_path}

        except UnidentifiedImageError:
            logger.error(f"Cannot identify image file (corrupted?): {img_path}. Skipping sample {idx}.")
            # Return a dummy sample or raise error. Returning dummy can help training continue.
            # Be careful with dummy data size and type.
            dummy_image = torch.zeros((3, 224, 224), dtype=torch.float32) # Adjust size if needed
            # Use a valid label or a specific indicator label if possible
            dummy_label = label if isinstance(label, int) else 0
            return {"image": dummy_image, "label": dummy_label, "path": f"INVALID:{img_path}"}
        except Exception as e:
            logger.error(f"Error loading or transforming image {img_path} (sample {idx}): {e}")
            # Return dummy data again
            dummy_image = torch.zeros((3, 224, 224), dtype=torch.float32)
            dummy_label = label if isinstance(label, int) else 0
            return {"image": dummy_image, "label": dummy_label, "path": f"ERROR:{img_path}"}

    def get_class_weights(self, mode="sqrt_inv") -> Optional[torch.Tensor]:
        """
        Calculate class weights for handling imbalanced datasets.

        Args:
            mode: Method for calculating weights ('inv', 'sqrt_inv', 'effective').

        Returns:
            Tensor of class weights, or None if samples are not loaded.
        """
        if not self.samples or not self.classes:
            logger.warning("Cannot calculate class weights: samples or classes not available.")
            return None

        num_classes = len(self.classes)
        labels = [label for _, label in self.samples]
        class_counts = np.bincount(labels, minlength=num_classes)

        if np.any(class_counts == 0):
             logger.warning(f"Some classes have zero samples in this dataset split ('{self.split}'). Weights for these classes will be 0.")

        if mode == "inv":
            # Inverse frequency
            weights = np.where(class_counts > 0, 1.0 / class_counts, 0)
        elif mode == "sqrt_inv":
            # Inverse square root frequency (often balances better)
            weights = np.where(class_counts > 0, 1.0 / np.sqrt(class_counts), 0)
        elif mode == "effective":
            # Effective Number of Samples weighting
            beta = 0.999 # Common value
            effective_num = 1.0 - np.power(beta, class_counts)
            weights = np.where(effective_num > 0, (1.0 - beta) / effective_num, 0)
        else:
            logger.error(f"Unknown class weight mode: {mode}. Returning None.")
            return None

        # Normalize weights (optional, adjust based on loss function behavior)
        # Normalizing to sum to num_classes can sometimes be helpful
        total_weight = np.sum(weights)
        if total_weight > 0:
            weights = weights / total_weight * num_classes
        else:
             logger.warning("Total weight is zero, cannot normalize. Returning equal weights (1.0).")
             weights = np.ones_like(weights) # Fallback to equal weights if all counts were 0

        logger.debug(f"Calculated class weights for split '{self.split}' (mode={mode}): {weights}")
        return torch.tensor(weights, dtype=torch.float32)
```

## File: core/data/__init__.py

- Extension: .py
- Language: python
- Size: 385 bytes
- Created: 2025-03-28 14:29:44
- Modified: 2025-03-28 14:29:44

### Code

```python
from plantdoc.core.data.datamodule import PlantDiseaseDataModule
from plantdoc.core.data.datasets import PlantDiseaseDataset
from plantdoc.core.data.transforms import AlbumentationsWrapper, get_transforms
from plantdoc.data.prepare_data import prepare_data

__all__ = [
    "PlantDiseaseDataset",
    "PlantDiseaseDataModule",
    "AlbumentationsWrapper",
    "get_transforms",
    
]

```

## File: core/data/datamodule.py

- Extension: .py
- Language: python
- Size: 16204 bytes
- Created: 2025-03-28 14:28:16
- Modified: 2025-03-28 14:28:16

### Code

```python
# datamodule.py stub
# Path: plantdoc/core/data/datamodule.py
# Description: Data module for plant disease classification using PyTorch Lightning

import os
from typing import List, Optional

import numpy as np
import torch
from omegaconf import DictConfig

# Import from the new project structure
from plantdoc.core.data.dataset import (
    PlantDiseaseDataset,
)
from plantdoc.core.data.transforms import (  # Assuming transforms.py exists based on the second example
    AlbumentationsWrapper,
    get_transforms,
)
from plantdoc.utils.logging import get_logger
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

logger = get_logger(__name__)


class PlantDiseaseDataModule:
    """
    PyTorch Lightning DataModule for plant disease classification.

    Handles dataset splitting, transformations, and data loading.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize the data module.

        Args:
            cfg: Configuration object (Hydra DictConfig). Expected keys:
                data:
                    data_dir: Root directory of the raw dataset.
                    train_val_test_split: List/tuple of ratios for train, val, test splits.
                    random_seed: Seed for reproducibility of splits.
                loader:
                    batch_size: Batch size for data loaders.
                    num_workers: Number of workers for data loading.
                    pin_memory: Whether to use pin_memory.
                    drop_last: Whether to drop the last batch if smaller.
                    prefetch_factor: Prefetch factor for DataLoader.
                preprocessing: Configuration for image transforms (used by get_transforms).
                augmentation: Configuration for image augmentations (used by get_transforms).
        """
        super().__init__()
        self.cfg = cfg
        # Use OmegaConf interpolation or direct access for paths
        self.data_dir = cfg.paths.raw_dir # Use resolved path from config
        self.batch_size = cfg.loader.batch_size
        self.num_workers = cfg.loader.num_workers
        self.pin_memory = cfg.loader.pin_memory
        self.drop_last = cfg.loader.drop_last
        self.prefetch_factor = cfg.loader.prefetch_factor
        self.train_val_test_split = cfg.data.train_val_test_split
        self.random_seed = cfg.data.random_seed

        # Placeholders for datasets and class info
        self.train_dataset: Optional[PlantDiseaseDataset] = None
        self.val_dataset: Optional[PlantDiseaseDataset] = None
        self.test_dataset: Optional[PlantDiseaseDataset] = None
        self.class_names: Optional[List[str]] = None
        self.num_classes: Optional[int] = None

        logger.info("PlantDiseaseDataModule initialized.")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Batch size: {self.batch_size}, Num workers: {self.num_workers}")

    def prepare_data(self):
        """
        Download or prepare data. Only runs on the main process.
        Placeholder for potential download logic.
        """
        # Example: Check if data exists, if not, download/extract
        if not os.path.exists(self.data_dir):
             logger.error(f"Data directory not found: {self.data_dir}")
             raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        logger.info("Data preparation step completed (checking data existence).")
        # If using HDF5 caching (from plantdl), this is where you might trigger cache creation.

    def setup(self, stage: Optional[str] = None):
        """
        Set up the datasets and splits. Runs on all processes.

        Args:
            stage: 'fit', 'validate', 'test', 'predict', or None.
                   Determines which datasets to set up.
        """
        logger.info(f"Setting up DataModule for stage: {stage}")

        # Get transforms for each split
        # Assuming get_transforms uses cfg.preprocessing and cfg.augmentation
        train_transform = AlbumentationsWrapper(get_transforms(self.cfg, split="train"))
        val_transform = AlbumentationsWrapper(get_transforms(self.cfg, split="val"))
        test_transform = AlbumentationsWrapper(get_transforms(self.cfg, split="test"))

        # --- Determine Class Names ---
        # Create a temporary dataset just to get class names consistently
        try:
            temp_dataset = PlantDiseaseDataset(
                data_dir=self.data_dir, # Assumes classes are subdirs of data_dir
                transform=None,
                split="train" # Split doesn't matter here
            )
            self.class_names = temp_dataset.classes
            self.num_classes = len(self.class_names)
            logger.info(f"Found {self.num_classes} classes: {self.class_names}")
        except FileNotFoundError:
             logger.error(f"Data directory or class subdirectories not found in {self.data_dir}")
             raise
        except Exception as e:
            logger.error(f"Error initializing temporary dataset to find classes: {e}")
            raise

        # --- Create Datasets (Handle pre-split or split on the fly) ---
        train_data_path = os.path.join(self.data_dir, "train")
        val_data_path = os.path.join(self.data_dir, "val")
        test_data_path = os.path.join(self.data_dir, "test")

        if os.path.exists(train_data_path) and os.path.exists(val_data_path) and os.path.exists(test_data_path):
            logger.info("Found pre-split train/val/test directories.")
            if stage == "fit" or stage is None:
                self.train_dataset = PlantDiseaseDataset(
                    data_dir=train_data_path,
                    transform=train_transform,
                    split="train",
                    classes=self.class_names,
                )
                self.val_dataset = PlantDiseaseDataset(
                    data_dir=val_data_path,
                    transform=val_transform,
                    split="val",
                    classes=self.class_names,
                )
            if stage == "test" or stage is None:
                 self.test_dataset = PlantDiseaseDataset(
                    data_dir=test_data_path,
                    transform=test_transform,
                    split="test",
                    classes=self.class_names,
                )
            # Handle validate stage explicitly if needed (usually covered by 'fit')
            if stage == "validate" and self.val_dataset is None:
                 self.val_dataset = PlantDiseaseDataset(
                    data_dir=val_data_path,
                    transform=val_transform,
                    split="val",
                    classes=self.class_names,
                 )

        else:
            logger.info("Pre-split directories not found. Splitting dataset on the fly.")
            logger.info(f"Using split ratio: {self.train_val_test_split} and seed: {self.random_seed}")

            full_dataset = PlantDiseaseDataset(
                data_dir=self.data_dir,
                transform=None,  # Apply transforms after splitting
                split="all", # Indicate it's the full dataset
                classes=self.class_names,
            )

            total_len = len(full_dataset)
            train_len = int(self.train_val_test_split[0] * total_len)
            val_len = int(self.train_val_test_split[1] * total_len)
            test_len = total_len - train_len - val_len

            if train_len + val_len + test_len != total_len:
                logger.warning(f"Split lengths ({train_len}, {val_len}, {test_len}) do not sum to total ({total_len}). Adjusting test_len.")
                test_len = total_len - train_len - val_len # Recalculate to be safe

            logger.info(f"Splitting into Train: {train_len}, Val: {val_len}, Test: {test_len}")

            # Split the dataset using torch.utils.data.random_split
            generator = torch.Generator().manual_seed(self.random_seed)
            train_subset, val_subset, test_subset = random_split(
                full_dataset, [train_len, val_len, test_len], generator=generator
            )

            # Create new datasets from subsets with the correct transforms
            if stage == "fit" or stage is None:
                self.train_dataset = self._create_dataset_from_subset(
                    train_subset, train_transform, "train"
                )
                self.val_dataset = self._create_dataset_from_subset(
                    val_subset, val_transform, "val"
                )
            if stage == "test" or stage is None:
                self.test_dataset = self._create_dataset_from_subset(
                    test_subset, test_transform, "test"
                )
            # Handle validate stage explicitly if needed
            if stage == "validate" and self.val_dataset is None:
                 self.val_dataset = self._create_dataset_from_subset(
                    val_subset, val_transform, "val"
                 )

        logger.info("DataModule setup complete.")
        if self.train_dataset: logger.info(f"Train dataset size: {len(self.train_dataset)}")
        if self.val_dataset: logger.info(f"Validation dataset size: {len(self.val_dataset)}")
        if self.test_dataset: logger.info(f"Test dataset size: {len(self.test_dataset)}")


    def _create_dataset_from_subset(
        self, subset, transform, split_name: str
    ) -> PlantDiseaseDataset:
        """
        Helper to create a PlantDiseaseDataset instance from a Subset,
        applying the correct transform.

        Args:
            subset: A torch.utils.data.Subset instance.
            transform: The transformations to apply.
            split_name: Name of the split ('train', 'val', 'test').

        Returns:
            A new PlantDiseaseDataset instance containing only the subset samples.
        """
        # Create a new dataset instance but populate it with subset samples
        dataset = PlantDiseaseDataset(
            data_dir=self.data_dir, # Keep original data_dir reference if needed
            transform=transform,
            split=split_name,
            classes=self.class_names,
        )
        # Overwrite samples with only those from the subset
        dataset.samples = [subset.dataset.samples[i] for i in subset.indices]
        return dataset

    def train_dataloader(self) -> DataLoader:
        """Get the training data loader with weighted sampling for class balance."""
        if not self.train_dataset:
            self.setup(stage='fit') # Ensure setup has run

        if not self.train_dataset:
             raise RuntimeError("Train dataset not initialized. Call setup() first.")

        # --- Weighted Random Sampling ---
        logger.info("Setting up weighted random sampler for training.")
        targets = [label for _, label in self.train_dataset.samples]

        class_counts = np.bincount(targets, minlength=self.num_classes)
        logger.debug(f"Class counts for sampling: {class_counts}")

        # Handle zero counts to avoid division by zero
        weights = np.where(class_counts > 0, 1.0 / np.sqrt(class_counts), 0)
        weights[class_counts == 0] = 0 # Explicitly set weight to 0 for classes with 0 samples

        # Assign weight to each sample based on its class
        sample_weights = torch.tensor([weights[t] for t in targets], dtype=torch.float)

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True # Sample with replacement
        )
        # --- End Sampling ---

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler, # Use sampler, shuffle=False is implicit
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        """Get the validation data loader."""
        if not self.val_dataset:
            self.setup(stage='fit') # Ensure setup has run (fit usually includes val)

        if not self.val_dataset:
             raise RuntimeError("Validation dataset not initialized. Call setup() first.")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False, # No shuffling for validation
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False, # Keep all validation samples
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader:
        """Get the test data loader."""
        if not self.test_dataset:
            self.setup(stage='test') # Ensure setup has run

        if not self.test_dataset:
             raise RuntimeError("Test dataset not initialized. Call setup() first.")

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False, # No shuffling for testing
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False, # Keep all test samples
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def get_class_names(self) -> List[str]:
        """Get the list of class names."""
        if not self.class_names:
            self.setup() # Ensure setup has run
        if not self.class_names:
            raise RuntimeError("Class names not available.")
        return self.class_names

    def get_num_classes(self) -> int:
        """Get the number of classes."""
        if not self.num_classes:
             self.setup() # Ensure setup has run
        if not self.num_classes:
            raise RuntimeError("Number of classes not available.")
        return self.num_classes

    def get_class_weights(self, mode="sqrt_inv") -> Optional[torch.Tensor]:
        """
        Calculate and return class weights, useful for loss functions.

        Args:
            mode: Method for calculating weights ('inv', 'sqrt_inv', 'effective').

        Returns:
            Tensor of class weights or None if train_dataset is not set.
        """
        if not self.train_dataset or not self.num_classes:
            logger.warning("Cannot calculate class weights: train_dataset or num_classes not set.")
            return None

        targets = [label for _, label in self.train_dataset.samples]
        class_counts = np.bincount(targets, minlength=self.num_classes)

        if mode == "inv":
            # Inverse frequency
            weights = np.where(class_counts > 0, 1.0 / class_counts, 0)
        elif mode == "sqrt_inv":
            # Inverse square root frequency (often balances better)
            weights = np.where(class_counts > 0, 1.0 / np.sqrt(class_counts), 0)
        elif mode == "effective":
            # Effective Number of Samples weighting: (1 - beta^N) / (1 - beta)
            # Commonly used beta values are 0.9, 0.99, 0.999, 0.9999
            beta = 0.999
            effective_num = 1.0 - np.power(beta, class_counts)
            weights = np.where(effective_num > 0, (1.0 - beta) / effective_num, 0)
        else:
            logger.error(f"Unknown class weight mode: {mode}")
            return None

        # Normalize weights to sum to num_classes (optional, but common)
        total_weight = np.sum(weights)
        if total_weight > 0:
            weights = weights / total_weight * self.num_classes
        else:
             logger.warning("Total weight is zero, cannot normalize.")
             weights = np.ones_like(weights) # Fallback to equal weights

        logger.info(f"Calculated class weights (mode={mode}): {weights}")
        return torch.tensor(weights, dtype=torch.float32)
```

## File: utils/metrics.py

- Extension: .py
- Language: python
- Size: 13214 bytes
- Created: 2025-03-28 15:03:21
- Modified: 2025-03-28 15:03:21

### Code

```python
# metrics.py stub
"""
Metrics calculation and tracking utilities for model evaluation.
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from plantdoc.utils.logging import get_logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

logger = get_logger(__name__)


class ClassificationMetrics:
    """
    Calculate and store classification metrics.
    
    This class handles calculation of various classification metrics like
    accuracy, precision, recall, F1 score, and confusion matrix.
    
    Args:
        num_classes: Number of classes in the dataset
        class_names: List of class names (if available)
    """
    
    def __init__(
        self, 
        num_classes: int, 
        class_names: Optional[List[str]] = None,
    ):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        
        # Initialize metrics storage
        self.reset()
        
    def reset(self):
        """Reset all metrics."""
        self.all_preds = []
        self.all_targets = []
        self.class_correct = np.zeros(self.num_classes)
        self.class_total = np.zeros(self.num_classes)
        self.confusion_mat = np.zeros((self.num_classes, self.num_classes))
        self.metrics = {}
        
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with batch predictions.
        
        Args:
            logits: Model output logits (B, C)
            targets: Ground truth labels (B,)
        """
        preds = torch.argmax(logits, dim=1)
        preds_np = preds.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        # Store predictions and targets for global metrics
        self.all_preds.append(preds_np)
        self.all_targets.append(targets_np)
        
        # Update class-wise accuracy counters
        for i in range(len(targets_np)):
            label = targets_np[i]
            self.class_total[label] += 1
            if preds_np[i] == label:
                self.class_correct[label] += 1
                
        # Update confusion matrix
        batch_cm = confusion_matrix(
            targets_np, preds_np, labels=range(self.num_classes)
        )
        self.confusion_mat += batch_cm
        
    def compute(self, prefix: str = "") -> Dict[str, float]:
        """
        Compute all metrics.
        
        Args:
            prefix: Optional prefix for metric names
        
        Returns:
            Dictionary of metrics
        """
        if not self.all_preds:
            logger.warning("No predictions to compute metrics from.")
            return {}
            
        # Concatenate all predictions and targets
        all_preds = np.concatenate(self.all_preds)
        all_targets = np.concatenate(self.all_targets)
        
        # Calculate global metrics
        accuracy = accuracy_score(all_targets, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average="weighted", zero_division=0
        )
        
        # Store in metrics dict with optional prefix
        metrics = {
            f"{prefix}accuracy": float(accuracy),
            f"{prefix}precision": float(precision),
            f"{prefix}recall": float(recall),
            f"{prefix}f1": float(f1),
        }
        
        # Calculate per-class metrics
        class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average=None, labels=range(self.num_classes), zero_division=0
        )
        
        # Class accuracy
        class_accuracy = np.divide(
            self.class_correct, self.class_total, 
            out=np.zeros_like(self.class_correct), 
            where=self.class_total > 0
        )
        
        # Store per-class metrics
        for i in range(self.num_classes):
            class_name = self.class_names[i].replace(" ", "_")
            metrics[f"{prefix}class_{class_name}_accuracy"] = float(class_accuracy[i])
            metrics[f"{prefix}class_{class_name}_precision"] = float(class_precision[i])
            metrics[f"{prefix}class_{class_name}_recall"] = float(class_recall[i])
            metrics[f"{prefix}class_{class_name}_f1"] = float(class_f1[i])
            
        # Store all metrics
        self.metrics = metrics
        return metrics
        
    def get_confusion_matrix(self) -> np.ndarray:
        """Get the confusion matrix."""
        return self.confusion_mat
        
    def get_classification_report(self) -> Dict[str, Dict[str, float]]:
        """Get a detailed classification report."""
        if not self.all_preds:
            logger.warning("No predictions to generate classification report.")
            return {}
            
        all_preds = np.concatenate(self.all_preds)
        all_targets = np.concatenate(self.all_targets)
        
        report = classification_report(
            all_targets, all_preds, 
            labels=range(self.num_classes), 
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        return report
        
    def save_metrics(self, output_dir: Union[str, Path], filename: str = "metrics.json"):
        """
        Save metrics to JSON file.
        
        Args:
            output_dir: Directory to save the metrics
            filename: Name of the output file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / filename
        
        try:
            # Ensure metrics are serializable
            serializable_metrics = {}
            for k, v in self.metrics.items():
                if isinstance(v, (np.float32, np.float64, np.int32, np.int64)):
                    serializable_metrics[k] = float(v)
                else:
                    serializable_metrics[k] = v
                    
            with open(output_path, "w") as f:
                json.dump(serializable_metrics, f, indent=2)
                
            logger.info(f"Saved metrics to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
            
    def save_classification_report(self, output_dir: Union[str, Path], filename: str = "classification_report.json"):
        """
        Save classification report to JSON file.
        
        Args:
            output_dir: Directory to save the report
            filename: Name of the output file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / filename
        
        try:
            report = self.get_classification_report()
            
            # Convert numpy values to Python types
            def convert_numpy(obj):
                if isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
                    
            serializable_report = convert_numpy(report)
            
            with open(output_path, "w") as f:
                json.dump(serializable_report, f, indent=2)
                
            logger.info(f"Saved classification report to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save classification report: {e}")
            

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 8),
    normalize: bool = True,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
) -> plt.Figure:
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        output_path: Path to save the plot
        figsize: Figure size (width, height)
        normalize: Whether to normalize the confusion matrix
        title: Plot title
        cmap: Colormap
        
    Returns:
        Matplotlib figure
    """
    if normalize:
        # Normalize by row (true labels)
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)  # avoid div by zero
        cm_plot = cm_norm
        fmt = '.2f'
    else:
        cm_plot = cm
        fmt = 'd'
        
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        cm_plot, annot=False, fmt=fmt, cmap=cmap,
        xticklabels=[name[:15] for name in class_names],  # Shorten long names
        yticklabels=[name[:15] for name in class_names],
        ax=ax
    )
    
    # Remove ticks but keep labels
    ax.set_xticks(np.arange(len(class_names)) + 0.5)
    ax.set_yticks(np.arange(len(class_names)) + 0.5)
    
    # Set labels and title
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_label('Normalized Confusion' if normalize else 'Count')
    
    plt.tight_layout()
    
    # Save figure if output path provided
    if output_path:
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrix plot to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save confusion matrix plot: {e}")
    
    return fig


def plot_metrics_history(
    metrics_history: Dict[str, List[float]],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 8),
    metrics_to_plot: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Plot training/validation metrics history.
    
    Args:
        metrics_history: Dictionary of metric lists
        output_path: Path to save the plot
        figsize: Figure size (width, height)
        metrics_to_plot: List of metrics to plot (default: loss, accuracy)
        
    Returns:
        Matplotlib figure
    """
    # Default metrics to plot
    metrics_to_plot = metrics_to_plot or ['loss', 'accuracy']
    
    # Filter available metrics
    available_metrics = set()
    for metric in metrics_history:
        # Strip prefix like 'train_' or 'val_'
        base_metric = metric.split('_', 1)[-1] if '_' in metric else metric
        available_metrics.add(base_metric)
    
    # Only plot metrics that are available
    metrics_to_plot = [m for m in metrics_to_plot if m in available_metrics]
    
    if not metrics_to_plot:
        logger.warning("No metrics available to plot")
        return None
        
    # Create figure
    fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=figsize, sharex=True)
    
    # If only one metric, axes is not a list
    if len(metrics_to_plot) == 1:
        axes = [axes]
        
    # Plot each metric
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        
        # Find train and val metrics
        train_key = f'train_{metric}' if f'train_{metric}' in metrics_history else None
        val_key = f'val_{metric}' if f'val_{metric}' in metrics_history else None
        
        # Plot train metric if available
        if train_key and metrics_history[train_key]:
            ax.plot(metrics_history[train_key], label=f'Train {metric}', marker='o', markersize=3)
            
        # Plot val metric if available
        if val_key and metrics_history[val_key]:
            ax.plot(metrics_history[val_key], label=f'Validation {metric}', marker='s', markersize=3)
            
        # Set labels and legend
        ax.set_ylabel(metric.capitalize())
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
    # Set x-axis label for the bottom subplot
    axes[-1].set_xlabel('Epoch')
    
    # Set title
    fig.suptitle('Training Metrics History', fontsize=14)
    
    plt.tight_layout()
    
    # Save figure if output path provided
    if output_path:
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved metrics history plot to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save metrics history plot: {e}")
    
    return fig
```

## File: utils/paths.py

- Extension: .py
- Language: python
- Size: 4480 bytes
- Created: 2025-03-28 15:03:41
- Modified: 2025-03-28 15:03:41

### Code

```python
# paths.py stub
"""
Path handling utilities for the project.
"""

import os
from pathlib import Path
from typing import Optional, Union

from plantdoc.utils.logging import get_logger

logger = get_logger(__name__)


def get_project_root() -> Path:
    """
    Get the absolute path to the project root directory.
    
    This searches for a .git directory or setup.py file to locate the project root.
    If neither is found, it uses the current working directory.
    
    Returns:
        Path to the project root directory
    """
    # Start from the directory of this file
    current_dir = Path(__file__).resolve().parent
    
    # Traverse up looking for .git directory or setup.py file
    while current_dir != current_dir.parent:
        if (current_dir / ".git").exists() or (current_dir / "setup.py").exists():
            return current_dir
        current_dir = current_dir.parent
    
    # If we can't find a project root, use the current working directory
    logger.warning("Could not determine project root. Using current working directory.")
    return Path.cwd()


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {path}")
    return path


def get_data_dir(data_type: str = "raw") -> Path:
    """
    Get the path to a data directory.
    
    Args:
        data_type: Type of data directory ("raw", "processed", "interim", etc.)
        
    Returns:
        Path to the data directory
    """
    valid_types = ["raw", "processed", "interim", "external"]
    if data_type not in valid_types:
        logger.warning(f"Unknown data type: {data_type}. Using 'raw' instead.")
        data_type = "raw"
        
    data_dir = get_project_root() / "data" / data_type
    ensure_dir(data_dir)
    return data_dir


def get_outputs_dir(experiment_name: Optional[str] = None) -> Path:
    """
    Get the path to the outputs directory.
    
    Args:
        experiment_name: Optional experiment name for subdirectory
        
    Returns:
        Path to the outputs directory
    """
    outputs_dir = get_project_root() / "outputs"
    ensure_dir(outputs_dir)
    
    if experiment_name:
        experiment_dir = outputs_dir / experiment_name
        ensure_dir(experiment_dir)
        return experiment_dir
    
    return outputs_dir


def get_checkpoints_dir(experiment_name: str) -> Path:
    """
    Get the path to the model checkpoints directory.
    
    Args:
        experiment_name: Experiment name
        
    Returns:
        Path to the checkpoints directory
    """
    checkpoints_dir = get_outputs_dir(experiment_name) / "checkpoints"
    ensure_dir(checkpoints_dir)
    return checkpoints_dir


def get_logs_dir(experiment_name: str) -> Path:
    """
    Get the path to the logs directory.
    
    Args:
        experiment_name: Experiment name
        
    Returns:
        Path to the logs directory
    """
    logs_dir = get_outputs_dir(experiment_name) / "logs"
    ensure_dir(logs_dir)
    return logs_dir


def get_reports_dir(experiment_name: str) -> Path:
    """
    Get the path to the reports directory.
    
    Args:
        experiment_name: Experiment name
        
    Returns:
        Path to the reports directory
    """
    reports_dir = get_outputs_dir(experiment_name) / "reports"
    ensure_dir(reports_dir)
    return reports_dir


def get_plots_dir(experiment_name: str) -> Path:
    """
    Get the path to the plots directory.
    
    Args:
        experiment_name: Experiment name
        
    Returns:
        Path to the plots directory
    """
    plots_dir = get_reports_dir(experiment_name) / "plots"
    ensure_dir(plots_dir)
    return plots_dir


def resolve_path(path: Union[str, Path], relative_to: Optional[Union[str, Path]] = None) -> Path:
    """
    Resolve a path, making it absolute if it's relative.
    
    Args:
        path: Path to resolve
        relative_to: Base path for relative paths
        
    Returns:
        Absolute Path object
    """
    path = Path(path)
    
    if path.is_absolute():
        return path
    
    if relative_to:
        return Path(relative_to) / path
    
    # Relative to project root by default
    return get_project_root() / path
```

## File: utils/__init__.py

- Extension: .py
- Language: python
- Size: 0 bytes
- Created: 2025-03-28 13:05:10
- Modified: 2025-03-28 13:05:10

### Code

```python

```

## File: utils/logger.py

- Extension: .py
- Language: python
- Size: 5115 bytes
- Created: 2025-03-28 14:08:20
- Modified: 2025-03-28 14:08:20

### Code

```python
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

# Dictionary to keep track of loggers that have been created
_LOGGERS: Dict[str, logging.Logger] = {}
_GLOBAL_FILE_HANDLER = None


def get_logger(
    name: str,
    log_level: Union[str, int] = "INFO",
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None,
    timestamp: bool = False,
    use_colors: bool = True,
) -> logging.Logger:
    """
    Create or retrieve a configured logger instance.

    Args:
        name: Name of the logger (typically __name__)
        log_level: Log level (e.g. INFO, DEBUG)
        log_file: Optional filename (e.g. 'train.log')
        log_dir: Directory to store log file
        timestamp: If True, append timestamp to log_file
        use_colors: Use colored log output for console

    Returns:
        logging.Logger instance
    """
    global _GLOBAL_FILE_HANDLER

    if name in _LOGGERS:
        return _LOGGERS[name]

    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = name != "plantdoc"

    # Clear old handlers
    if logger.handlers:
        logger.handlers.clear()

    # Formatter
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(log_format, date_format)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if use_colors and sys.stdout.isatty():
        console_handler.setFormatter(ColoredFormatter(log_format, date_format))
    else:
        console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_dir and log_file:
        os.makedirs(log_dir, exist_ok=True)
        if timestamp:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"{Path(log_file).stem}_{timestamp_str}{Path(log_file).suffix}"
        file_path = os.path.join(log_dir, log_file)

        if _GLOBAL_FILE_HANDLER is None and name == "plantdoc":
            _GLOBAL_FILE_HANDLER = logging.FileHandler(file_path)
            _GLOBAL_FILE_HANDLER.setFormatter(formatter)

        if _GLOBAL_FILE_HANDLER:
            logger.addHandler(_GLOBAL_FILE_HANDLER)
            logger.debug(f"Logging to file: {file_path}")

    _LOGGERS[name] = logger
    return logger


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[37m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[41m",
        "RESET": "\033[0m",
    }

    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        return super().format(record)


def configure_logging(cfg) -> logging.Logger:
    """
    Global logging setup using Hydra config.

    Args:
        cfg: DictConfig from Hydra

    Returns:
        Root logger
    """
    global _GLOBAL_FILE_HANDLER
    _GLOBAL_FILE_HANDLER = None

    # Pull settings from config
    command = cfg.get("command", "run")
    log_level = cfg.logging.get("level", "INFO")
    use_colors = cfg.logging.get("use_colors", True)
    log_dir = Path(getattr(cfg.paths, "logs_dir", "outputs/logs"))
    log_file = f"{command}.log"

    # Silence Hydra and noisy libraries
    for noisy in ["hydra", "matplotlib", "PIL"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Create main app logger and others
    root_logger = get_logger(
        "plantdoc",
        log_level=log_level,
        log_dir=log_dir,
        log_file=log_file,
        use_colors=use_colors,
    )

    for module in ["data", "models", "training", "evaluation", "prediction", "cli", "utils"]:
        get_logger(f"plantdoc.core.{module}", log_level=log_level, use_colors=use_colors)

    # Root logger for external libs (suppress to WARNING+)
    root_root_logger = logging.getLogger()
    root_root_logger.setLevel(logging.WARNING)
    if _GLOBAL_FILE_HANDLER and _GLOBAL_FILE_HANDLER not in root_root_logger.handlers:
        root_root_logger.addHandler(_GLOBAL_FILE_HANDLER)

    root_logger.info(f"Logging initialized. Level: {log_level} | Log dir: {log_dir}")
    return root_logger


def log_execution_params(logger, cfg):
    """
    Log core execution metadata for reproducibility.

    Args:
        logger: Logger instance
        cfg: DictConfig from Hydra
    """
    logger.info("------ Execution Context ------")
    logger.info(f"Command        : {cfg.get('command', 'N/A')}")
    logger.info(f"Model          : {cfg.model.get('name', 'N/A')}")
    logger.info(f"Dataset        : {cfg.data.get('dataset_name', 'N/A')}")
    logger.info(f"Output Dir     : {cfg.get('output_dir', 'N/A')}")
    logger.info(f"Time           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("-------------------------------")

```

## File: utils/visualization.py

- Extension: .py
- Language: python
- Size: 13697 bytes
- Created: 2025-03-28 15:06:28
- Modified: 2025-03-28 15:06:28

### Code

```python
"""
Visualization utilities for model analysis and reporting.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix

from plantdoc.utils.logging import get_logger
from plantdoc.utils.paths import ensure_dir

logger = get_logger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)


def plot_training_history(
    history: Dict[str, List[float]],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 10),
    metrics: Optional[List[str]] = None,
) -> Figure:
    """
    Plot training and validation metrics history.
    
    Args:
        history: Dictionary with metrics history
        output_path: Path to save the figure
        figsize: Figure size
        metrics: List of metrics to plot (default: loss, accuracy)
        
    Returns:
        Matplotlib figure
    """
    metrics = metrics or ['loss', 'accuracy']
    
    # Filter metrics to include only those present in history
    present_metrics = []
    for metric in metrics:
        train_key = metric
        val_key = f'val_{metric}'
        
        if train_key in history or val_key in history:
            present_metrics.append(metric)
    
    if not present_metrics:
        logger.warning("No metrics found in history to plot")
        return None
    
    # Create figure with subplots for each metric
    fig, axes = plt.subplots(len(present_metrics), 1, figsize=figsize, sharex=True)
    
    # If only one metric, ensure axes is a list
    if len(present_metrics) == 1:
        axes = [axes]
    
    for i, metric in enumerate(present_metrics):
        ax = axes[i]
        
        # Train metric
        train_key = metric
        if train_key in history:
            ax.plot(history[train_key], marker='o', markersize=4, linestyle='-', 
                   label=f'Train {metric.capitalize()}')
        
        # Validation metric
        val_key = f'val_{metric}'
        if val_key in history:
            ax.plot(history[val_key], marker='s', markersize=4, linestyle='--', 
                   label=f'Val {metric.capitalize()}')
        
        # Add title, labels, legend
        ax.set_title(f'{metric.capitalize()} vs. Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    # Add epoch label to bottom subplot
    axes[-1].set_xlabel('Epoch')
    
    plt.tight_layout()
    
    # Save figure if output path provided
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training history plot to {output_path}")
    
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 12),
    normalize: bool = True,
    cmap: str = "Blues",
    title: str = "Confusion Matrix",
) -> Figure:
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        output_path: Path to save the figure
        figsize: Figure size
        normalize: Whether to normalize by row
        cmap: Colormap
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Create a normalized version of the confusion matrix
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.zeros_like(cm, dtype=float)
        np.divide(cm, row_sums, out=cm_norm, where=row_sums != 0)
    else:
        cm_norm = cm
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Shorten class names if needed
    shortened_names = []
    for name in class_names:
        if len(name) > 20:
            shortened_names.append(name[:18] + '...')
        else:
            shortened_names.append(name)
    
    # Plot heatmap
    im = ax.imshow(cm_norm, cmap=cmap)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Proportion" if normalize else "Count", rotation=-90, va="bottom")
    
    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    # Add ticks
    ax.set_xticks(np.arange(len(shortened_names)))
    ax.set_yticks(np.arange(len(shortened_names)))
    ax.set_xticklabels(shortened_names, rotation=45, ha='right', rotation_mode='anchor')
    ax.set_yticklabels(shortened_names)
    
    # Add grid lines
    ax.set_xticks(np.arange(len(class_names) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(class_names) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    # Add text annotations (only for small matrices)
    if len(class_names) <= 30:
        fmt = '.2f' if normalize else 'd'
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                value = cm_norm[i, j]
                threshold = cm_norm.max() / 2.
                color = "white" if value > threshold else "black"
                ax.text(j, i, format(value, fmt),
                        ha="center", va="center",
                        color=color)
    
    plt.tight_layout()
    
    # Save figure if output path provided
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix plot to {output_path}")
    
    return fig


def plot_class_metrics(
    metrics: Dict[str, float],
    class_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 10),
    metrics_to_plot: List[str] = ['accuracy', 'precision', 'recall', 'f1'],
    sort_by: Optional[str] = 'f1',
) -> Figure:
    """
    Plot class-wise metrics.
    
    Args:
        metrics: Dictionary of class metrics
        class_names: List of class names
        output_path: Path to save the figure
        figsize: Figure size
        metrics_to_plot: List of metrics to plot
        sort_by: Metric to sort by
        
    Returns:
        Matplotlib figure
    """
    # Extract class metrics
    class_metrics = {}
    
    for class_idx, class_name in enumerate(class_names):
        class_metrics[class_name] = {}
        for metric in metrics_to_plot:
            metric_key = f"class_{class_name.replace(' ', '_')}_{metric}"
            if metric_key in metrics:
                class_metrics[class_name][metric] = metrics[metric_key]
    
    # Convert to DataFrame
    import pandas as pd
    df = pd.DataFrame(class_metrics).T
    
    # Sort if requested
    if sort_by and sort_by in metrics_to_plot:
        df = df.sort_values(by=sort_by, ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        df, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues',
        linewidths=0.5,
        cbar=True,
        ax=ax
    )
    
    # Set title and labels
    ax.set_title('Performance Metrics by Class')
    ax.set_ylabel('Class')
    ax.set_xlabel('Metric')
    
    plt.tight_layout()
    
    # Save figure if output path provided
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved class metrics plot to {output_path}")
    
    return fig


def plot_training_time(
    epoch_times: List[float],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> Figure:
    """
    Plot training time per epoch.
    
    Args:
        epoch_times: List of times per epoch in seconds
        output_path: Path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Plot epoch times
    epochs = np.arange(1, len(epoch_times) + 1)
    ax1.plot(epochs, epoch_times, marker='o', linestyle='-', markersize=4)
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Training Time per Epoch')
    ax1.grid(True, alpha=0.3)
    
    # Plot cumulative time
    cum_times = np.cumsum(epoch_times)
    hours = cum_times / 3600
    ax2.plot(epochs, hours, marker='s', linestyle='-', markersize=4)
    ax2.set_ylabel('Cumulative Time (hours)')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Cumulative Training Time')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if output path provided
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training time plot to {output_path}")
    
    return fig


def plot_model_comparison(
    metrics_list: List[Dict[str, float]],
    model_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 8),
    metrics_to_compare: List[str] = ['accuracy', 'precision', 'recall', 'f1'],
) -> Figure:
    """
    Plot comparison of multiple models.
    
    Args:
        metrics_list: List of metrics dictionaries for each model
        model_names: List of model names
        output_path: Path to save the figure
        figsize: Figure size
        metrics_to_compare: List of metrics to compare
        
    Returns:
        Matplotlib figure
    """
    if len(metrics_list) != len(model_names):
        logger.error("Number of metrics dictionaries must match number of model names")
        return None
    
    # Extract metrics for each model
    comparison_data = []
    for model_idx, metrics in enumerate(metrics_list):
        model_data = {'Model': model_names[model_idx]}
        for metric in metrics_to_compare:
            if metric in metrics:
                model_data[metric] = metrics[metric]
            else:
                model_data[metric] = None
        comparison_data.append(model_data)
    
    # Convert to DataFrame
    import pandas as pd
    df = pd.DataFrame(comparison_data)
    
    # Set Model as index
    df.set_index('Model', inplace=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot bar chart
    df.plot(kind='bar', ax=ax)
    
    # Set title and labels
    ax.set_title('Model Comparison')
    ax.set_ylabel('Score')
    ax.set_ylim([0, 1])
    
    # Add value annotations
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)
    
    # Add legend
    ax.legend(title='Metric')
    
    plt.tight_layout()
    
    # Save figure if output path provided
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved model comparison plot to {output_path}")
    
    return fig


def plot_learning_rate(
    lr_history: List[float],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Learning Rate Schedule",
) -> Figure:
    """
    Plot learning rate schedule.
    
    Args:
        lr_history: List of learning rates
        output_path: Path to save the figure
        figsize: Figure size
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot learning rate
    epochs = np.arange(1, len(lr_history) + 1)
    ax.plot(epochs, lr_history, marker='o', linestyle='-', markersize=4)
    
    # Set log scale for y-axis
    ax.set_yscale('log')
    
    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if output path provided
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved learning rate plot to {output_path}")
    
    return fig


def get_model_size(model: torch.nn.Module) -> float:
    """
    Calculate model size in MB.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    # Convert to MB
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb


def count_model_parameters(model: torch.nn.Module) -> int:
    """
    Count total parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

## File: utils/seed.py

- Extension: .py
- Language: python
- Size: 1920 bytes
- Created: 2025-03-28 15:04:49
- Modified: 2025-03-28 15:04:49

### Code

```python
"""
Seed setting utilities for reproducibility.
"""

import os
import random

import numpy as np
import torch
from plantdoc.utils.logging import get_logger

logger = get_logger(__name__)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set seeds for reproducibility.
    
    This sets seeds for random, numpy, torch, and other libraries
    to ensure reproducible results.
    
    Args:
        seed: Seed number
        deterministic: Whether to set deterministic algorithms in torch
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Set CUDA seeds if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior (note: this may impact performance)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set deterministic algorithms for ops with non-deterministic implementations
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    
    logger.info(f"Set random seed to {seed} (deterministic={deterministic})")


def worker_init_fn(worker_id: int) -> None:
    """
    Initialize seeds for DataLoader workers.
    
    This should be passed to DataLoader's worker_init_fn parameter
    to ensure each worker has a different but reproducible seed.
    
    Args:
        worker_id: Worker ID from DataLoader
    """
    # Get base seed from torch
    base_seed = torch.initial_seed()
    
    # Different seed for each worker but still deterministic
    seeded_worker_id = base_seed + worker_id
    
    # Set seed for this worker
    random.seed(seeded_worker_id)
    np.random.seed(seeded_worker_id % (2**32 - 1))  # numpy only accepts 32-bit seeds
    torch.manual_seed(seeded_worker_id)
```

## File: cli/__init__.py

- Extension: .py
- Language: python
- Size: 0 bytes
- Created: 2025-03-28 13:05:10
- Modified: 2025-03-28 13:05:10

### Code

```python

```

## File: cli/main.py

- Extension: .py
- Language: python
- Size: 9705 bytes
- Created: 2025-03-28 15:08:57
- Modified: 2025-03-28 15:08:57

### Code

```python
"""
Command-line interface for the CBAM Classification project.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import hydra
import typer
from omegaconf import DictConfig, OmegaConf

# Add the parent directory to the path to allow imports from the package
sys.path.append(str(Path(__file__).parent.parent))

from plantdoc.core.data.datamodule import PlantDiseaseDataModule
from plantdoc.core.models.registry import get_model_class
from plantdoc.core.training.train import train_model
from plantdoc.reports.generate_plots import generate_plots_for_report
from plantdoc.reports.generate_report import generate_report
from plantdoc.utils.logging import configure_logging, log_execution_params
from plantdoc.utils.seed import set_seed

app = typer.Typer(help="CBAM Classification CLI")


@app.command()
def train(
    config_path: str = typer.Option(
        "configs/config.yaml", "--config", "-c", help="Path to configuration file"
    ),
    experiment_name: str = typer.Option(
        None, "--experiment", "-e", help="Experiment name (overrides config value)"
    ),
    model_name: str = typer.Option(
        None, "--model", "-m", help="Model name (overrides config value)"
    ),
    epochs: Optional[int] = typer.Option(
        None, "--epochs", help="Number of epochs (overrides config value)"
    ),
):
    """
    Run the training pipeline.
    
    This command trains a model using the specified configuration.
    """
    # Load configuration using Hydra
    with hydra.initialize_config_module(config_module="configs"):
        cfg = hydra.compose(config_name="config")
    
    # Override config values if provided via command line
    if experiment_name:
        cfg.paths.experiment_name = experiment_name
    
    if model_name:
        cfg.model.name = model_name
    
    if epochs:
        cfg.training.epochs = epochs
    
    # Configure logging
    logger = configure_logging(cfg)
    logger.info(f"Starting training with model: {cfg.model.name}")
    
    # Set seed for reproducibility
    set_seed(cfg.data.random_seed, deterministic=cfg.training.deterministic)
    
    # Log execution parameters
    log_execution_params(logger, cfg)
    
    # Create data module
    data_module = PlantDiseaseDataModule(cfg)
    data_module.prepare_data()
    data_module.setup()
    
    # Get model class and initialize model
    model_class = get_model_class(cfg.model.name)
    model = model_class(**cfg.model)
    
    # Train the model
    results = train_model(
        model=model,
        train_loader=data_module.train_dataloader(),
        val_loader=data_module.val_dataloader(),
        cfg=cfg,
    )
    
    logger.info(f"Training completed. Best validation accuracy: {results['best_val_acc']:.4f}")
    
    # Generate plots and report if enabled
    if cfg.reporting.generate_plots:
        logger.info("Generating plots...")
        generate_plots_for_report(cfg.paths.experiment_dir)
    
    if cfg.reporting.generate_report:
        logger.info("Generating training report...")
        generate_report(cfg.paths.experiment_dir)
    
    return results


@app.command()
def eval(
    config_path: str = typer.Option(
        "configs/config.yaml", "--config", "-c", help="Path to configuration file"
    ),
    checkpoint_path: str = typer.Option(
        None, "--checkpoint", "-ckpt", help="Path to model checkpoint"
    ),
    split: str = typer.Option(
        "test", "--split", "-s", help="Dataset split to evaluate on (test, val, train)"
    ),
):
    """
    Run the evaluation pipeline.
    
    This command evaluates a trained model on a specific dataset split.
    """
    # Load configuration using Hydra
    with hydra.initialize_config_module(config_module="configs"):
        cfg = hydra.compose(config_name="config")
    
    # Configure logging
    logger = configure_logging(cfg)
    logger.info(f"Starting evaluation on {split} split")
    
    # Resolve the checkpoint path
    if checkpoint_path is None:
        checkpoint_path = Path(cfg.paths.checkpoint_dir) / "best_model.pth"
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Set seed for reproducibility
    set_seed(cfg.data.random_seed, deterministic=True)
    
    # Import here to avoid circular imports
    from plantdoc.core.evaluation.evaluate import evaluate_model
    
    # Create data module
    data_module = PlantDiseaseDataModule(cfg)
    data_module.prepare_data()
    data_module.setup(stage="test")
    
    # Get the appropriate dataloader
    if split == "test":
        dataloader = data_module.test_dataloader()
    elif split == "val":
        dataloader = data_module.val_dataloader()
    elif split == "train":
        dataloader = data_module.train_dataloader()
    else:
        logger.error(f"Invalid split: {split}. Must be one of 'test', 'val', or 'train'.")
        raise ValueError(f"Invalid split: {split}")
    
    # Get model class and initialize model
    model_class = get_model_class(cfg.model.name)
    model = model_class(**cfg.model)
    
    # Evaluate the model
    metrics = evaluate_model(
        model=model,
        dataloader=dataloader,
        checkpoint_path=checkpoint_path,
        cfg=cfg,
    )
    
    logger.info(f"Evaluation completed. Accuracy: {metrics['accuracy']:.4f}")
    
    return metrics


@app.command()
def tune(
    config_path: str = typer.Option(
        "configs/config.yaml", "--config", "-c", help="Path to configuration file"
    ),
    n_trials: int = typer.Option(
        100, "--trials", "-t", help="Number of Optuna trials"
    ),
    study_name: str = typer.Option(
        "cbam_tuning", "--study", "-s", help="Optuna study name"
    ),
):
    """
    Run hyperparameter tuning using Optuna.
    
    This command tunes hyperparameters for a model using Optuna.
    """
    # Load configuration using Hydra
    with hydra.initialize_config_module(config_module="configs"):
        cfg = hydra.compose(config_name="config")
    
    # Configure logging
    logger = configure_logging(cfg)
    logger.info(f"Starting hyperparameter tuning with {n_trials} trials")
    
    # Set seed for reproducibility
    set_seed(cfg.data.random_seed, deterministic=False)  # Non-deterministic for tuning
    
    # Import the tuning module
    from plantdoc.core.tuning.optuna_runner import run_optuna_study
    from plantdoc.core.tuning.search_space import get_search_space
    
    # Get search space based on model
    search_space = get_search_space(cfg.model.name)
    
    # Run Optuna study
    best_params, study = run_optuna_study(
        cfg=cfg,
        search_space=search_space,
        n_trials=n_trials,
        study_name=study_name,
    )
    
    logger.info(f"Tuning completed. Best parameters: {best_params}")
    logger.info(f"Best validation score: {study.best_value:.4f}")
    
    # Save best parameters
    output_file = Path(cfg.paths.experiment_dir) / "best_params.yaml"
    with open(output_file, "w") as f:
        OmegaConf.save(config=OmegaConf.create(best_params), f=f)
    
    logger.info(f"Best parameters saved to {output_file}")
    
    return best_params


@app.command()
def report(
    experiment: str = typer.Option(
        ..., "--experiment", "-e", help="Name of the experiment or path to experiment directory"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output directory for the report"
    ),
    no_plots: bool = typer.Option(
        False, "--no-plots", help="Skip generating plots for the report"
    ),
):
    """
    Generate a training report for an experiment.
    
    This command generates an HTML report with visualizations from training results.
    """
    # Resolve experiment directory
    experiment_dir = Path(experiment)
    if not experiment_dir.is_absolute():
        # Try to find in outputs directory
        output_parent = Path(__file__).parents[1] / "outputs"
        experiment_dir = output_parent / experiment_dir
    
    # Check if experiment directory exists
    if not experiment_dir.exists():
        typer.echo(f"Error: Experiment directory not found: {experiment_dir}")
        raise typer.Exit(code=1)
    
    # Generate report
    generate_report(
        experiment_dir=experiment_dir,
        output_dir=output,
        generate_plots=not no_plots,
    )
    
    typer.echo(f"Report generated for experiment: {experiment_dir}")


@app.command()
def plots(
    experiment: str = typer.Option(
        ..., "--experiment", "-e", help="Name of the experiment or path to experiment directory"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output directory for the plots"
    ),
):
    """
    Generate plots from training results.
    
    This command generates visualizations from training metrics.
    """
    # Resolve experiment directory
    experiment_dir = Path(experiment)
    if not experiment_dir.is_absolute():
        # Try to find in outputs directory
        output_parent = Path(__file__).parents[1] / "outputs"
        experiment_dir = output_parent / experiment_dir
    
    # Check if experiment directory exists
    if not experiment_dir.exists():
        typer.echo(f"Error: Experiment directory not found: {experiment_dir}")
        raise typer.Exit(code=1)
    
    # Generate plots
    generate_plots_for_report(
        experiment_dir=experiment_dir,
        output_dir=output,
    )
    
    typer.echo(f"Plots generated for experiment: {experiment_dir}")


if __name__ == "__main__":
    app()
```

## File: configs/overrides/__init__.py

- Extension: .py
- Language: python
- Size: 0 bytes
- Created: 2025-03-28 13:05:10
- Modified: 2025-03-28 13:05:10

### Code

```python

```

## File: configs/model/__init__.py

- Extension: .py
- Language: python
- Size: 0 bytes
- Created: 2025-03-28 13:05:10
- Modified: 2025-03-28 13:05:10

### Code

```python

```

## File: configs/hydra/__init__.py

- Extension: .py
- Language: python
- Size: 0 bytes
- Created: 2025-03-28 13:05:10
- Modified: 2025-03-28 13:05:10

### Code

```python

```

## File: reports/generate_report.py

- Extension: .py
- Language: python
- Size: 13220 bytes
- Created: 2025-03-28 15:05:43
- Modified: 2025-03-28 15:05:43

### Code

```python
# generate_report.py stub
"""
Generate HTML reports from training results using Jinja2 templates.
"""

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import jinja2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape
from plantdoc.reports.generate_plots import generate_plots_for_report
from plantdoc.utils.logging import get_logger
from plantdoc.utils.paths import ensure_dir, get_reports_dir

logger = get_logger(__name__)


def format_time(seconds: float) -> str:
    """
    Format time in seconds to a human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string (e.g. "2h 30m 15s")
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"


def format_percentage(value: float) -> str:
    """
    Format a value as a percentage string.
    
    Args:
        value: Value to format (0-1)
        
    Returns:
        Formatted percentage string (e.g. "75.5%")
    """
    return f"{value * 100:.2f}%"


def load_metrics(metrics_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load metrics from a JSON file.
    
    Args:
        metrics_path: Path to metrics JSON file
        
    Returns:
        Dictionary of metrics
    """
    try:
        with open(metrics_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load metrics from {metrics_path}: {e}")
        return {}


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary of configuration
    """
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return {}


def load_history(history_path: Union[str, Path]) -> Dict[str, List[float]]:
    """
    Load training history from a JSON file.
    
    Args:
        history_path: Path to history JSON file
        
    Returns:
        Dictionary of training history
    """
    try:
        with open(history_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load history from {history_path}: {e}")
        return {}


def load_class_names(class_names_path: Union[str, Path]) -> List[str]:
    """
    Load class names from a text file.
    
    Args:
        class_names_path: Path to class names file
        
    Returns:
        List of class names
    """
    try:
        with open(class_names_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Failed to load class names from {class_names_path}: {e}")
        return []


def get_model_info(model_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get information about a saved model.
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Dictionary of model information
    """
    try:
        model_info = {}
        
        # Load model state dict to get size
        state_dict = torch.load(model_path, map_location='cpu')
        
        if isinstance(state_dict, dict):
            # Extract metadata if available
            if "epoch" in state_dict:
                model_info["epoch"] = state_dict["epoch"]
            
            if "model_state_dict" in state_dict:
                # Calculate model size
                param_size = 0
                num_params = 0
                num_layers = 0
                
                for param_name, param in state_dict["model_state_dict"].items():
                    param_size += param.nelement() * param.element_size()
                    num_params += param.nelement()
                    # Count layers (approximate based on unique param names)
                    if param_name.count('.') == 1:  # Just counting direct layers
                        num_layers += 1
                
                model_info["size_mb"] = param_size / (1024 ** 2)
                model_info["num_parameters"] = num_params
                model_info["num_layers"] = num_layers
        
        return model_info
    
    except Exception as e:
        logger.error(f"Failed to get model info from {model_path}: {e}")
        return {}


def render_template(
    template_name: str,
    output_path: Union[str, Path],
    context: Dict[str, Any],
    templates_dir: Optional[Union[str, Path]] = None,
) -> bool:
    """
    Render a Jinja2 template to an output file.
    
    Args:
        template_name: Name of the template file
        output_path: Path to save the rendered output
        context: Dictionary of variables to pass to the template
        templates_dir: Directory containing templates
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Default templates directory is in the reports module
        if templates_dir is None:
            templates_dir = Path(__file__).parent / "templates"
        
        # Set up Jinja2 environment
        env = Environment(
            loader=FileSystemLoader(templates_dir),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        
        # Add custom filters
        env.filters['format_time'] = format_time
        env.filters['format_percentage'] = format_percentage
        
        # Get template
        template = env.get_template(template_name)
        
        # Render template
        output = template.render(**context)
        
        # Create output directory if needed
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        
        # Write output
        with open(output_path, 'w') as f:
            f.write(output)
        
        logger.info(f"Rendered template {template_name} to {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to render template {template_name}: {e}")
        return False


def copy_assets(assets_dir: Union[str, Path], output_dir: Union[str, Path]) -> bool:
    """
    Copy assets (CSS, JS, images) to the output directory.
    
    Args:
        assets_dir: Directory containing assets
        output_dir: Output directory
        
    Returns:
        True if successful, False otherwise
    """
    try:
        assets_dir = Path(assets_dir)
        output_dir = Path(output_dir)
        
        if not assets_dir.exists():
            logger.warning(f"Assets directory {assets_dir} does not exist")
            return False
        
        # Create output directory if needed
        output_assets_dir = output_dir / "assets"
        ensure_dir(output_assets_dir)
        
        # Copy assets
        for asset in assets_dir.glob("**/*"):
            if asset.is_file():
                relative_path = asset.relative_to(assets_dir)
                output_path = output_assets_dir / relative_path
                ensure_dir(output_path.parent)
                shutil.copy2(asset, output_path)
        
        logger.info(f"Copied assets from {assets_dir} to {output_assets_dir}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to copy assets: {e}")
        return False


def get_class_performance(metrics: Dict[str, Any], class_names: List[str]) -> List[Dict[str, Any]]:
    """
    Extract class-wise performance metrics.
    
    Args:
        metrics: Dictionary of metrics
        class_names: List of class names
        
    Returns:
        List of class performance dictionaries
    """
    class_performance = []
    
    for class_name in class_names:
        class_key = class_name.replace(" ", "_")
        
        # Extract metrics for this class
        class_data = {
            "name": class_name,
            "precision": metrics.get(f"class_{class_key}_precision", 0.0),
            "recall": metrics.get(f"class_{class_key}_recall", 0.0),
            "f1": metrics.get(f"class_{class_key}_f1", 0.0),
            "accuracy": metrics.get(f"class_{class_key}_accuracy", 0.0),
        }
        
        class_performance.append(class_data)
    
    # Sort by F1 score (descending)
    class_performance.sort(key=lambda x: x["f1"], reverse=True)
    
    return class_performance


def generate_report(
    experiment_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    template_name: str = "training_report.html",
    generate_plots: bool = True,
) -> None:
    """
    Generate a training report for an experiment.
    
    Args:
        experiment_dir: Directory containing experiment data
        output_dir: Directory to save the report (default: experiment_dir/reports)
        template_name: Name of the template file
        generate_plots: Whether to generate plots for the report
    """
    experiment_dir = Path(experiment_dir)
    
    if not experiment_dir.exists():
        logger.error(f"Experiment directory {experiment_dir} does not exist")
        return
    
    if output_dir is None:
        output_dir = experiment_dir / "reports"
    
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    
    logger.info(f"Generating report for experiment in {experiment_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Generate plots if requested
    plots_dir = output_dir / "plots"
    if generate_plots:
        logger.info("Generating plots for report")
        generate_plots_for_report(experiment_dir, plots_dir)
    
    # Load data
    metrics_path = experiment_dir / "metrics.json"
    config_path = experiment_dir / "config.yaml"
    history_path = experiment_dir / "history.json"
    class_names_path = experiment_dir / "class_names.txt"
    model_path = experiment_dir / "checkpoints" / "best_model.pth"
    
    metrics = load_metrics(metrics_path)
    config = load_config(config_path)
    history = load_history(history_path)
    class_names = load_class_names(class_names_path)
    model_info = get_model_info(model_path)
    
    # Create context for template
    context = {
        "experiment_name": experiment_dir.name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "config": config,
        "model_info": model_info,
        "class_names": class_names,
        "class_performance": get_class_performance(metrics, class_names),
        "plots_dir": "plots",  # Relative path from output_dir
    }
    
    # Add history data if available
    if history:
        context["history"] = history
    
    # Add model name from config
    if config and "model" in config and "name" in config["model"]:
        context["model_name"] = config["model"]["name"]
    else:
        context["model_name"] = "Unknown"
    
    # Add performance metrics
    for key in ["accuracy", "precision", "recall", "f1"]:
        if key in metrics:
            context[key] = metrics[key]
    
    # Add training time
    if "total_time" in metrics:
        context["total_time"] = metrics["total_time"]
        context["total_time_formatted"] = format_time(metrics["total_time"])
    
    # Render template
    output_path = output_dir / template_name
    templates_dir = Path(__file__).parent / "templates"
    
    render_template(
        template_name=template_name,
        output_path=output_path,
        context=context,
        templates_dir=templates_dir,
    )
    
    # Copy assets
    assets_dir = templates_dir / "assets"
    copy_assets(assets_dir, output_dir)
    
    logger.info(f"Report generated: {output_path}")


def main():
    """
    Main entry point for report generation.
    """
    parser = argparse.ArgumentParser(description="Generate training report")
    parser.add_argument(
        "--experiment", "-e", type=str, required=True,
        help="Name of the experiment or path to experiment directory"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output directory for the report"
    )
    parser.add_argument(
        "--template", "-t", type=str, default="training_report.html",
        help="Name of the template file"
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip generating plots for the report"
    )
    
    args = parser.parse_args()
    
    # Resolve experiment directory
    experiment_dir = Path(args.experiment)
    if not experiment_dir.is_absolute():
        # Try to find in outputs directory
        outputs_dir = Path(__file__).parents[2] / "outputs"
        experiment_dir = outputs_dir / experiment_dir
    
    # Generate report
    generate_report(
        experiment_dir=experiment_dir,
        output_dir=args.output,
        template_name=args.template,
        generate_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
```

## File: reports/generate_plots.py

- Extension: .py
- Language: python
- Size: 12692 bytes
- Created: 2025-03-28 15:06:37
- Modified: 2025-03-28 15:06:37

### Code

```python
"""
Generate plots for training reports.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from plantdoc.utils.logging import get_logger
from plantdoc.utils.metrics import plot_confusion_matrix, plot_metrics_history
from plantdoc.utils.paths import ensure_dir
from plantdoc.utils.visualization import (
    plot_class_metrics,
    plot_confusion_matrix,
    plot_learning_rate,
    plot_training_history,
    plot_training_time,
)
from sklearn.metrics import confusion_matrix

logger = get_logger(__name__)


def load_json(file_path: Union[str, Path]) -> Dict:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary of loaded data
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON from {file_path}: {e}")
        return {}


def load_history(experiment_dir: Union[str, Path]) -> Dict[str, List[float]]:
    """
    Load training history from an experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory
        
    Returns:
        Dictionary of training history
    """
    history_path = Path(experiment_dir) / "history.json"
    
    if not history_path.exists():
        # Try metrics_logger.jsonl or other possibilities
        logger_path = Path(experiment_dir) / "metrics_logger.jsonl"
        if logger_path.exists():
            try:
                # Load JSONL file (one JSON object per line)
                with open(logger_path, 'r') as f:
                    lines = f.readlines()
                
                history = {}
                for line in lines:
                    try:
                        entry = json.loads(line.strip())
                        for key, value in entry.items():
                            if isinstance(value, (int, float)):
                                if key not in history:
                                    history[key] = []
                                history[key].append(value)
                    except:
                        pass
                
                return history
            except Exception as e:
                logger.error(f"Failed to load metrics logger from {logger_path}: {e}")
                return {}
        else:
            logger.warning(f"No history file found in {experiment_dir}")
            return {}
    
    return load_json(history_path)


def load_metrics(experiment_dir: Union[str, Path]) -> Dict:
    """
    Load metrics from an experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory
        
    Returns:
        Dictionary of metrics
    """
    metrics_path = Path(experiment_dir) / "metrics.json"
    
    if not metrics_path.exists():
        logger.warning(f"No metrics file found in {experiment_dir}")
        return {}
    
    return load_json(metrics_path)


def load_confusion_matrix(experiment_dir: Union[str, Path]) -> Optional[np.ndarray]:
    """
    Load confusion matrix from an experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory
        
    Returns:
        Confusion matrix as numpy array, or None if not found
    """
    cm_path = Path(experiment_dir) / "confusion_matrix.npy"
    
    if not cm_path.exists():
        logger.warning(f"No confusion matrix file found in {experiment_dir}")
        return None
    
    try:
        return np.load(cm_path)
    except Exception as e:
        logger.error(f"Failed to load confusion matrix from {cm_path}: {e}")
        return None


def load_class_names(experiment_dir: Union[str, Path]) -> List[str]:
    """
    Load class names from an experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory
        
    Returns:
        List of class names
    """
    class_names_path = Path(experiment_dir) / "class_names.txt"
    
    if not class_names_path.exists():
        logger.warning(f"No class_names.txt found in {experiment_dir}")
        return []
    
    try:
        with open(class_names_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Failed to load class names from {class_names_path}: {e}")
        return []


def plot_training_history_from_file(
    history: Dict[str, List[float]],
    output_path: Union[str, Path],
) -> None:
    """
    Plot training history from a history dictionary.
    
    Args:
        history: Dictionary of training history
        output_path: Path to save the plot
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    
    # Filter out non-scalar values
    scalar_history = {}
    for key, values in history.items():
        if isinstance(values, list) and all(isinstance(v, (int, float)) for v in values):
            scalar_history[key] = values
    
    # Identify key metrics to plot
    metrics_to_plot = []
    for metric in ['loss', 'accuracy', 'precision', 'recall', 'f1']:
        # Check if we have both train and val versions
        train_key = f'train_{metric}' if f'train_{metric}' in scalar_history else metric
        val_key = f'val_{metric}'
        
        if train_key in scalar_history and val_key in scalar_history:
            metrics_to_plot.append(metric)
    
    if not metrics_to_plot:
        # Try direct metrics if no train/val versions
        metrics_to_plot = [m for m in ['loss', 'accuracy', 'precision', 'recall', 'f1'] 
                         if m in scalar_history]
    
    if not metrics_to_plot:
        logger.warning("No suitable metrics found for plotting training history")
        return
    
    # Plot training history
    plot_training_history(
        history=scalar_history,
        output_path=output_path,
        metrics=metrics_to_plot,
    )
    
    logger.info(f"Saved training history plot to {output_path}")


def plot_confusion_matrix_from_file(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Union[str, Path],
) -> None:
    """
    Plot confusion matrix from a numpy array.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        output_path: Path to save the plot
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        cm=cm,
        class_names=class_names,
        output_path=output_path,
        normalize=True,
    )
    
    logger.info(f"Saved confusion matrix plot to {output_path}")


def plot_training_time_from_history(
    history: Dict[str, List[float]],
    output_path: Union[str, Path],
) -> None:
    """
    Plot training time from history.
    
    Args:
        history: Dictionary of training history
        output_path: Path to save the plot
    """
    # Check if we have time per epoch
    time_key = 'time' if 'time' in history else 'train_time'
    
    if time_key not in history:
        logger.warning("No training time data found in history")
        return
    
    times = history[time_key]
    
    # Plot training time
    plot_training_time(
        epoch_times=times,
        output_path=output_path,
    )
    
    logger.info(f"Saved training time plot to {output_path}")


def plot_class_metrics_from_metrics(
    metrics: Dict,
    class_names: List[str],
    output_path: Union[str, Path],
) -> None:
    """
    Plot class metrics from a metrics dictionary.
    
    Args:
        metrics: Dictionary of metrics
        class_names: List of class names
        output_path: Path to save the plot
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    
    # Extract class metrics
    class_metrics = {}
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
    
    for class_name in class_names:
        class_key = class_name.replace(' ', '_')
        class_metrics[class_name] = {}
        
        for metric in metrics_to_plot:
            metric_key = f'class_{class_key}_{metric}'
            if metric_key in metrics:
                class_metrics[class_name][metric] = metrics[metric_key]
    
    if not class_metrics:
        logger.warning("No class metrics found in metrics dictionary")
        return
    
    # Plot class metrics
    plot_class_metrics(
        metrics=metrics,
        class_names=class_names,
        output_path=output_path,
    )
    
    logger.info(f"Saved class metrics plot to {output_path}")


def plot_learning_rate_from_history(
    history: Dict[str, List[float]],
    output_path: Union[str, Path],
) -> None:
    """
    Plot learning rate from history.
    
    Args:
        history: Dictionary of training history
        output_path: Path to save the plot
    """
    # Check if we have learning rate data
    lr_key = 'lr' if 'lr' in history else 'learning_rate'
    
    if lr_key not in history:
        logger.warning("No learning rate data found in history")
        return
    
    lr_history = history[lr_key]
    
    # Plot learning rate
    plot_learning_rate(
        lr_history=lr_history,
        output_path=output_path,
    )
    
    logger.info(f"Saved learning rate plot to {output_path}")


def generate_plots_for_report(
    experiment_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
) -> None:
    """
    Generate all plots for a training report.
    
    Args:
        experiment_dir: Path to experiment directory
        output_dir: Directory to save plots (default: experiment_dir/reports/plots)
    """
    experiment_dir = Path(experiment_dir)
    
    if not experiment_dir.exists():
        logger.error(f"Experiment directory {experiment_dir} does not exist")
        return
    
    if output_dir is None:
        output_dir = experiment_dir / "reports" / "plots"
    
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    
    logger.info(f"Generating plots for experiment in {experiment_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load data
    history = load_history(experiment_dir)
    metrics = load_metrics(experiment_dir)
    cm = load_confusion_matrix(experiment_dir)
    class_names = load_class_names(experiment_dir)
    
    # Plot training history
    if history:
        plot_training_history_from_file(
            history=history,
            output_path=output_dir / "training_history.png",
        )
    
    # Plot confusion matrix
    if cm is not None and class_names:
        plot_confusion_matrix_from_file(
            cm=cm,
            class_names=class_names,
            output_path=output_dir / "confusion_matrix.png",
        )
    
    # Plot training time
    if history and ('time' in history or 'train_time' in history):
        plot_training_time_from_history(
            history=history,
            output_path=output_dir / "training_time.png",
        )
    
    # Plot class metrics
    if metrics and class_names:
        plot_class_metrics_from_metrics(
            metrics=metrics,
            class_names=class_names,
            output_path=output_dir / "class_metrics.png",
        )
    
    # Plot learning rate
    if history and ('lr' in history or 'learning_rate' in history):
        plot_learning_rate_from_history(
            history=history,
            output_path=output_dir / "learning_rate.png",
        )
    
    logger.info("Finished generating plots for report")


def main():
    """
    Main entry point for plot generation.
    """
    parser = argparse.ArgumentParser(description="Generate plots for training report")
    parser.add_argument(
        "--experiment", "-e", type=str, required=True,
        help="Name of the experiment or path to experiment directory"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output directory for the plots"
    )
    
    args = parser.parse_args()
    
    # Resolve experiment directory
    experiment_dir = Path(args.experiment)
    if not experiment_dir.is_absolute():
        # Try to find in outputs directory
        outputs_dir = Path(__file__).parents[2] / "outputs"
        experiment_dir = outputs_dir / experiment_dir
    
    # Generate plots
    generate_plots_for_report(
        experiment_dir=experiment_dir,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
```

## File: reports/templates/training_report.html

- Extension: .html
- Language: html
- Size: 28658 bytes
- Created: 2025-03-28 15:08:08
- Modified: 2025-03-28 15:08:08

### Code

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Training Report - {{ model_name }}</title>
    <link
      rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
    />
    <style>
      :root {
        --bg-primary: #09090b;
        --bg-secondary: #09090b;
        --bg-tertiary: #09090b;
        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
        --text-tertiary: #94a3b8;
        --accent-primary: #777777;
        --accent-secondary: #ffffffb3;
        --accent-tertiary: #b5fdbc;
        --success: #b5fdbc;
        --warning: #f59e0b;
        --danger: #ef4444;
        --info: #3b82f6;
        --border-radius: 8px;
        --card-radius: 12px;
        --shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        --transition: all 0.3s ease;
        --border-color: #23272e;
      }

      * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
      }

      body {
        font-family: "SF Pro Display", -apple-system, BlinkMacSystemFont,
          "Segoe UI", Roboto, Oxygen, Ubuntu, sans-serif;
        background-color: var(--bg-primary);
        color: var(--text-primary);
        line-height: 1.6;
      }

      .container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 0 1.5rem;
      }

      .grid {
        display: grid;
        grid-template-columns: 240px 1fr;
        gap: 1.5rem;
        min-height: 100vh;
      }

      /* Sidebar */
      .sidebar {
        background-color: var(--bg-secondary);
        border-right: 1px solid var(--border-color);
        padding: 1.5rem 0;
        position: sticky;
        top: 0;
        height: 100vh;
        overflow-y: auto;
      }

      .sidebar-logo {
        padding: 0 1.5rem;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
      }

      .logo-icon {
        width: 2.5rem;
        height: 2.5rem;
        background: linear-gradient(135deg, #777777, #b5fdbc);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        color: white;
      }

      .logo-text {
        font-size: 1.25rem;
        font-weight: 600;
        letter-spacing: 0.5px;
      }

      .nav-section {
        margin-bottom: 1.5rem;
      }

      .nav-heading {
        padding: 0 1.5rem;
        margin-bottom: 0.75rem;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: var(--text-tertiary);
      }

      .nav-items {
        list-style-type: none;
      }

      .nav-item {
        padding: 0.75rem 1.5rem;
        cursor: pointer;
        transition: var(--transition);
        display: flex;
        align-items: center;
        gap: 0.75rem;
        border-left: 3px solid transparent;
      }

      .nav-item:hover {
        background-color: rgba(255, 255, 255, 0.05);
      }

      .nav-item.active {
        background-color: #23272e;
        border-left: 3px solid #b5fdbc;
      }

      .nav-item i {
        font-size: 1.1rem;
        color: #b5fdbc;
      }

      .nav-item.active i {
        color: #b5fdbc;
      }

      .nav-link {
        color: #b5fdbc;
        text-decoration: none;
        font-size: 0.95rem;
        font-weight: 500;
      }

      .nav-item.active .nav-link {
        color: var(--accent-primary);
      }

      /* Main Content */
      .main-content {
        padding: 2rem 0;
      }

      .header {
        margin-bottom: 2rem;
      }

      .header-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(
          90deg,
          var(--accent-primary),
          var(--accent-tertiary)
        );
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
      }

      .header-subtitle {
        color: var(--text-secondary);
        font-size: 1.1rem;
        max-width: 800px;
      }

      .dashboard {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 1.5rem;
        margin-bottom: 2rem;
      }

      .metric-card {
        background-color: var(--bg-secondary);
        border-radius: var(--card-radius);
        padding: 1.5rem;
        box-shadow: var(--shadow);
        display: flex;
        flex-direction: column;
      }

      .metric-label {
        font-size: 0.9rem;
        color: var(--text-tertiary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.75rem;
      }

      .metric-value {
        font-size: 2.25rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
      }

      .metric-info {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.875rem;
        color: var(--success);
      }

      .metric-info.negative {
        color: var(--danger);
      }

      .metric-info i {
        font-size: 0.75rem;
      }

      .chart-section {
        background-color: var(--bg-secondary);
        border-radius: var(--card-radius);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow);
      }

      .section-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
      }

      .section-title {
        font-size: 1.25rem;
        font-weight: 600;
      }

      .section-actions {
        display: flex;
        gap: 0.75rem;
      }

      .btn {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: var(--border-radius);
        font-size: 0.875rem;
        font-weight: 500;
        cursor: pointer;
        transition: var(--transition);
        border: none;
      }

      .btn-outline {
        background-color: transparent;
        border: 1px solid var(--border-color);
        color: var(--text-secondary);
      }

      .btn-outline:hover {
        background-color: rgba(255, 255, 255, 0.05);
      }

      .btn-primary {
        background-color: var(--accent-primary);
        color: white;
      }

      .btn-primary:hover {
        background-color: #7c3aed;
      }

      .chart-content {
        position: relative;
      }

      .chart-image {
        width: 100%;
        border-radius: var(--border-radius);
        border: 1px solid var(--border-color);
      }

      .chart-caption {
        margin-top: 1rem;
        font-size: 0.9rem;
        color: var(--text-tertiary);
      }

      .data-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9rem;
      }

      .data-table th {
        background-color: var(--bg-tertiary);
        padding: 0.75rem 1rem;
        text-align: left;
        border-bottom: 1px solid var(--border-color);
        color: var(--text-tertiary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 0.75rem;
        font-weight: 600;
      }

      .data-table td {
        padding: 0.75rem 1rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
      }

      .data-table tr:last-child td {
        border-bottom: none;
      }

      .data-table tr:hover td {
        background-color: rgba(255, 255, 255, 0.03);
      }

      .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        padding: 0.25rem 0.5rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
      }

      .status-success {
        background-color: rgba(16, 185, 129, 0.2);
        color: var(--success);
      }

      .status-warning {
        background-color: rgba(245, 158, 11, 0.2);
        color: var(--warning);
      }

      .grid-2 {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1.5rem;
        margin-bottom: 1.5rem;
      }

      .model-details {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
      }

      .detail-item {
        padding: 1rem;
        background-color: var(--bg-tertiary);
        border-radius: var(--border-radius);
      }

      .detail-label {
        font-size: 0.75rem;
        color: var(--text-tertiary);
        margin-bottom: 0.25rem;
      }

      .detail-value {
        font-size: 1rem;
        font-weight: 600;
      }

      .analysis-section {
        background-color: var(--bg-secondary);
        border-radius: var(--card-radius);
        margin-bottom: 1.5rem;
        overflow: hidden;
      }

      .analysis-header {
        padding: 1.25rem 1.5rem;
        background-color: rgba(255, 255, 255, 0.03);
        border-bottom: 1px solid var(--border-color);
        font-size: 1.1rem;
        font-weight: 600;
      }

      .analysis-content {
        padding: 1.5rem;
      }

      .insight-list {
        display: flex;
        flex-direction: column;
        gap: 1rem;
      }

      .insight-item {
        display: flex;
        gap: 0.75rem;
      }

      .insight-icon {
        width: 2rem;
        height: 2rem;
        background-color: rgba(181, 253, 188, 0.2);
        color: #b5fdbc;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: var(--accent-primary);
        flex-shrink: 0;
      }

      .insight-content {
        font-size: 0.95rem;
      }

      /* Scrollbar */
      ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
      }

      ::-webkit-scrollbar-track {
        background: var(--bg-primary);
      }

      ::-webkit-scrollbar-thumb {
        background: var(--bg-tertiary);
        border-radius: 3px;
      }

      ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-primary);
      }

      /* Responsive */
      @media (max-width: 1024px) {
        .grid {
          grid-template-columns: 1fr;
        }

        .sidebar {
          display: none;
        }

        .grid-2 {
          grid-template-columns: 1fr;
        }
      }

      @media (max-width: 768px) {
        .dashboard {
          grid-template-columns: 1fr;
        }

        .model-details {
          grid-template-columns: 1fr;
        }
      }

      /* Performance gauge */
      .gauge-container {
        position: relative;
        width: 100%;
        height: 8px;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
        margin-top: 0.5rem;
        overflow: hidden;
      }

      .gauge-fill {
        height: 100%;
        background: linear-gradient(
          90deg,
          var(--accent-primary),
          var(--accent-tertiary)
        );
        border-radius: 4px;
      }

      /* Conclusion alert */
      .alert {
        padding: 1.25rem;
        border-radius: var(--border-radius);
        margin: 1.5rem 0;
        display: flex;
        gap: 1rem;
        align-items: flex-start;
      }

      .alert-success {
        background-color: rgba(16, 185, 129, 0.1);
        border-left: 4px solid var(--success);
      }

      .alert-icon {
        color: var(--success);
        font-size: 1.5rem;
      }

      .alert-content h4 {
        margin-bottom: 0.5rem;
        color: var(--success);
      }

      .alert-content p {
        color: var(--text-secondary);
        font-size: 0.95rem;
      }

      /* Classification table */
      .class-performance {
        margin-top: 1rem;
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
        gap: 0.75rem;
      }

      .class-item {
        background-color: var(--bg-tertiary);
        border-radius: var(--border-radius);
        padding: 0.75rem;
      }

      .class-name {
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .class-metrics {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
        font-size: 0.75rem;
      }

      .class-metric {
        display: flex;
        justify-content: space-between;
      }

      .class-metric-label {
        color: var(--text-tertiary);
      }

      .class-metric-value {
        font-weight: 600;
        color: var(--text-primary);
      }
    </style>
  </head>

  <body>
    <div class="grid">
      <!-- Sidebar -->
      <aside class="sidebar">
        <div class="sidebar-logo">
          <div class="logo-icon">
            <i class="fas fa-brain"></i>
          </div>
          <div class="logo-text">PlantDL</div>
        </div>

        <div class="nav-section">
          <div class="nav-heading">Dashboard</div>
          <ul class="nav-items">
            <li class="nav-item active">
              <i class="fas fa-chart-line"></i>
              <a href="#overview" class="nav-link">Performance Overview</a>
            </li>
            <li class="nav-item">
              <i class="fas fa-cog"></i>
              <a href="#configuration" class="nav-link">Model Configuration</a>
            </li>
            <li class="nav-item">
              <i class="fas fa-chart-area"></i>
              <a href="#training" class="nav-link">Training Metrics</a>
            </li>
            <li class="nav-item">
              <i class="fas fa-table"></i>
              <a href="#confusion" class="nav-link">Confusion Matrix</a>
            </li>
            <li class="nav-item">
              <i class="fas fa-clock"></i>
              <a href="#time" class="nav-link">Training Time</a>
            </li>
          </ul>
        </div>

        <div class="nav-section">
          <div class="nav-heading">Analysis</div>
          <ul class="nav-items">
            <li class="nav-item">
              <i class="fas fa-microscope"></i>
              <a href="#model-analysis" class="nav-link">Model Analysis</a>
            </li>
            <li class="nav-item">
              <i class="fas fa-layer-group"></i>
              <a href="#class-performance" class="nav-link"
                >Class Performance</a
              >
            </li>
            <li class="nav-item">
              <i class="fas fa-lightbulb"></i>
              <a href="#insights" class="nav-link">Key Insights</a>
            </li>
            <li class="nav-item">
              <i class="fas fa-flag-checkered"></i>
              <a href="#conclusion" class="nav-link">Conclusion</a>
            </li>
          </ul>
        </div>

        <div class="nav-section">
          <div class="nav-heading">Resources</div>
          <ul class="nav-items">
            <li class="nav-item">
              <i class="fas fa-code"></i>
              <a href="#" class="nav-link">View Model Code</a>
            </li>
            <li class="nav-item">
              <i class="fas fa-download"></i>
              <a href="#" class="nav-link">Download Report</a>
            </li>
            <li class="nav-item">
              <i class="fas fa-question-circle"></i>
              <a href="#" class="nav-link">Documentation</a>
            </li>
          </ul>
        </div>
      </aside>

      <!-- Main Content -->
      <main class="main-content">
        <div class="container">
          <header class="header" id="overview">
            <h1 class="header-title">{{ model_name }} Performance Report</h1>
            <p class="header-subtitle">
              This report provides a comprehensive analysis of the
              {{ model_name }} model trained for multi-class classification
              with {{ class_names|length }} classes.
            </p>
          </header>

          <!-- Performance Dashboard -->
          <div class="dashboard">
            <div class="metric-card">
              <div class="metric-label">Accuracy</div>
              <div class="metric-value">{{ (metrics.accuracy * 100)|round(2) }}%</div>
              {% if metrics.val_accuracy %}
              <div class="metric-info">
                <i class="fas fa-info-circle"></i>
                <span>Val: {{ (metrics.val_accuracy * 100)|round(2) }}%</span>
              </div>
              {% endif %}
            </div>

            <div class="metric-card">
              <div class="metric-label">Precision</div>
              <div class="metric-value">{{ (metrics.precision * 100)|round(2) }}%</div>
              {% if metrics.val_precision %}
              <div class="metric-info">
                <i class="fas fa-info-circle"></i>
                <span>Val: {{ (metrics.val_precision * 100)|round(2) }}%</span>
              </div>
              {% endif %}
            </div>

            <div class="metric-card">
              <div class="metric-label">Recall</div>
              <div class="metric-value">{{ (metrics.recall * 100)|round(2) }}%</div>
              {% if metrics.val_recall %}
              <div class="metric-info">
                <i class="fas fa-info-circle"></i>
                <span>Val: {{ (metrics.val_recall * 100)|round(2) }}%</span>
              </div>
              {% endif %}
            </div>

            <div class="metric-card">
              <div class="metric-label">F1 Score</div>
              <div class="metric-value">{{ (metrics.f1 * 100)|round(2) }}%</div>
              {% if metrics.val_f1 %}
              <div class="metric-info">
                <i class="fas fa-info-circle"></i>
                <span>Val: {{ (metrics.val_f1 * 100)|round(2) }}%</span>
              </div>
              {% endif %}
            </div>
          </div>

          <!-- Model Configuration -->
          <section class="analysis-section" id="configuration">
            <div class="analysis-header">
              <i class="fas fa-cog"></i> Model Configuration
            </div>
            <div class="analysis-content">
              <div class="model-details">
                <div class="detail-item">
                  <div class="detail-label">Model Name</div>
                  <div class="detail-value">{{ model_name }}</div>
                </div>
                <div class="detail-item">
                  <div class="detail-label">Optimizer</div>
                  <div class="detail-value">{{ config.optimizer.name|default('Adam') }}</div>
                </div>
                <div class="detail-item">
                  <div class="detail-label">Learning Rate</div>
                  <div class="detail-value">{{ config.optimizer.lr|default('1e-3') }}</div>
                </div>
                <div class="detail-item">
                  <div class="detail-label">Batch Size</div>
                  <div class="detail-value">{{ config.loader.batch_size|default('32') }}</div>
                </div>
                <div class="detail-item">
                  <div class="detail-label">Epochs</div>
                  <div class="detail-value">{{ config.training.epochs|default('100') }}</div>
                </div>
                <div class="detail-item">
                  <div class="detail-label">Loss Function</div>
                  <div class="detail-value">{{ config.loss.name|default('CrossEntropy') }}</div>
                </div>
                <div class="detail-item">
                  <div class="detail-label">Dataset</div>
                  <div class="detail-value">{{ config.data.dataset_name|default('Unknown') }}</div>
                </div>
                <div class="detail-item">
                  <div class="detail-label">Num Classes</div>
                  <div class="detail-value">{{ class_names|length }}</div>
                </div>
                <div class="detail-item">
                  <div class="detail-label">Model Size Mb</div>
                  <div class="detail-value">{{ model_info.size_mb|round(2) }}</div>
                </div>
                <div class="detail-item">
                  <div class="detail-label">Num Layers</div>
                  <div class="detail-value">{{ model_info.num_layers|default('Unknown') }}</div>
                </div>
                <div class="detail-item">
                  <div class="detail-label">Num Parameters</div>
                  <div class="detail-value">{{ "{:,}".format(model_info.num_parameters) }}</div>
                </div>
                <div class="detail-item">
                  <div class="detail-label">Total Training Time</div>
                  <div class="detail-value">{{ total_time|format_time }}</div>
                </div>
                <div class="detail-item">
                  <div class="detail-label">Training Time</div>
                  <div class="detail-value">{{ total_time|format_time }}</div>
                </div>
              </div>
            </div>
          </section>

          <!-- Training History -->
          <section class="chart-section" id="training">
            <div class="section-header">
              <h2 class="section-title">Training History</h2>
              <div class="section-actions">
                <button class="btn btn-outline">
                  <i class="fas fa-download"></i>
                  <span>Export</span>
                </button>
                <button class="btn btn-outline">
                  <i class="fas fa-expand"></i>
                  <span>Fullscreen</span>
                </button>
              </div>
            </div>
            <div class="chart-content">
              <img
                src="{{ plots_dir }}/training_history.png"
                alt="Training History"
                class="chart-image"
              />
              <div class="chart-caption">
                <p>
                  The training history shows convergence patterns for loss and
                  accuracy metrics over time.
                </p>
              </div>
            </div>
          </section>

          <!-- Confusion Matrix -->
          <section class="chart-section" id="confusion">
            <div class="section-header">
              <h2 class="section-title">Confusion Matrix</h2>
              <div class="section-actions">
                <button class="btn btn-outline">
                  <i class="fas fa-download"></i>
                  <span>Export</span>
                </button>
                <button class="btn btn-outline">
                  <i class="fas fa-expand"></i>
                  <span>Fullscreen</span>
                </button>
              </div>
            </div>
            <div class="chart-content">
              <img
                src="{{ plots_dir }}/confusion_matrix.png"
                alt="Confusion Matrix"
                class="chart-image"
              />
              <div class="chart-caption">
                <p>
                  The confusion matrix visualizes classification performance
                  across {{ class_names|length }} classes.
                </p>
              </div>
            </div>
          </section>

          <!-- Training Time Analysis -->
          <section class="chart-section" id="time">
            <div class="section-header">
              <h2 class="section-title">Training Time Analysis</h2>
              <div class="section-actions">
                <button class="btn btn-outline">
                  <i class="fas fa-download"></i>
                  <span>Export</span>
                </button>
                <button class="btn btn-outline">
                  <i class="fas fa-expand"></i>
                  <span>Fullscreen</span>
                </button>
              </div>
            </div>
            <div class="chart-content">
              <img
                src="{{ plots_dir }}/training_time.png"
                alt="Training Time Analysis"
                class="chart-image"
              />
              <div class="chart-caption">
                <p>
                  Analysis of per-epoch training time and cumulative training
                  duration.
                </p>
              </div>
            </div>
          </section>

          <!-- Class Performance -->
          <section class="analysis-section" id="class-performance">
            <div class="analysis-header">
              <i class="fas fa-layer-group"></i> Class Performance
            </div>
            <div class="analysis-content">
              <p style="margin-bottom: 1rem">
                Performance metrics across all {{ class_names|length }} classes.
              </p>

              <div class="class-performance">
                {% for class_data in class_performance[:8] %}
                <div class="class-item">
                  <div class="class-name">{{ class_data.name }}</div>
                  <div class="class-metrics">
                    <div class="class-metric">
                      <span class="class-metric-label">Precision:</span>
                      <span class="class-metric-value">{{ (class_data.precision * 100)|round(1) }}%</span>
                    </div>
                    <div class="class-metric">
                      <span class="class-metric-label">Recall:</span>
                      <span class="class-metric-value">{{ (class_data.recall * 100)|round(1) }}%</span>
                    </div>
                    <div class="class-metric">
                      <span class="class-metric-label">F1:</span>
                      <span class="class-metric-value">{{ (class_data.f1 * 100)|round(1) }}%</span>
                    </div>
                  </div>
                </div>
                {% endfor %}
              </div>
            </div>
          </section>

          <!-- Conclusion -->
          <section class="analysis-section" id="conclusion">
            <div class="analysis-header">
              <i class="fas fa-flag-checkered"></i> Conclusion & Recommendations
            </div>
            <div class="analysis-content">
              <p>
                The {{ model_name }} model demonstrates {{ (metrics.accuracy * 100)|round(2) }}% accuracy for
                the classification task with {{ class_names|length }} classes.
              </p>

              <div class="alert alert-success">
                <div class="alert-icon">
                  <i class="fas fa-check-circle"></i>
                </div>
                <div class="alert-content">
                  <h4>Model Evaluation</h4>
                  <p>
                    This report provides a comprehensive analysis of the model's
                    performance metrics, including accuracy, precision, recall,
                    and F1 score.
                  </p>
                </div>
              </div>
            </div>
          </section>
        </div>
      </main>
    </div>

    <script>
      // Highlight active section on scroll
      document.addEventListener("DOMContentLoaded", () => {
        const sections = document.querySelectorAll("section, header");
        const navItems = document.querySelectorAll(".nav-item");

        const observerOptions = {
          root: null,
          rootMargin: "0px",
          threshold: 0.5,
        };

        const observer = new IntersectionObserver((entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) {
              const id = entry.target.getAttribute("id");
              navItems.forEach((item) => {
                item.classList.remove("active");
                const link = item.querySelector(".nav-link");
                if (link && link.getAttribute("href") === "#" + id) {
                  item.classList.add("active");
                }
              });
            }
          });
        }, observerOptions);

        sections.forEach((section) => {
          if (section.getAttribute("id")) {
            observer.observe(section);
          }
        });

        // Smooth scrolling for navigation links
        navItems.forEach((item) => {
          const link = item.querySelector(".nav-link");
          if (link) {
            link.addEventListener("click", (e) => {
              e.preventDefault();
              const targetId = link.getAttribute("href").substring(1);
              const targetElement = document.getElementById(targetId);
              if (targetElement) {
                window.scrollTo({
                  top: targetElement.offsetTop,
                  behavior: "smooth",
                });
              }
            });
          }
        });
      });
    </script>
  </body>
</html>
```

## File: reports/templates/__init__.py

- Extension: .py
- Language: python
- Size: 259 bytes
- Created: 2025-03-28 15:08:23
- Modified: 2025-03-28 15:08:23

### Code

```python
"""
Templates for HTML report generation.
"""

from pathlib import Path

# Get the path to the templates directory
TEMPLATES_DIR = Path(__file__).parent.absolute()

# Define template file paths
TRAINING_REPORT_TEMPLATE = TEMPLATES_DIR / "training_report.html"
```

