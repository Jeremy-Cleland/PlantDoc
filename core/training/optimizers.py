"""
Optimizer configurations for model training.
"""

from typing import Any, Dict, List

import torch.nn as nn
from torch.optim import SGD, Adam, AdamW, Optimizer

from utils.logger import get_logger

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
        logger.info(
            "Model does not have 'backbone' and 'head' attributes. Using single LR group."
        )
        # Simple case - just return all parameters in one group
        return [
            {"params": model.parameters(), "lr": base_lr, "weight_decay": weight_decay}
        ]

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
        param_groups.append(
            {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay}
        )
        logger.info(
            f"  Backbone group: LR={backbone_lr:.2e}, Weight Decay={weight_decay:.2e}"
        )
    if head_params:
        param_groups.append(
            {"params": head_params, "lr": head_lr, "weight_decay": weight_decay}
        )
        logger.info(
            f"  Head group:     LR={head_lr:.2e}, Weight Decay={weight_decay:.2e}"
        )
    if other_params:
        param_groups.append(
            {"params": other_params, "lr": other_lr, "weight_decay": weight_decay}
        )
        logger.warning(
            f"  Found {len(other_params)} parameters outside backbone/head. Using LR={other_lr:.2e}."
        )

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
        raise TypeError(
            f"Optimizer configuration must be a dictionary, got {type(cfg)}"
        )

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
            nesterov=nesterov,
        )
    elif optimizer_name == "adam":
        optimizer = Adam(
            param_groups,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=optimizer_weight_decay,
        )
    elif optimizer_name == "adamw":
        optimizer = AdamW(
            param_groups,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=optimizer_weight_decay,
        )
    else:
        logger.warning(
            f"Unsupported optimizer: '{optimizer_name}', falling back to Adam"
        )
        optimizer = Adam(
            param_groups,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=optimizer_weight_decay,
        )

    logger.info(f"Optimizer '{type(optimizer).__name__}' created successfully.")
    return optimizer
