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