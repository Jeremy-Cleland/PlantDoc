# loss.py stub
"""
Loss functions for plant disease classification.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logger import get_logger

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
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing

        if weight is not None and not isinstance(weight, torch.Tensor):
            logger.warning(
                "WeightedCrossEntropyLoss 'weight' should be a torch.Tensor. Converting list/tuple."
            )
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
            logger.warning(
                "Label smoothing applied but target is 2D. Converting target to indices using argmax."
            )
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


class FocalLoss(nn.Module):
    """
    Focal Loss as described in https://arxiv.org/abs/1708.02002.
    Useful for imbalanced datasets.

    Args:
        alpha: Weighting factor in range (0,1) for one-hot encoding
        gamma: Focusing parameter, gamma > 0 reduces the relative loss for well-classified examples
        reduction: 'mean', 'sum' or 'none'
        label_smoothing: Float in [0, 1]. Label smoothing coefficient for soft targets
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        logger.info(f"Initialized FocalLoss with gamma={gamma}, alpha={alpha}")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate focal loss.

        Args:
            inputs: Predictions from model (before softmax) (B, C)
            targets: Ground truth class indices (B,) or one-hot/soft targets (B, C)

        Returns:
            Loss tensor
        """
        if targets.ndim == 1:
            num_classes = inputs.size(1)
            device = inputs.device

            if self.label_smoothing > 0:
                targets_int64 = targets.to(torch.int64)
                targets_one_hot = torch.zeros(
                    targets.size(0), num_classes, device=device
                )
                targets_one_hot.scatter_(1, targets_int64.unsqueeze(1), 1)
                targets_one_hot = (
                    targets_one_hot * (1 - self.label_smoothing)
                    + self.label_smoothing / num_classes
                )
            else:
                targets_int64 = targets.to(torch.int64)
                targets_one_hot = F.one_hot(targets_int64, num_classes).float()
        else:
            targets_one_hot = targets

        inputs_softmax = F.softmax(inputs, dim=1)

        pt = (targets_one_hot * inputs_softmax).sum(1)
        focal_weight = (1 - pt) ** self.gamma

        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing if targets.ndim == 1 else 0.0,
        )

        loss = focal_weight * ce_loss

        if self.alpha > 0:
            alpha_weight = targets_one_hot * self.alpha + (1 - targets_one_hot) * (
                1 - self.alpha
            )
            loss = alpha_weight.sum(1) * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class CenterLoss(nn.Module):
    """
    Center Loss for better feature discrimination.

    Encourages intra-class compactness by minimizing the distance
    between features and their corresponding class centers.

    Args:
        num_classes: Number of classes in the dataset
        feature_dim: Dimension of the feature vector
        device: Device to store the centers
        weight: Weight for the center loss
        lr: Learning rate for updating the centers
        use_softplus: Whether to use softplus activation for more stable gradients
    """

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        device: torch.device = None,
        weight: float = 1.0,
        lr: float = 0.1,
        use_softplus: bool = True,
    ):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.weight = weight
        self.lr = lr
        self.use_softplus = use_softplus

        self.centers = nn.Parameter(
            torch.randn(num_classes, feature_dim), requires_grad=True
        )

        if device is not None:
            self.centers = self.centers.to(device)

        logger.info(
            f"Initialized CenterLoss with {num_classes} classes, "
            f"feature_dim={feature_dim}, weight={weight}"
        )

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate the center loss.

        Args:
            features: Feature vectors from the network (before classifier)
            labels: Ground truth labels

        Returns:
            Loss value
        """
        batch_size = features.size(0)

        if self.centers.device != features.device:
            self.centers.data = self.centers.data.to(features.device)

        logger.debug(
            f"Feature dimensions: {features.shape}, centers dimensions: {self.centers.shape}"
        )

        if labels.dim() > 1 and labels.size(1) > 1:
            labels = torch.argmax(labels, dim=1)

        if torch.min(labels) < 0 or torch.max(labels) >= self.num_classes:
            logger.warning(
                f"Invalid label values detected. Min: {torch.min(labels)}, Max: {torch.max(labels)}, Classes: {self.num_classes}"
            )
            labels = torch.clamp(labels, 0, self.num_classes - 1)

        centers_batch = self.centers[labels]
        diff = features - centers_batch

        if self.use_softplus:
            distances = F.softplus(torch.sum(diff * diff, dim=1))
        else:
            distances = torch.sum(diff * diff, dim=1)

        loss = self.weight * torch.mean(distances)

        return loss

    def update_centers(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Manually update the centers (alternative to automatic gradient update).

        Can be called after the optimizer step to update centers with
        a different learning rate if needed.

        Args:
            features: Feature vectors from the network
            labels: Ground truth labels
        """
        if self.centers.device != features.device:
            self.centers.data = self.centers.data.to(features.device)

        if labels.dim() > 1 and labels.size(1) > 1:
            labels = torch.argmax(labels, dim=1)

        for cls_idx in torch.unique(labels):
            cls_features = features[labels == cls_idx]
            if cls_features.size(0) > 0:
                center_update = torch.mean(cls_features, dim=0) - self.centers[cls_idx]
                self.centers.data[cls_idx] += self.lr * center_update


class CombinedLoss(nn.Module):
    """
    Combines multiple loss functions with specified weights.

    Args:
        loss_fns: List of loss functions
        loss_weights: List of weights for each loss function
    """

    def __init__(self, loss_fns, loss_weights=None):
        super(CombinedLoss, self).__init__()
        self.loss_fns = loss_fns

        if loss_weights is None:
            loss_weights = [1.0] * len(loss_fns)

        if len(loss_weights) != len(loss_fns):
            raise ValueError("Number of loss functions and weights must match")

        self.loss_weights = loss_weights

    def forward(self, inputs, targets, features=None):
        """
        Calculate combined loss.

        Args:
            inputs: Predictions from model
            targets: Ground truth labels
            features: Feature vectors (optional, for CenterLoss)

        Returns:
            Combined loss value
        """
        total_loss = 0.0

        for _i, (loss_fn, weight) in enumerate(zip(self.loss_fns, self.loss_weights)):
            if isinstance(loss_fn, CenterLoss):
                if features is not None:
                    loss = loss_fn(features, targets)
                else:
                    logger.warning(
                        "CenterLoss requires feature vectors, but none provided. Skipping."
                    )
                    continue
            else:
                loss = loss_fn(inputs, targets)

            total_loss += weight * loss

        return total_loss


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

    loss_name = (
        cfg.get("name", "cross_entropy").lower().replace("_", "").replace("-", "")
    )
    logger.info(f"Initializing loss function: '{loss_name}'")

    # Extract common parameters
    reduction = cfg.get("reduction", "mean")
    label_smoothing = cfg.get("label_smoothing", 0.0)

    # Handle specific loss types
    if loss_name in ["crossentropy", "crossentropyloss"]:
        logger.info(
            f"  Using CrossEntropyLoss with reduction='{reduction}', label_smoothing={label_smoothing}"
        )
        return nn.CrossEntropyLoss(reduction=reduction, label_smoothing=label_smoothing)

    elif loss_name in ["weightedcrossentropy", "weightedcrossentropyloss"]:
        weights_list = cfg.get("weights")
        weights_tensor = None
        if weights_list is not None:
            try:
                weights_tensor = torch.tensor(weights_list, dtype=torch.float)
                logger.info(
                    f"  WeightedCrossEntropy params: using {len(weights_list)} class weights, "
                    f"reduction='{reduction}', label_smoothing={label_smoothing}"
                )
            except Exception as e:
                logger.error(
                    f"  Failed to convert weights to tensor: {e}. Using unweighted CE."
                )
                return nn.CrossEntropyLoss(
                    reduction=reduction, label_smoothing=label_smoothing
                )

        return WeightedCrossEntropyLoss(
            weight=weights_tensor, reduction=reduction, label_smoothing=label_smoothing
        )

    elif loss_name in ["focalloss", "focal"]:
        alpha = cfg.get("alpha", 0.25)
        gamma = cfg.get("gamma", 2.0)

        logger.info(
            f"  Using FocalLoss with alpha={alpha}, gamma={gamma}, "
            f"reduction='{reduction}', label_smoothing={label_smoothing}"
        )

        return FocalLoss(
            alpha=alpha,
            gamma=gamma,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    elif loss_name in ["centerloss", "center"]:
        num_classes = cfg.get("num_classes", 39)  # Default from config
        feature_dim = cfg.get("feature_dim", 256)  # Default from your model head config
        weight = cfg.get("weight", 0.1)
        lr = cfg.get("lr", 0.1)
        use_softplus = cfg.get("use_softplus", True)

        logger.info(
            f"  Using CenterLoss with num_classes={num_classes}, feature_dim={feature_dim}, "
            f"weight={weight}, lr={lr}, use_softplus={use_softplus}"
        )

        return CenterLoss(
            num_classes=num_classes,
            feature_dim=feature_dim,
            weight=weight,
            lr=lr,
            use_softplus=use_softplus,
        )

    elif loss_name in ["combined", "combinedloss"]:
        # Check if we have components list
        components = cfg.get("components", [])

        if not components:
            logger.warning(
                "No components specified for CombinedLoss, using CrossEntropyLoss"
            )
            return nn.CrossEntropyLoss(
                reduction=reduction, label_smoothing=label_smoothing
            )

        loss_fns = []
        loss_weights = []

        for comp in components:
            comp_name = comp.get("name", "cross_entropy")
            comp_weight = comp.get("weight", 1.0)

            # Create a config for this component
            comp_cfg = {k: v for k, v in comp.items() if k not in ["weight"]}

            # Get the loss function for this component
            loss_fn = get_loss_fn(comp_cfg)

            loss_fns.append(loss_fn)
            loss_weights.append(comp_weight)

            logger.info(f"  Added component {comp_name} with weight {comp_weight}")

        return CombinedLoss(loss_fns=loss_fns, loss_weights=loss_weights)

    else:
        logger.warning(
            f"Unsupported loss function: '{loss_name}', falling back to CrossEntropyLoss"
        )
        return nn.CrossEntropyLoss(reduction=reduction, label_smoothing=label_smoothing)
