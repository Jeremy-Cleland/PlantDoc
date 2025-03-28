# Path: PlantDoc/core/models/base

"""
Base model for the plant disease classification models.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

from utils.logger import get_logger

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
