# Path: PlantDoc/core/models/heads/residual.py
"""
Residual head implementation for plant disease classification.
"""

import torch
import torch.nn as nn

from utils.logger import get_logger

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
