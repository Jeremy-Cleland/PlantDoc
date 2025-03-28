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

from utils.logger import get_logger

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
