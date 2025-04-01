"""
Convolutional Block Attention Module (CBAM) and related components.

This module contains the implementation of CBAM and its components,
which combines channel and spatial attention mechanisms to enhance
the representation power of CNNs. Based on the paper:
"CBAM: Convolutional Block Attention Module" (https://arxiv.org/abs/1807.06521)
"""

import torch
import torch.nn as nn

from utils.logger import get_logger

logger = get_logger(__name__)


class ChannelAttention(nn.Module):
    """
    Channel attention module for CBAM.

    This applies attention across channels using both max pooling and
    average pooling.

    Args:
        channels: Number of input channels
        reduction_ratio: Reduction ratio for the MLP (default: 16)
    """

    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1, bias=False),
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
        super().__init__()

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


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.

    Implements the stochastic depth regularization technique
    described in the paper: "Deep Networks with Stochastic Depth"
    (https://arxiv.org/abs/1603.09382)

    Args:
        drop_prob: Probability of dropping the path (default: 0.0)
    """

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # Work with different dims
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize
        output = x.div(keep_prob) * random_tensor
        return output


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    This combines channel attention and spatial attention sequentially.

    Args:
        channels: Number of input channels
        reduction_ratio: Reduction ratio for the channel attention (default: 16)
        spatial_kernel_size: Kernel size for spatial attention (default: 7)
        drop_path_prob: Probability for drop path regularization (default: 0.0)
    """

    def __init__(
        self, channels, reduction_ratio=16, spatial_kernel_size=7, drop_path_prob=0.0
    ):
        super().__init__()

        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

        # Use the more sophisticated DropPath implementation
        self.drop_path_prob = drop_path_prob
        self.drop_path = (
            DropPath(drop_path_prob) if drop_path_prob > 0 else nn.Identity()
        )

        logger.info(
            f"Initialized CBAM with channels={channels}, reduction_ratio={reduction_ratio}, "
            f"spatial_kernel_size={spatial_kernel_size}, drop_path_prob={drop_path_prob}"
        )

    def forward(self, x):
        # Apply channel attention
        channel_att = self.channel_attention(x)
        x = x * channel_att

        # Apply spatial attention
        spatial_att = self.spatial_attention(x)
        x = x * spatial_att

        # Apply drop path (stochastic depth)
        if self.training and self.drop_path_prob > 0:
            x = self.drop_path(x)

        return x
