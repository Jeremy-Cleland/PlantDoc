"""
Building blocks for ResNet architectures with CBAM attention.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the attention modules from the centralized implementation
from core.models.attention import ChannelAttention, DropPath, SpatialAttention


class BasicBlock(nn.Module):
    """
    Basic block for ResNet with CBAM attention.

    Args:
        inplanes: Number of input channels
        planes: Number of output channels
        stride: Stride for the first convolutional layer
        downsample: Downsample function to match dimensions
        cbam_reduction: Reduction ratio for CBAM attention
        drop_path_prob: Probability of dropping the path
        spatial_kernel_size: Kernel size for spatial attention
    """

    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        cbam_reduction=16,
        drop_path_prob=0.0,
        spatial_kernel_size=7,
    ):
        super(BasicBlock, self).__init__()

        # Conv blocks
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # CBAM attention modules from the centralized implementation
        self.ca = ChannelAttention(planes, cbam_reduction)
        self.sa = SpatialAttention(spatial_kernel_size)

        # Shortcut connection
        self.downsample = downsample

        # Drop path (Stochastic Depth)
        self.drop_path = (
            DropPath(drop_path_prob) if drop_path_prob > 0 else nn.Identity()
        )

    def forward(self, x):
        residual = x

        # Forward pass through conv blocks
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply attention modules
        out = self.ca(out) * out
        out = self.sa(out) * out

        # Apply downsample if needed
        if self.downsample is not None:
            residual = self.downsample(x)

        # Apply stochastic depth
        out = residual + self.drop_path(out)
        out = self.relu(out)

        return out


def make_layer(
    block, inplanes, planes, blocks, stride=1, cbam_reduction=16, drop_path_prob=0.0, spatial_kernel_size=7
):
    """
    Create a layer of consecutive blocks.

    Args:
        block: Block class to use
        inplanes: Number of input channels
        planes: Number of output channels
        blocks: Number of blocks in the layer
        stride: Stride for the first block
        cbam_reduction: Reduction ratio for CBAM attention
        drop_path_prob: Probability of dropping the path
        spatial_kernel_size: Kernel size for spatial attention

    Returns:
        nn.Sequential: Sequential container of blocks
    """
    downsample = None

    # Create downsample layer if needed (when stride > 1 or inplanes != planes)
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []

    # Add first block with potential downsampling
    layers.append(
        block(inplanes, planes, stride, downsample, cbam_reduction, drop_path_prob, spatial_kernel_size)
    )

    # Add remaining blocks
    new_inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(
            block(new_inplanes, planes, 1, None, cbam_reduction, drop_path_prob, spatial_kernel_size)
        )

    return nn.Sequential(*layers)
