"""
Building blocks for ResNet architectures with CBAM attention.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel attention module for CBAM.

    Args:
        in_planes: Number of input channels
        reduction_ratio: Reduction ratio for the hidden layer
    """

    def __init__(self, in_planes, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // reduction_ratio, in_planes, 1, bias=False),
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return torch.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial attention module for CBAM.

    Args:
        kernel_size: Size of the convolutional kernel
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return torch.sigmoid(out)


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample when applied in main path of residual blocks.

    Args:
        drop_prob: Probability of dropping the path
    """

    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


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

        # CBAM attention
        self.ca = ChannelAttention(planes, reduction_ratio=cbam_reduction)
        self.sa = SpatialAttention()

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
    block, inplanes, planes, blocks, stride=1, cbam_reduction=16, drop_path_prob=0.0
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
        block(inplanes, planes, stride, downsample, cbam_reduction, drop_path_prob)
    )

    # Add remaining blocks
    new_inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(
            block(new_inplanes, planes, 1, None, cbam_reduction, drop_path_prob)
        )

    return nn.Sequential(*layers)
