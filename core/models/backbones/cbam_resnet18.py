# Path: core/models/backbones/cbam_resnet18.py
"""
CBAM-Only ResNet18 backbone for plant disease classification.
"""

import torch
import torch.nn as nn
import torchvision.models as models

from core.models.backbones.blocks import BasicBlock, make_layer
from utils.logger import get_logger

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
        spatial_kernel_size: Kernel size for spatial attention
    """

    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        cbam_reduction=16,
        drop_path_prob=0.0,
        in_channels=3,
        spatial_kernel_size=7,
    ):
        super(CBAMResNet, self).__init__()

        self.inplanes = 64
        self.spatial_kernel_size = spatial_kernel_size

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
            spatial_kernel_size=spatial_kernel_size,
        )

        self.layer2 = make_layer(
            block,
            64,
            128,
            layers[1],
            stride=2,
            cbam_reduction=cbam_reduction,
            drop_path_prob=stage_drop_paths[1],
            spatial_kernel_size=spatial_kernel_size,
        )

        self.layer3 = make_layer(
            block,
            128,
            256,
            layers[2],
            stride=2,
            cbam_reduction=cbam_reduction,
            drop_path_prob=stage_drop_paths[2],
            spatial_kernel_size=spatial_kernel_size,
        )

        self.layer4 = make_layer(
            block,
            256,
            512,
            layers[3],
            stride=2,
            cbam_reduction=cbam_reduction,
            drop_path_prob=stage_drop_paths[3],
            spatial_kernel_size=spatial_kernel_size,
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

        # Storage for attention maps
        self.attention_maps = {}

        logger.info(
            f"Initialized CBAMResNet with cbam_reduction={cbam_reduction}, "
            f"drop_path_prob={drop_path_prob}, spatial_kernel_size={spatial_kernel_size}"
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

    def get_attention_maps(self):
        """
        Get the collected attention maps.
        
        Returns:
            Dictionary of attention maps
        """
        return self.attention_maps

    def get_gradcam_target_layer(self):
        """
        Return the target layer for GradCAM visualization.
        
        Returns:
            The appropriate convolutional layer for GradCAM
        """
        return self.layer4[-1].conv2


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
    kwargs["spatial_kernel_size"] = kwargs.get("spatial_kernel_size", 7)

    model = CBAMResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    if pretrained:
        # Load standard ResNet weights and adapt to our model
        state_dict = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        ).state_dict()
        model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded pretrained weights for ResNet18")

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
        spatial_kernel_size: Kernel size for spatial attention
    """

    def __init__(
        self,
        pretrained=True,
        freeze_layers=False,
        reduction_ratio=16,
        regularization=None,
        feature_fusion=False,
        in_channels=3,
        spatial_kernel_size=7,
    ):
        super(CBAMResNet18Backbone, self).__init__()

        # Process regularization parameters
        if regularization is None:
            regularization = {}

        self.drop_path_prob = regularization.get("drop_path_prob", 0.0)
        self.feature_fusion = feature_fusion
        self.spatial_kernel_size = spatial_kernel_size

        # Create ResNet backbone with CBAM only
        self.backbone = cbam_resnet18(
            pretrained=pretrained,
            cbam_reduction=reduction_ratio,
            drop_path_prob=self.drop_path_prob,
            in_channels=in_channels,
            spatial_kernel_size=self.spatial_kernel_size,
        )

        # Remove the final fully connected layer
        self.backbone.fc = nn.Identity()

        # Output dimension for further layers
        self.output_dim = 512

        # Freeze layers if requested
        if freeze_layers:
            self.freeze_layers()

        # For storing intermediate attention maps
        self._attention_hooks = []
        self._attention_maps = {}

        # Register hooks to capture attention maps
        self._register_attention_hooks()

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

            logger.info("Initialized feature fusion for multi-scale features")

        logger.info(f"Initialized CBAMResNet18Backbone with features={self.output_dim}")

    def _register_attention_hooks(self):
        """Register forward hooks to capture attention maps"""

        # Clear any existing hooks
        for hook in self._attention_hooks:
            hook.remove()
        self._attention_hooks = []

        # Function to capture channel attention maps
        def get_channel_attention_hook(name):
            def hook(module, input, output):
                self._attention_maps[f"{name}_channel"] = output.detach()
            return hook

        # Function to capture spatial attention maps
        def get_spatial_attention_hook(name):
            def hook(module, input, output):
                self._attention_maps[f"{name}_spatial"] = output.detach()
            return hook

        # Register hooks for BasicBlock attention modules
        for layer_name, layer in [
            ("layer1", self.backbone.layer1),
            ("layer2", self.backbone.layer2),
            ("layer3", self.backbone.layer3),
            ("layer4", self.backbone.layer4),
        ]:
            for block_idx, block in enumerate(layer):
                if hasattr(block, "ca"):
                    hook = block.ca.register_forward_hook(
                        get_channel_attention_hook(f"{layer_name}_{block_idx}")
                    )
                    self._attention_hooks.append(hook)

                if hasattr(block, "sa"):
                    hook = block.sa.register_forward_hook(
                        get_spatial_attention_hook(f"{layer_name}_{block_idx}")
                    )
                    self._attention_hooks.append(hook)

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

    def get_attention_maps(self, x=None):
        """
        Get attention maps from the model.
        
        Args:
            x: Optional input tensor to forward through the model first
            
        Returns:
            Dictionary of attention maps from different layers
        """
        # Forward pass if input is provided
        if x is not None:
            with torch.no_grad():
                _ = self.forward(x)

        return self._attention_maps

    def get_gradcam_target_layer(self):
        """
        Return the target layer for GradCAM visualization.
        
        Returns:
            The appropriate convolutional layer for GradCAM
        """
        return self.backbone.layer4[-1].conv2

    def forward(self, x):
        """
        Forward pass through the backbone.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Feature tensor of shape (B, output_dim, 1, 1)
        """
        # Reset attention maps before forward pass
        self._attention_maps = {}

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
