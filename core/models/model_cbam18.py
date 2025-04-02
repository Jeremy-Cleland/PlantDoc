# Path: PlantDoc/core/models/model_cbam18.py

"""
CBAM-Only ResNet18 model for plant disease classification.
"""

import torch

from core.models.backbones.cbam_resnet18 import CBAMResNet18Backbone
from core.models.heads.residual import ResidualHead
from utils.logger import get_logger

from .base import BaseModel
from .registry import register_model

logger = get_logger(__name__)


@register_model(
    "cbam_only_resnet18",
    # Parameter metadata for validation and documentation
    num_classes={
        "type": "int",
        "default": 39,
        "description": "Number of output classes",
        "range": [1, 1000],
    },
    pretrained={
        "type": "bool",
        "default": True,
        "description": "Whether to use pretrained ImageNet weights",
    },
    freeze_backbone={
        "type": "bool",
        "default": False,
        "description": "Whether to freeze the backbone initially",
    },
    dropout_rate={
        "type": "float",
        "default": 0.2,
        "description": "Dropout rate for the head",
        "range": [0.0, 0.9],
    },
    head_type={
        "type": "str",
        "default": "linear",
        "description": "Type of classification head to use",
        "choices": ["linear", "mlp", "residual"],
    },
    hidden_dim={
        "type": "int",
        "default": None,
        "description": "Dimension of hidden layer in the head (None to auto-calculate)",
    },
    in_channels={
        "type": "int",
        "default": 3,
        "description": "Number of input channels",
        "range": [1, 4],
    },
    input_size={
        "type": "list",
        "default": [256, 256],
        "description": "Input image size [height, width]",
    },
    reduction_ratio={
        "type": "int",
        "default": 16,
        "description": "Reduction ratio for CBAM attention",
        "range": [1, 64],
    },
    regularization={
        "type": "dict",
        "default": {
            "stochastic_depth_prob": 0.0,
            "drop_path_prob": 0.0,
        },
        "description": "Regularization parameters for stochastic depth and dropout",
    },
    feature_fusion={
        "type": "bool",
        "default": False,
        "description": "Whether to use feature fusion from multiple layers",
    },
    use_residual_head={
        "type": "bool",
        "default": False,
        "description": "Whether to use residual head for classification",
    },
    spatial_kernel_size={
        "type": "int",
        "default": 7,
        "description": "Kernel size for spatial attention in CBAM",
        "choices": [3, 5, 7, 9],
    },
)
class CBAMResNet18Model(BaseModel):
    """
    CBAM-Only ResNet18 model for plant disease classification with attention.

    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        freeze_backbone: Whether to freeze the backbone
        dropout_rate: Dropout rate for the head
        head_type: Type of head to use
        hidden_dim: Dimension of hidden layer in the head
        in_channels: Number of input channels
        input_size: Input image size
        reduction_ratio: Reduction ratio for CBAM
        regularization: Regularization parameters
        feature_fusion: Whether to use feature fusion from multiple layers
        use_residual_head: Whether to use residual head for classification
        spatial_kernel_size: Kernel size for spatial attention in CBAM
    """

    def __init__(
        self,
        num_classes=39,
        pretrained=True,
        freeze_backbone=False,
        dropout_rate=0.2,
        head_type="linear",
        hidden_dim=None,
        in_channels=3,
        input_size=(256, 256),
        reduction_ratio=16,
        regularization=None,
        feature_fusion=False,
        use_residual_head=False,
        spatial_kernel_size=7,
    ):
        # Store model-specific parameters
        self.reduction_ratio = reduction_ratio
        self.feature_fusion = feature_fusion
        self.use_residual_head = use_residual_head
        self.spatial_kernel_size = spatial_kernel_size

        # Initialize with custom parameters
        super().__init__(
            backbone_name="cbam_resnet18",
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout_rate=dropout_rate,
            head_type=head_type,
            hidden_dim=hidden_dim,
            in_channels=in_channels,
            input_size=input_size,
            regularization=regularization,
        )

        logger.info(
            f"Initialized CBAMResNet18Model with feature_fusion={feature_fusion}, "
            f"use_residual_head={use_residual_head}, reduction_ratio={reduction_ratio}, "
            f"spatial_kernel_size={spatial_kernel_size}"
        )

    def _create_backbone(self):
        """
        Create CBAM-Only ResNet18 backbone.

        Returns:
            Tuple of (backbone, output_dimension)
        """
        backbone = CBAMResNet18Backbone(
            pretrained=self.pretrained,
            freeze_layers=self.frozen_backbone,
            reduction_ratio=self.reduction_ratio,
            regularization=self.regularization,
            feature_fusion=self.feature_fusion,
            in_channels=self.in_channels,
            spatial_kernel_size=self.spatial_kernel_size,
        )
        return backbone, backbone.output_dim

    def _create_head(self):
        """
        Create classification head.

        Returns:
            Classification head module
        """
        if self.use_residual_head:
            # Use residual head for better feature discrimination
            hidden_dim = self.hidden_dim or self.backbone_dim // 2
            return ResidualHead(
                in_features=self.backbone_dim,
                hidden_dim=hidden_dim,
                num_classes=self.num_classes,
                dropout_rate=self.dropout_rate,
            )
        else:
            # Use standard head from base model
            return super()._create_head()

    def gradual_unfreeze(self, stage):
        """
        Gradually unfreeze backbone layers for controlled fine-tuning.

        Args:
            stage: Stage of unfreezing (0-4)
        """
        if hasattr(self.backbone, "gradual_unfreeze"):
            self.backbone.gradual_unfreeze(stage)
            if stage > 0:
                self.frozen_backbone = False
            logger.info(f"Applied gradual unfreezing stage {stage} to CBAM-ResNet18")
        else:
            logger.warning("Backbone does not support gradual unfreezing")
            if stage > 0:
                self.unfreeze_backbone()

    def freeze_backbone(self):
        """
        Freeze backbone layers to prevent their weights from being updated.
        This is useful for transfer learning or fine-tuning.
        """
        if hasattr(self.backbone, "freeze_layers"):
            self.backbone.freeze_layers()
            logger.info("Froze CBAM-ResNet18 backbone layers")
        else:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Froze CBAM-ResNet18 backbone using generic method")
        self.frozen_backbone = True

    def forward_features(self, x):
        """
        Extract features without classification.

        This is useful for feature visualization or center loss.

        Args:
            x: Input tensor

        Returns:
            Feature tensor before classification
        """
        # Get features from backbone
        features = self.backbone(x)
        # Flatten for feature extraction
        flat_features = torch.flatten(features, 1)
        return flat_features

    def get_attention_maps(self, x=None):
        """
        Get the attention maps from the CBAM modules.

        Args:
            x: Input tensor (optional)

        Returns:
            Dictionary of attention maps from different layers
        """
        if x is None:
            logger.warning("No input tensor provided for attention map extraction")
            return {}
            
        if not hasattr(self.backbone, "get_attention_maps"):
            logger.warning("Backbone does not support attention map extraction")
            return {}

        return self.backbone.get_attention_maps(x)

    def get_gradcam_target_layer(self):
        """
        Get the target layer to use for GradCAM visualization.

        Returns:
            The target layer for GradCAM visualization
        """
        # Target the last convolutional layer before attention
        try:
            # Get the last convolutional layer in layer4
            if hasattr(self.backbone, "backbone") and hasattr(
                self.backbone.backbone, "layer4"
            ):
                # Try to get the last conv layer in the last block
                if hasattr(self.backbone.backbone.layer4[-1], "conv2"):
                    logger.info("Using layer4[-1].conv2 as GradCAM target layer")
                    return self.backbone.backbone.layer4[-1].conv2

            # Fallback to the entire layer4
            logger.info(
                "Using backbone.backbone.layer4 as GradCAM target layer (fallback)"
            )
            return self.backbone.backbone.layer4

        except (AttributeError, IndexError) as e:
            logger.warning(f"Error accessing expected layer in CBAM-ResNet18: {e}")

            # Ultimate fallback - any convolutional layer
            for name, module in self.backbone.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    logger.info(
                        f"Using {name} as GradCAM target layer (ultimate fallback)"
                    )
                    return module

            logger.error("Could not find any suitable target layer for GradCAM")
            return None
