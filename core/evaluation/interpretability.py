"""
Model interpretability tools for PlantDoc - Enhanced GradCAM implementation.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.logger import get_logger
from utils.paths import ensure_dir

logger = get_logger(__name__)


def evaluate_model(model: nn.Module, data_loader: DataLoader) -> Dict:
    """
    Evaluate a model on a dataset.
    """
    pass


class GradCAM:
    """
    Enhanced Gradient-weighted Class Activation Mapping (Grad-CAM) implementation.

    This class implements Grad-CAM, which visualizes important regions in the image
    for model predictions by using gradients flowing into the final convolutional layer.

    Features:
    - Automatic target layer detection for various model architectures
    - Supports both tensor inputs and direct image path inputs
    - Provides comprehensive visualization options
    - Handles both standard and CBAM-augmented models
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        input_size: Tuple[int, int] = (224, 224),
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
    ):
        """
        Initialize GradCAM.

        Args:
            model: PyTorch model to analyze
            target_layer: Target layer for GradCAM. If None, will try to find the most appropriate layer.
            input_size: Size to which input images will be resized (height, width)
            mean: Mean values for image normalization
            std: Std values for image normalization
        """
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device

        # Store preprocessing parameters
        self.input_size = input_size
        self.mean = mean
        self.std = std

        # Find target layer if not provided
        self.target_layer = (
            target_layer if target_layer is not None else self._find_target_layer()
        )

        # Register hooks
        self.activations = None
        self.gradients = None
        self.forward_hook = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_hook = self.target_layer.register_full_backward_hook(
            self._backward_hook
        )

        logger.info(
            f"GradCAM initialized with target layer: {self.target_layer.__class__.__name__}"
        )

    def _find_target_layer(self) -> nn.Module:
        """
        Automatically find an appropriate target layer for GradCAM.

        This method supports various model architectures including:
        - ResNet models (including CBAM variants)
        - EfficientNet models
        - VGG-like models
        - Generic CNN architectures

        Returns:
            Target convolutional layer for GradCAM
        """
        # For CBAM ResNet models (specific to PlantDoc)
        if hasattr(self.model, "get_gradcam_target_layer"):
            target_layer = self.model.get_gradcam_target_layer()
            if target_layer is not None:
                logger.info("Using model's recommended GradCAM target layer")
                return target_layer

        # For models with backbone attribute (common in PlantDoc)
        if hasattr(self.model, "backbone"):
            backbone = self.model.backbone

            # For ResNet-like backbones
            if hasattr(backbone, "layer4"):
                if hasattr(backbone.layer4[-1], "conv2"):
                    logger.info("Using backbone.layer4[-1].conv2 as target layer")
                    return backbone.layer4[-1].conv2
                else:
                    logger.info("Using backbone.layer4[-1] as target layer")
                    return backbone.layer4[-1]

            # For CBAM-specific architectures
            if hasattr(backbone, "backbone") and hasattr(backbone.backbone, "layer4"):
                if hasattr(backbone.backbone.layer4[-1], "conv2"):
                    logger.info(
                        "Using backbone.backbone.layer4[-1].conv2 as target layer"
                    )
                    return backbone.backbone.layer4[-1].conv2
                else:
                    logger.info("Using backbone.backbone.layer4[-1] as target layer")
                    return backbone.backbone.layer4[-1]

        # For EfficientNet-like models
        if hasattr(self.model, "features"):
            # Find the last conv layer in features
            conv_layers = [
                m for m in self.model.features.modules() if isinstance(m, nn.Conv2d)
            ]
            if conv_layers:
                logger.info(
                    f"Using the last Conv2d layer from features as target layer"
                )
                return conv_layers[-1]

        # Generic approach: find the last conv layer in the entire model
        last_conv = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
                last_conv_name = name

        if last_conv is not None:
            logger.info(
                f"Using {last_conv_name} as target layer (last Conv2d in model)"
            )
            return last_conv

        # If no appropriate layer found
        logger.error("Could not find an appropriate layer for GradCAM")
        raise ValueError("Could not find convolutional layer for GradCAM")

    def _forward_hook(self, module, input, output):
        """Hook for forward pass to capture activations."""
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        """Hook for backward pass to capture gradients."""
        self.gradients = grad_output[0].detach()

    def preprocess_image(
        self, image_input: Union[str, Image.Image, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        Preprocess image for model input.

        Args:
            image_input: Image input as path string, PIL Image, numpy array, or tensor

        Returns:
            Preprocessed image tensor (1, C, H, W)
        """
        # Handle various input types
        if isinstance(image_input, str):
            # Load image from path
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image file not found: {image_input}")
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input
        elif isinstance(image_input, np.ndarray):
            # Convert numpy array to PIL Image
            if image_input.dtype == np.uint8:
                image = Image.fromarray(image_input)
            else:
                # Assume float array in range [0, 1]
                image = Image.fromarray((image_input * 255).astype(np.uint8))
        elif isinstance(image_input, torch.Tensor):
            # If already a tensor, ensure correct format and normalization
            if image_input.ndim == 4:  # (B, C, H, W)
                if image_input.size(0) != 1:
                    logger.warning(
                        f"Batch size > 1 detected, using only the first image"
                    )
                return image_input[:1].to(self.device)
            elif image_input.ndim == 3:  # (C, H, W)
                # Add batch dimension
                return image_input.unsqueeze(0).to(self.device)
            else:
                raise ValueError(f"Unexpected tensor shape: {image_input.shape}")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        # Apply transforms to PIL Image
        transform = transforms.Compose(
            [
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

        return transform(image).unsqueeze(0).to(self.device)

    def compute_cam(
        self, input_tensor: torch.Tensor, target_category: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute GradCAM activation map.

        Args:
            input_tensor: Preprocessed input image tensor (1, C, H, W)
            target_category: Target class index. If None, uses the predicted class.

        Returns:
            GradCAM activation map as numpy array (H, W)
        """
        # Forward pass
        self.model.zero_grad()
        output = self.model(input_tensor)

        # Handle different output formats (some models might return (logits, features))
        if isinstance(output, tuple):
            output = output[0]

        # If target_category is None, use the predicted class
        if target_category is None:
            target_category = torch.argmax(output, dim=1).item()

        # Compute gradients
        target = output[0, target_category]
        target.backward()

        # Ensure gradients and activations are available
        if self.gradients is None or self.activations is None:
            logger.error(
                "Gradients or activations are None. Hooks may not be properly set."
            )
            # Return empty array as fallback
            return np.zeros(self.input_size)

        # Global average pooling of gradients
        weights = torch.mean(self.gradients[0], dim=(1, 2))

        # Weight the channels of the activation map with the gradient weights
        cam = torch.zeros_like(self.activations[0, 0]).to(self.device)
        for i, w in enumerate(weights):
            cam += w * self.activations[0, i]

        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:  # Avoid division by zero
            cam = cam / (cam.max() + 1e-7)

        # Resize to match input image size
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=self.input_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        # Convert to numpy array
        return cam.cpu().numpy()

    def __call__(
        self,
        image_input: Union[str, Image.Image, np.ndarray, torch.Tensor],
        target_category: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate GradCAM for the input image.

        Args:
            image_input: Image input as path string, PIL Image, numpy array, or tensor
            target_category: Target class index. If None, uses the predicted class.

        Returns:
            GradCAM activation map as numpy array (H, W)
        """
        # Preprocess image
        input_tensor = self.preprocess_image(image_input)

        # Compute CAM
        return self.compute_cam(input_tensor, target_category)

    def visualize(
        self,
        image_input: Union[str, Image.Image, np.ndarray, torch.Tensor],
        target_category: Optional[int] = None,
        output_path: Optional[str] = None,
        alpha: float = 0.5,
        colormap: str = "jet",
        show: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate and visualize GradCAM for the input image.

        Args:
            image_input: Image input as path string, PIL Image, numpy array, or tensor
            target_category: Target class index. If None, uses the predicted class.
            output_path: Path to save the visualization. If None, doesn't save.
            alpha: Blending factor for the heatmap overlay
            colormap: Matplotlib colormap name for the heatmap
            show: Whether to display the visualization

        Returns:
            Tuple of (original_image, heatmap, overlaid_image) as numpy arrays
        """
        # Get original image for visualization
        if isinstance(image_input, str):
            original_image = np.array(
                Image.open(image_input).convert("RGB").resize(self.input_size)
            )
        elif isinstance(image_input, Image.Image):
            original_image = np.array(image_input.resize(self.input_size))
        elif isinstance(image_input, np.ndarray):
            # Resize numpy array
            original_image = np.array(
                Image.fromarray(
                    image_input.astype(np.uint8)
                    if image_input.dtype != np.uint8
                    else image_input
                ).resize(self.input_size)
            )
        elif isinstance(image_input, torch.Tensor):
            # Handle tensor (assume normalized)
            if image_input.ndim == 4:  # (B, C, H, W)
                img_tensor = image_input[0]
            else:  # (C, H, W)
                img_tensor = image_input

            # Denormalize and convert to numpy
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            original_image = (img_np - img_np.min()) / (
                img_np.max() - img_np.min() + 1e-7
            )
            original_image = (original_image * 255).astype(np.uint8)
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        # Compute CAM
        cam = self(image_input, target_category)

        # Create heatmap
        cmap = plt.get_cmap(colormap)
        heatmap = cmap(cam)[:, :, :3]  # Remove alpha channel

        # Overlay heatmap on original image
        overlaid = original_image * (1 - alpha) + heatmap * 255 * alpha
        overlaid = overlaid.astype(np.uint8)

        # Save visualization if output path provided
        if output_path:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(original_image)
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            axes[1].imshow(heatmap)
            axes[1].set_title("GradCAM Heatmap")
            axes[1].axis("off")

            axes[2].imshow(overlaid)
            axes[2].set_title("GradCAM Overlay")
            axes[2].axis("off")

            plt.tight_layout()

            # Create output directory if it doesn't exist
            ensure_dir(Path(output_path).parent)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            if not show:
                plt.close(fig)

        # Show plot if requested
        if show:
            plt.show()

        return original_image, heatmap, overlaid

    def predict_and_explain(
        self,
        image_input: Union[str, Image.Image, np.ndarray, torch.Tensor],
        class_names: Optional[List[str]] = None,
        top_k: int = 3,
        output_path: Optional[str] = None,
    ) -> Dict:
        """
        Make a prediction and explain it with GradCAM.

        Args:
            image_input: Image input as path string, PIL Image, numpy array, or tensor
            class_names: List of class names (optional)
            top_k: Number of top predictions to explain
            output_path: Path to save the visualization. If None, doesn't save.

        Returns:
            Dictionary with prediction results and explanations
        """
        # Preprocess image
        input_tensor = self.preprocess_image(image_input)

        # Forward pass
        with torch.no_grad():
            output = self.model(input_tensor)

            # Handle different output formats (some models might return (logits, features))
            if isinstance(output, tuple):
                output = output[0]

            # Get probabilities
            probs = F.softmax(output, dim=1)[0]

            # Get top-k predictions
            top_probs, top_classes = torch.topk(probs, top_k)

        # Convert to numpy for easier handling
        top_probs = top_probs.cpu().numpy()
        top_classes = top_classes.cpu().numpy()

        # Create class labels
        if class_names is not None:
            class_labels = [class_names[i] for i in top_classes]
        else:
            class_labels = [f"Class {i}" for i in top_classes]

        # Generate explanations for each top prediction
        explanations = []
        for i, class_idx in enumerate(top_classes):
            # Generate CAM for this class
            cam = self.compute_cam(input_tensor, class_idx)
            explanations.append(
                {
                    "class_index": int(class_idx),
                    "class_name": class_labels[i],
                    "probability": float(top_probs[i]),
                    "cam": cam,
                }
            )

        # Visualize if output_path is provided
        if output_path:
            # Get original image
            if isinstance(image_input, str):
                original_image = np.array(
                    Image.open(image_input).convert("RGB").resize(self.input_size)
                )
            elif isinstance(image_input, Image.Image):
                original_image = np.array(image_input.resize(self.input_size))
            elif isinstance(image_input, np.ndarray):
                # Resize numpy array
                original_image = np.array(
                    Image.fromarray(
                        image_input.astype(np.uint8)
                        if image_input.dtype != np.uint8
                        else image_input
                    ).resize(self.input_size)
                )
            elif isinstance(image_input, torch.Tensor):
                # Handle tensor (assume normalized)
                if input_tensor.ndim == 4:  # (B, C, H, W)
                    img_tensor = input_tensor[0]
                else:  # (C, H, W)
                    img_tensor = input_tensor

                # Denormalize and convert to numpy
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                original_image = (img_np - img_np.min()) / (
                    img_np.max() - img_np.min() + 1e-7
                )
                original_image = (original_image * 255).astype(np.uint8)

            # Create figure with original image and top-k explanations
            nrows = min(top_k, 3)
            fig, axes = plt.subplots(nrows, 2, figsize=(10, 4 * nrows))

            # If only one row, ensure axes is 2D
            if nrows == 1:
                axes = axes.reshape(1, -1)

            # Plot original image in first column of each row
            for i in range(nrows):
                # Get CAM and create heatmap
                cam = explanations[i]["cam"]
                cmap = plt.get_cmap("jet")
                heatmap = cmap(cam)[:, :, :3]

                # Overlay heatmap on original image
                alpha = 0.5
                overlaid = original_image * (1 - alpha) + heatmap * 255 * alpha
                overlaid = overlaid.astype(np.uint8)

                # Plot
                if i == 0:
                    axes[i, 0].imshow(original_image)
                    axes[i, 0].set_title("Original Image")
                    axes[i, 0].axis("off")
                else:
                    axes[i, 0].axis("off")

                axes[i, 1].imshow(overlaid)
                axes[i, 1].set_title(
                    f"{explanations[i]['class_name']} ({explanations[i]['probability']:.4f})"
                )
                axes[i, 1].axis("off")

            plt.tight_layout()

            # Create output directory if it doesn't exist
            ensure_dir(Path(output_path).parent)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

        # Return results
        return {
            "top_predictions": [
                {
                    "class_index": int(cls),
                    "class_name": name,
                    "probability": float(prob),
                }
                for cls, name, prob in zip(top_classes, class_labels, top_probs)
            ],
            "explanations": explanations,
        }

    def cleanup(self):
        """Remove hooks when done to prevent memory leaks."""
        self.forward_hook.remove()
        self.backward_hook.remove()
        logger.info("GradCAM hooks removed")

    def __del__(self):
        """Clean up hooks when instance is deleted."""
        try:
            self.forward_hook.remove()
            self.backward_hook.remove()
        except:
            pass  # Hooks might already be removed or model might be gone


# Example usage functions


def visualize_gradcam(
    model: nn.Module,
    image_path: str,
    target_class: Optional[int] = None,
    class_names: Optional[List[str]] = None,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """
    Convenience function to visualize GradCAM for a single image.

    Args:
        model: PyTorch model
        image_path: Path to the input image
        target_class: Target class index (if None, uses predicted class)
        class_names: List of class names for visualization (optional)
        output_path: Path to save the visualization

    Returns:
        Overlaid image with GradCAM heatmap
    """
    # Initialize GradCAM
    gradcam = GradCAM(model)

    # Generate visualization
    _, _, overlaid = gradcam.visualize(
        image_path, target_category=target_class, output_path=output_path
    )

    # Clean up
    gradcam.cleanup()

    return overlaid


def explain_model_predictions(
    model: nn.Module,
    image_path: str,
    class_names: List[str],
    output_path: Optional[str] = None,
) -> Dict:
    """
    Explain model predictions for an image with GradCAM.

    Args:
        model: PyTorch model
        image_path: Path to the input image
        class_names: List of class names
        output_path: Path to save the visualization

    Returns:
        Dictionary with prediction results and explanations
    """
    # Initialize GradCAM
    gradcam = GradCAM(model)

    # Generate explanation
    results = gradcam.predict_and_explain(
        image_path, class_names=class_names, output_path=output_path
    )

    # Clean up
    gradcam.cleanup()

    return results
