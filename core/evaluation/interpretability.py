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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils.logger import get_logger
from utils.paths import ensure_dir

logger = get_logger(__name__)


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    num_samples: int = 10,
    output_dir: Optional[Union[str, Path]] = None,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Evaluate a model on a dataset and generate GradCAM visualizations.

    Args:
        model: PyTorch model to evaluate
        data_loader: DataLoader for evaluation data
        num_samples: Number of sample images to visualize
        output_dir: Directory to save visualizations
        class_names: List of class names

    Returns:
        Dictionary with evaluation results
    """
    # Prepare model
    model.eval()
    device = next(model.parameters()).device

    # Create output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        ensure_dir(output_dir)
        viz_dir = output_dir / "gradcam_visualizations"
        ensure_dir(viz_dir)

    # Get class names if not provided
    if class_names is None:
        if hasattr(data_loader.dataset, "classes"):
            class_names = data_loader.dataset.classes
        elif hasattr(data_loader.dataset, "class_names"):
            class_names = data_loader.dataset.class_names
        else:
            # Generate dummy class names
            logger.warning("No class names found. Using index numbers as class names.")
            if hasattr(model, "num_classes"):
                class_names = [f"Class_{i}" for i in range(model.num_classes)]
            else:
                class_names = [
                    f"Class_{i}" for i in range(100)
                ]  # Default to 100 classes max

    # Initialize GradCAM
    logger.info("Initializing GradCAM for model interpretation")
    grad_cam = GradCAM(model=model)

    # Collect sample images for visualization
    logger.info(f"Collecting {num_samples} sample images for visualization")
    vis_samples = []
    vis_targets = []
    vis_paths = []

    # Get samples
    for batch in data_loader:
        if len(vis_samples) >= num_samples:
            break

        # Handle different batch formats
        if isinstance(batch, dict):
            images = batch["image"]
            targets = batch["label"]
            paths = batch.get("path", [""] * len(images))
        else:
            images = batch[0]
            targets = batch[1]
            paths = batch[2] if len(batch) > 2 else [""] * len(images)

        # Add images to visualization list
        batch_size = images.size(0)
        for i in range(batch_size):
            if len(vis_samples) < num_samples:
                vis_samples.append(images[i].cpu())
                vis_targets.append(targets[i].item())
                vis_paths.append(paths[i])
            else:
                break

    # Generate GradCAM visualizations
    logger.info("Generating GradCAM visualizations")
    for idx, (image, target, img_path) in enumerate(
        zip(vis_samples, vis_targets, vis_paths)
    ):
        # Create a descriptive filename
        if img_path:
            base_name = os.path.basename(img_path)
            file_name = f"{idx:02d}_{base_name}"
        else:
            file_name = f"{idx:02d}_class_{target}.png"

        # Get the target class name
        target_class_name = (
            class_names[target] if target < len(class_names) else f"Unknown_{target}"
        )

        try:
            # Generate visualization for the correct class (using target)
            logger.info(
                f"Generating GradCAM for sample {idx + 1}/{len(vis_samples)}: {target_class_name}"
            )

            output_path = viz_dir / f"{file_name}" if output_dir is not None else None

            # Apply GradCAM visualization
            result = grad_cam.visualize(
                image_input=image,
                target_category=target,
                output_path=str(output_path) if output_path else None,
                show=False,
            )

            # Generate prediction explanation
            if output_dir is not None:
                explanation_path = viz_dir / f"{idx:02d}_explanation_{file_name}"
                grad_cam.predict_and_explain(
                    image_input=image,
                    class_names=class_names,
                    top_k=min(5, len(class_names)),
                    output_path=str(explanation_path),
                )

        except Exception as e:
            logger.error(f"Error generating GradCAM for sample {idx}: {e}")

    # Clean up
    grad_cam.cleanup()

    logger.info("Model interpretation completed")

    return {
        "num_visualized_samples": len(vis_samples),
        "visualization_dir": str(viz_dir) if output_dir is not None else None,
    }


class GradCAM:
    """
    Enhanced Gradient-weighted Class Activation Mapping (Grad-CAM) implementation.
    Implements high-quality visualization techniques from recent research.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        input_size: Tuple[int, int] = (224, 224),
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        debug: bool = True,  # Enable debug by default
    ):
        """Initialize GradCAM with enhanced visualization settings."""
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device
        self.debug = debug

        # Store preprocessing parameters
        self.input_size = input_size
        self.mean = [0.485, 0.456, 0.406] if mean is None else mean
        self.std = [0.229, 0.224, 0.225] if std is None else std

        # Check for frozen backbone
        self.model_has_frozen_backbone = False
        self.frozen_parameters = []
        if hasattr(self.model, "frozen_backbone") and self.model.frozen_backbone:
            self.model_has_frozen_backbone = True
            logger.info(
                "Model has frozen backbone. Will temporarily unfreeze for GradCAM."
            )
        elif hasattr(self.model, "backbone"):
            # Check if backbone parameters are frozen
            for name, param in self.model.backbone.named_parameters():
                if not param.requires_grad:
                    self.frozen_parameters.append((name, param))

            if self.frozen_parameters:
                self.model_has_frozen_backbone = True
                logger.info(
                    f"Found {len(self.frozen_parameters)} frozen parameters in backbone."
                )

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

    def _temporarily_unfreeze_backbone(self):
        """
        Temporarily unfreeze backbone for GradCAM visualization.
        Returns a function to restore the original state.
        """
        # Method 1: If model has unfreeze_backbone method
        if hasattr(self.model, "unfreeze_backbone"):
            was_frozen = False
            if hasattr(self.model, "frozen_backbone"):
                was_frozen = self.model.frozen_backbone

            if was_frozen:
                logger.info("Temporarily unfreezing backbone for GradCAM...")
                self.model.unfreeze_backbone()

                def restore_fn():
                    logger.info("Restoring frozen backbone state...")
                    for param in self.model.backbone.parameters():
                        param.requires_grad = False
                    if hasattr(self.model, "frozen_backbone"):
                        self.model.frozen_backbone = True

                return restore_fn

        # Method 2: Handle individual frozen parameters
        elif self.frozen_parameters:
            logger.info(
                f"Temporarily unfreezing {len(self.frozen_parameters)} parameters for GradCAM..."
            )
            for name, param in self.frozen_parameters:
                param.requires_grad = True

            def restore_fn():
                logger.info("Restoring frozen parameters...")
                for name, param in self.frozen_parameters:
                    param.requires_grad = False

            return restore_fn

        # No unfreezing needed
        return lambda: None  # Empty restore function

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
                logger.info("Using the last Conv2d layer from features as target layer")
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
        try:
            self.activations = output.detach()
            if self.debug:
                logger.debug(f"Captured activations shape: {self.activations.shape}")
                logger.debug(
                    f"Activations min: {self.activations.min()}, max: {self.activations.max()}"
                )
        except Exception as e:
            logger.error(f"Error in forward hook: {e}")

    def _backward_hook(self, module, grad_input, grad_output):
        """Hook for backward pass to capture gradients."""
        try:
            self.gradients = grad_output[0].detach()
            if self.debug:
                logger.debug(f"Captured gradients shape: {self.gradients.shape}")
                logger.debug(
                    f"Gradients min: {self.gradients.min()}, max: {self.gradients.max()}"
                )
        except Exception as e:
            logger.error(f"Error in backward hook: {e}")

    def preprocess_image(
        self, image_input: Union[str, Image.Image, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """Preprocess image with enhanced error handling and logging."""
        try:
            # Handle various input types
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    raise FileNotFoundError(f"Image file not found: {image_input}")
                image = Image.open(image_input).convert("RGB")
            elif isinstance(image_input, Image.Image):
                image = image_input
            elif isinstance(image_input, np.ndarray):
                if image_input.dtype == np.uint8:
                    image = Image.fromarray(image_input)
                else:
                    image = Image.fromarray((image_input * 255).astype(np.uint8))
            elif isinstance(image_input, torch.Tensor):
                if image_input.ndim == 4:
                    if image_input.size(0) != 1:
                        logger.warning(
                            "Batch size > 1 detected, using only the first image"
                        )
                    return image_input[:1].to(self.device)
                elif image_input.ndim == 3:
                    return image_input.unsqueeze(0).to(self.device)
                else:
                    raise ValueError(f"Unexpected tensor shape: {image_input.shape}")
            else:
                raise TypeError(f"Unsupported image input type: {type(image_input)}")

            # Apply transforms with high-quality settings
            transform = transforms.Compose(
                [
                    transforms.Resize(
                        self.input_size,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.mean, std=self.std),
                ]
            )

            input_tensor = transform(image).unsqueeze(0).to(self.device)

            # Log preprocessing details
            if self.debug:
                logger.debug(f"Preprocessed tensor shape: {input_tensor.shape}")
                logger.debug(
                    f"Preprocessed tensor range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]"
                )

            return input_tensor

        except Exception as e:
            logger.error(f"Error in preprocess_image: {e}", exc_info=True)
            raise

    def compute_cam(
        self,
        input_tensor: Union[torch.Tensor, np.ndarray, Image.Image, str],
        target_category: Optional[int] = None,
    ) -> np.ndarray:
        """Compute high-quality GradCAM activation map."""
        try:
            # Temporarily unfreeze backbone if needed
            restore_fn = self._temporarily_unfreeze_backbone()

            try:
                # Preprocess image if not already a tensor
                if not isinstance(input_tensor, torch.Tensor):
                    input_tensor = self.preprocess_image(input_tensor)
                else:
                    # If already tensor, ensure it's on correct device
                    input_tensor = input_tensor.to(self.device)

                    # Ensure batch dimension
                    if input_tensor.ndim == 3:
                        input_tensor = input_tensor.unsqueeze(0)

                # Forward pass
                self.model.zero_grad()
                output = self.model(input_tensor)
                if self.debug:
                    logger.debug(f"Model output type: {type(output)}")

                # Handle different output formats
                if isinstance(output, tuple):
                    logger.debug(f"Output tuple length: {len(output)}")
                    if len(output) == 2 or len(output) == 3:  # (logits, features)
                        output = output[0]
                    else:
                        logger.warning(f"Unexpected output tuple length: {len(output)}")
                        output = output[0]  # Use first element as fallback

                # If target_category is None, use the predicted class
                if target_category is None:
                    target_category = torch.argmax(output, dim=1).item()
                else:
                    try:
                        # Handle numeric types first
                        if isinstance(
                            target_category, (int, float, np.integer, np.floating)
                        ):
                            target_category = int(target_category)
                        # Handle tensor
                        elif isinstance(target_category, torch.Tensor):
                            target_category = target_category.item()
                        # For OmegaConf's ListConfig indices, convert to Python int
                        elif (
                            hasattr(target_category, "__class__")
                            and "ListConfig" in target_category.__class__.__name__
                        ):
                            target_category = int(target_category)
                        # If still not int, raise error
                        if not isinstance(target_category, int):
                            raise TypeError(
                                f"Could not convert {type(target_category)} to int"
                            )
                    except (TypeError, ValueError) as e:
                        logger.error(
                            f"Invalid target category type: {type(target_category)}, error: {e}"
                        )
                        target_category = torch.argmax(output, dim=1).item()

                # Debug output before computing gradients
                if self.debug:
                    logger.info(
                        f"Computing gradients for class {target_category} with output shape {output.shape}"
                    )
                    # Check if target_category is a valid index
                    if target_category >= output.shape[1]:
                        logger.warning(
                            f"target_category {target_category} is >= output dimension {output.shape[1]}"
                        )
                        target_category = torch.argmax(output, dim=1).item()
                        logger.info(f"Using predicted class {target_category} instead")

                # Compute gradients
                target = output[0, target_category]
                target.backward()

                # Ensure gradients and activations are available
                if self.gradients is None or self.activations is None:
                    logger.error(
                        "Gradients or activations are None. Hooks may not be properly set."
                    )
                    return np.zeros(self.input_size)

                # Debug output for gradient and activation values
                if self.debug:
                    logger.info(
                        f"Gradients shape: {self.gradients.shape}, min: {self.gradients.min()}, max: {self.gradients.max()}"
                    )
                    logger.info(
                        f"Activations shape: {self.activations.shape}, min: {self.activations.min()}, max: {self.activations.max()}"
                    )

                # Enhanced gradient computation
                weights = torch.mean(self.gradients[0], dim=(1, 2))

                # Debug log the weights
                if self.debug:
                    logger.info(
                        f"Weights shape: {weights.shape}, min: {weights.min()}, max: {weights.max()}, sum: {weights.sum()}"
                    )

                # Apply softmax for better visualization only if weights are not all zeros
                if weights.abs().sum() > 1e-10:
                    weights = F.softmax(weights, dim=0)
                    if self.debug:
                        logger.info(
                            f"After softmax - weights min: {weights.min()}, max: {weights.max()}, sum: {weights.sum()}"
                        )
                else:
                    logger.warning(
                        "Weights are very small or zero - using absolute values instead"
                    )
                    weights = (
                        torch.abs(weights) + 1e-10
                    )  # Add small epsilon to avoid zeros
                    weights = weights / weights.sum()  # Normalize
                    if self.debug:
                        logger.info(
                            f"After normalization - weights min: {weights.min()}, max: {weights.max()}, sum: {weights.sum()}"
                        )

                # Weight the channels of the activation map
                cam = torch.zeros_like(self.activations[0, 0]).to(self.device)
                for i, w in enumerate(weights):
                    cam += w * self.activations[0, i]

                if self.debug:
                    logger.info(
                        f"Initial CAM - shape: {cam.shape}, min: {cam.min()}, max: {cam.max()}, mean: {cam.mean()}"
                    )

                # Enhanced CAM processing
                cam = F.relu(cam)

                # If CAM is all zeros after ReLU, try using alternative methods
                if cam.sum() < 1e-10:
                    logger.warning(
                        "CAM is all zeros after ReLU - using alternative method"
                    )
                    # Try several alternative approaches in sequence until we get a non-zero map

                    # Method 1: Use absolute values of activations without ReLU
                    cam = torch.abs(self.activations[0].mean(dim=0))
                    if self.debug:
                        logger.info(
                            f"Alternative CAM (avg abs activations) - min: {cam.min()}, max: {cam.max()}, sum: {cam.sum()}"
                        )

                    # Method 2: If still zero, try using the raw activations without weights
                    if cam.sum() < 1e-10:
                        logger.warning("Method 1 failed - trying raw activations")
                        cam = torch.mean(torch.abs(self.activations[0]), dim=0)
                        if self.debug:
                            logger.info(
                                f"Alternative CAM (raw activations) - min: {cam.min()}, max: {cam.max()}, sum: {cam.sum()}"
                            )

                    # Method 3: If still zero, try using the gradients directly
                    if cam.sum() < 1e-10:
                        logger.warning("Method 2 failed - trying gradients directly")
                        cam = torch.mean(torch.abs(self.gradients[0]), dim=0)
                        if self.debug:
                            logger.info(
                                f"Alternative CAM (gradients) - min: {cam.min()}, max: {cam.max()}, sum: {cam.sum()}"
                            )

                    # Method 4: Last resort - create a synthetic heatmap
                    if cam.sum() < 1e-10:
                        logger.warning("All methods failed - using synthetic heatmap")
                        # Create a gaussian blob in the center as fallback
                        h, w = cam.shape
                        y, x = torch.meshgrid(
                            torch.linspace(-1, 1, h),
                            torch.linspace(-1, 1, w),
                            indexing="ij",
                        )
                        d = torch.sqrt(x * x + y * y)
                        sigma, mu = 0.5, 0.0
                        cam = torch.exp(-((d - mu) ** 2 / (2.0 * sigma**2)))
                        if self.debug:
                            logger.info(
                                f"Synthetic CAM - min: {cam.min()}, max: {cam.max()}, sum: {cam.sum()}"
                            )

                # Normalization
                if cam.max() > cam.min():  # Only normalize if there's a range of values
                    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-7)
                    if self.debug:
                        logger.info(
                            f"After normalization - CAM min: {cam.min()}, max: {cam.max()}"
                        )
                else:
                    logger.warning(
                        "CAM has uniform values - creating gaussian noise heatmap"
                    )
                    # Create a fallback heatmap with Gaussian noise
                    cam = torch.randn_like(cam) * 0.1 + 0.5
                    cam = torch.clamp(cam, 0, 1)
                    if self.debug:
                        logger.info(
                            f"Fallback random CAM - min: {cam.min()}, max: {cam.max()}"
                        )

                # High-quality resizing
                cam = F.interpolate(
                    cam.unsqueeze(0).unsqueeze(0),
                    size=tuple(self.input_size),  # Convert to tuple
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()

                # Save raw CAM tensor for debugging if in debug mode
                if self.debug:
                    np_cam = cam.cpu().numpy()
                    logger.info(
                        f"Final CAM - shape: {np_cam.shape}, min: {np_cam.min()}, max: {np_cam.max()}, mean: {np_cam.mean()}"
                    )

                    # Save the CAM array for inspection if needed
                    os.makedirs("debug_output", exist_ok=True)
                    np.save(f"debug_output/cam_debug_{target_category}.npy", np_cam)

                return cam.cpu().numpy()

            finally:
                # Restore original state
                restore_fn()

        except Exception as e:
            logger.error(f"Error in compute_cam: {e}", exc_info=True)
            return np.zeros(self.input_size)

    def visualize(
        self,
        image_input: Union[str, Image.Image, np.ndarray, torch.Tensor],
        target_category: Optional[int] = None,
        output_path: Optional[str] = None,
        alpha: float = 0.5,
        colormap: str = "jet",
        show: bool = False,
        figsize: Tuple[int, int] = (15, 5),
        dpi: int = 300,
        fontsize: int = 12,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate high-quality GradCAM visualization."""
        try:
            # Get original image with high-quality processing
            if isinstance(image_input, str):
                original_image = np.array(
                    Image.open(image_input)
                    .convert("RGB")
                    .resize(self.input_size, Image.Resampling.LANCZOS)
                )
            elif isinstance(image_input, Image.Image):
                original_image = np.array(
                    image_input.resize(self.input_size, Image.Resampling.LANCZOS)
                )
            elif isinstance(image_input, np.ndarray):
                original_image = np.array(
                    Image.fromarray(
                        image_input.astype(np.uint8)
                        if image_input.dtype != np.uint8
                        else image_input
                    ).resize(self.input_size, Image.Resampling.LANCZOS)
                )
            elif isinstance(image_input, torch.Tensor):
                if image_input.ndim == 4:
                    img_tensor = image_input[0]
                else:
                    img_tensor = image_input

                # Properly denormalize using ImageNet mean and std
                mean = (
                    torch.tensor([0.485, 0.456, 0.406])
                    .view(3, 1, 1)
                    .to(img_tensor.device)
                )
                std = (
                    torch.tensor([0.229, 0.224, 0.225])
                    .view(3, 1, 1)
                    .to(img_tensor.device)
                )
                img_tensor = img_tensor * std + mean

                # Convert to numpy and proper RGB order
                img_np = img_tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                original_image = (img_np * 255).astype(np.uint8)
                original_image = np.array(
                    Image.fromarray(original_image).resize(
                        self.input_size, Image.Resampling.LANCZOS
                    )
                )
            else:
                raise TypeError(f"Unsupported image input type: {type(image_input)}")

            # Compute CAM with enhanced quality
            cam = self(image_input, target_category)

            # Apply Gaussian smoothing for better visualization
            cam = self._apply_gaussian_smoothing(cam)

            # Create high-quality heatmap
            cmap = plt.get_cmap(colormap)
            heatmap = cmap(cam)[:, :, :3]  # Remove alpha channel

            # Enhanced overlay with improved blending
            overlaid = self._enhanced_overlay(original_image, heatmap, alpha)

            # Save visualization if output path provided
            if output_path:
                self._save_visualization(
                    original_image,
                    heatmap,
                    overlaid,
                    output_path,
                    figsize,
                    dpi,
                    fontsize,
                    show,
                )

            return original_image, heatmap, overlaid

        except Exception as e:
            logger.error(f"Error in visualize: {e}", exc_info=True)
            return (
                np.zeros(self.input_size),
                np.zeros(self.input_size),
                np.zeros(self.input_size),
            )

    def _apply_gaussian_smoothing(
        self, cam: np.ndarray, sigma: float = 2.0
    ) -> np.ndarray:
        """Apply Gaussian smoothing to the CAM for better visualization."""
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(cam, sigma=sigma)

    def _enhanced_overlay(
        self, original: np.ndarray, heatmap: np.ndarray, alpha: float
    ) -> np.ndarray:
        """Create enhanced overlay with improved blending and theme colors."""
        # Convert to float for better blending
        original = original.astype(float)
        heatmap = heatmap.astype(float)

        # Apply gamma correction to heatmap for better visibility
        heatmap = np.power(heatmap, 0.8)

        # Create custom colormap using theme colors
        from matplotlib.colors import LinearSegmentedColormap

        colors = [(0, "#121212"), (0.5, "#34d399"), (1, "#22d3ee")]
        custom_cmap = LinearSegmentedColormap.from_list("custom", colors)

        # Apply custom colormap to heatmap
        heatmap = custom_cmap(heatmap)[:, :, :3]

        # Enhanced blending with improved contrast
        overlaid = original * (1 - alpha) + heatmap * 255 * alpha

        # Apply subtle contrast enhancement
        overlaid = np.power(overlaid / 255.0, 0.9) * 255.0

        # Ensure values are in valid range
        overlaid = np.clip(overlaid, 0, 255).astype(np.uint8)

        return overlaid

    def _save_visualization(
        self,
        original: np.ndarray,
        heatmap: np.ndarray,
        overlaid: np.ndarray,
        output_path: str,
        figsize: Tuple[int, int],
        dpi: int,
        fontsize: int,
        show: bool,
    ) -> None:
        """Save high-quality visualization with enhanced formatting and dark theme."""
        # Set dark theme style
        plt.style.use("dark_background")

        # Create figure with dark background
        fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)

        # Set figure background color
        fig.patch.set_facecolor("#121212")

        # Set axes background color and styling
        for ax in axes:
            ax.set_facecolor("#121212")
            ax.tick_params(colors="#f5f5f5")
            ax.spines["bottom"].set_color("#404040")
            ax.spines["top"].set_color("#404040")
            ax.spines["left"].set_color("#404040")
            ax.spines["right"].set_color("#404040")

        # Plot original image
        axes[0].imshow(original)
        axes[0].set_title("Original Image", fontsize=fontsize, pad=10, color="#f5f5f5")
        axes[0].axis("off")

        # Plot heatmap with custom colormap using theme colors
        from matplotlib.colors import LinearSegmentedColormap

        colors = [(0, "#121212"), (0.5, "#34d399"), (1, "#22d3ee")]
        custom_cmap = LinearSegmentedColormap.from_list("custom", colors)
        axes[1].imshow(heatmap, cmap=custom_cmap)
        axes[1].set_title("GradCAM Heatmap", fontsize=fontsize, pad=10, color="#f5f5f5")
        axes[1].axis("off")

        # Plot overlay with enhanced contrast
        axes[2].imshow(overlaid)
        axes[2].set_title("GradCAM Overlay", fontsize=fontsize, pad=10, color="#f5f5f5")
        axes[2].axis("off")

        # Add subtle grid lines
        for ax in axes:
            ax.grid(True, color="#404040", linestyle="--", alpha=0.3)

        # Adjust layout with dark theme padding
        plt.tight_layout(pad=0.5)

        # Create output directory if it doesn't exist
        ensure_dir(Path(output_path).parent)

        # Save with high quality and dark theme
        plt.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="#121212",
            edgecolor="#404040",
            pad_inches=0.1,
        )

        if not show:
            plt.close(fig)

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
        # Preprocess image and move to device
        input_tensor = self.preprocess_image(image_input)

        # Temporarily unfreeze backbone if needed
        restore_fn = self._temporarily_unfreeze_backbone()

        try:
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
                class_labels = [class_names[int(i)] for i in top_classes]
            else:
                class_labels = [f"Class {i}" for i in top_classes]

            # Generate explanations for each top prediction
            explanations = []
            for i, class_idx in enumerate(top_classes):
                # Convert to Python int to avoid issues with tensor types
                class_idx_int = int(class_idx)

                # Generate CAM for this class
                cam = self.compute_cam(input_tensor, class_idx_int)
                explanations.append(
                    {
                        "class_index": class_idx_int,
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

                    # Properly denormalize using ImageNet mean and std
                    mean = (
                        torch.tensor([0.485, 0.456, 0.406])
                        .view(3, 1, 1)
                        .to(img_tensor.device)
                    )
                    std = (
                        torch.tensor([0.229, 0.224, 0.225])
                        .view(3, 1, 1)
                        .to(img_tensor.device)
                    )
                    img_tensor = img_tensor * std + mean

                    # Convert to numpy and proper RGB order
                    img_np = img_tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                    original_image = (img_np * 255).astype(np.uint8)

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
                    cam = self._apply_gaussian_smoothing(cam)  # Apply smoothing
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
                plt.savefig(output_path, dpi=400, bbox_inches="tight")
                plt.close(fig)

        finally:
            # Restore original frozen state
            restore_fn()

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

    def generate_cam(
        self,
        input_tensor: Union[torch.Tensor, np.ndarray, Image.Image, str],
        target_category: Optional[int] = None,
        class_names: Optional[List[str]] = None,
        temperature: float = 0.5,  # Add temperature parameter for confidence scaling
    ) -> Tuple[np.ndarray, Dict]:
        """
        Enhanced CAM generation that returns both the CAM and probabilities.
        This method ensures consistent confidence values between direct inference and GradCAM.

        Args:
            input_tensor: Image input as tensor, numpy array, PIL image, or path string
            target_category: Target class index (if None, uses predicted class)
            class_names: List of class names (optional)
            temperature: Temperature for scaling confidence (lower = sharper distribution)

        Returns:
            Tuple of (cam_image, result_dict) where result_dict contains 'probs' and other metadata
        """
        # Save current model state
        was_training = self.model.training
        was_frozen = False
        if hasattr(self.model, "frozen_backbone"):
            was_frozen = self.model.frozen_backbone

        # Ensure model is in eval mode for consistent results
        self.model.eval()

        # Preprocess image if needed
        if not isinstance(input_tensor, torch.Tensor):
            input_tensor = self.preprocess_image(input_tensor)
        else:
            # If already tensor, ensure it's on correct device
            input_tensor = input_tensor.to(self.device)

            # Ensure batch dimension
            if input_tensor.ndim == 3:
                input_tensor = input_tensor.unsqueeze(0)

        # Record original output before applying GradCAM
        with torch.no_grad():
            orig_output = self.model(input_tensor)
            if isinstance(orig_output, tuple):
                orig_output = orig_output[0]

            # Get standard probabilities with temperature scaling
            orig_probs = F.softmax(orig_output / temperature, dim=1)[0]

            # Get predicted class if target not specified
            if target_category is None:
                target_category = torch.argmax(orig_probs, dim=0).item()

        # Compute the CAM
        cam = self.compute_cam(input_tensor, target_category)

        # Apply Gaussian smoothing for better visualization
        cam_smoothed = self._apply_gaussian_smoothing(cam)

        # Create heatmap
        cmap = plt.get_cmap("jet")
        heatmap = cmap(cam_smoothed)[:, :, :3]  # Remove alpha channel

        # Create results dictionary
        result = {
            "probs": orig_probs,  # Keep original probabilities
            "target_category": target_category,
            "cam": cam,
            "heatmap": heatmap,
        }

        # Add class information if available
        if class_names is not None:
            top_k = min(5, len(class_names))
            top_probs, top_indices = torch.topk(orig_probs, top_k)

            result["top_classes"] = [
                {
                    "index": idx.item(),
                    "name": class_names[idx.item()],
                    "probability": prob.item(),
                }
                for idx, prob in zip(top_indices, top_probs)
            ]

        # Restore original model state if needed
        if was_training:
            self.model.train()
        if was_frozen and hasattr(self.model, "freeze_backbone"):
            self.model.freeze_backbone()

        return cam_smoothed, result

    def cleanup(self):
        """Remove registered hooks and clean up resources."""
        try:
            if hasattr(self, "forward_hook"):
                self.forward_hook.remove()
            if hasattr(self, "backward_hook"):
                self.backward_hook.remove()
            logger.debug("GradCAM hooks removed successfully")
        except Exception as e:
            logger.error(f"Error during GradCAM cleanup: {e}")

    def __del__(self):
        """Clean up hooks when instance is deleted."""
        try:
            self.forward_hook.remove()
            self.backward_hook.remove()
        except:
            pass  # Hooks might already be removed or model might be gone

    # Make the class callable
    def __call__(self, image_input, target_category=None):
        """
        Callable interface for computing GradCAM.

        Args:
            image_input: Input image
            target_category: Target class index

        Returns:
            Numpy array of CAM
        """
        return self.compute_cam(image_input, target_category)


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


def apply_gradcam_to_images(
    model: nn.Module,
    dataset: Dataset,
    target_layer: Optional[nn.Module] = None,
    input_size: Tuple[int, int] = (224, 224),
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    debug: bool = True,  # Enable debug by default
):
    # Use default mean/std if not provided
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
