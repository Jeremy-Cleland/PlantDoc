"""
Visualization tools for attention mechanisms in CBAM models.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from core.models.attention import ChannelAttention, SpatialAttention
from utils.logger import get_logger
from utils.paths import ensure_dir

logger = get_logger(__name__)


class CBAMVisualizer:
    """
    Visualizer for Convolutional Block Attention Module (CBAM) attention maps.

    This class allows visualization of both channel and spatial attention maps
    from CBAM modules in a model.

    Args:
        model: PyTorch model containing CBAM attention modules
        input_size: Size to which input images will be resized (height, width)
        mean: Mean values for image normalization
        std: Std values for image normalization
    """

    def __init__(
        self,
        model: nn.Module,
        input_size: Tuple[int, int] = (224, 224),
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
    ):
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device

        # Store preprocessing parameters
        self.input_size = input_size
        self.mean = mean
        self.std = std

        # Track attention modules and hooks
        self.attention_modules = {}
        self.hooks = []
        self.attention_outputs = {}

        # Find and register hooks for all attention modules
        self._register_attention_hooks()

        logger.info(
            f"CBAMVisualizer initialized with {len(self.attention_modules)} attention modules"
        )

    def _register_attention_hooks(self):
        """Find and register hooks for all attention modules in the model."""
        # Reset lists
        self.attention_modules = {}
        self.hooks = []
        self.attention_outputs = {}

        # Find all attention modules
        for name, module in self.model.named_modules():
            if isinstance(module, ChannelAttention):
                self.attention_modules[f"{name}_channel"] = module
            elif isinstance(module, SpatialAttention):
                self.attention_modules[f"{name}_spatial"] = module

        # Register hooks for each module
        for name, module in self.attention_modules.items():
            hook = module.register_forward_hook(
                lambda m, input, output, name=name: self._attention_hook(
                    m, input, output, name
                )
            )
            self.hooks.append(hook)

        logger.info(
            f"Registered hooks for {len(self.attention_modules)} attention modules"
        )

    def _attention_hook(self, module, input, output, name):
        """Hook function to capture attention maps."""
        # Store attention maps
        self.attention_outputs[name] = output.detach()

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
                        "Batch size > 1 detected, using only the first image"
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

    def _get_original_image(
        self, image_input: Union[str, Image.Image, np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Get original image for visualization.

        Args:
            image_input: Image input as path string, PIL Image, numpy array, or tensor

        Returns:
            Original image as numpy array
        """
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
            for i in range(3):
                img_np[:, :, i] = img_np[:, :, i] * self.std[i] + self.mean[i]
            original_image = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        return original_image

    def get_attention_maps(
        self, image_input: Union[str, Image.Image, np.ndarray, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Get attention maps for an input image.

        Args:
            image_input: Image input as path string, PIL Image, numpy array, or tensor

        Returns:
            Dictionary of attention maps (name -> tensor)
        """
        # Clear previous outputs
        self.attention_outputs = {}

        # Preprocess image
        input_tensor = self.preprocess_image(image_input)

        # Forward pass through model
        with torch.no_grad():
            _ = self.model(input_tensor)

        # Return attention maps
        return self.attention_outputs

    def visualize_attention(
        self,
        image_input: Union[str, Image.Image, np.ndarray, torch.Tensor],
        output_dir: Optional[Union[str, Path]] = None,
        layer_filter: Optional[str] = None,
        alpha: float = 0.5,
        colormap: str = "viridis",
        figsize: Tuple[int, int] = (16, 12),
    ) -> Dict[str, plt.Figure]:
        """
        Visualize attention maps for an input image.

        Args:
            image_input: Image input as path string, PIL Image, numpy array, or tensor
            output_dir: Directory to save visualizations
            layer_filter: Filter string to select specific attention modules
            alpha: Blending factor for the heatmap overlay
            colormap: Matplotlib colormap name for the heatmap
            figsize: Figure size

        Returns:
            Dictionary of figure objects (name -> figure)
        """
        # Get original image
        original_image = self._get_original_image(image_input)

        # Get attention maps
        attention_maps = self.get_attention_maps(image_input)

        # Filter modules if needed
        if layer_filter:
            attention_maps = {
                name: value
                for name, value in attention_maps.items()
                if layer_filter in name
            }

        # Create output directory if needed
        if output_dir:
            output_dir = Path(output_dir)
            ensure_dir(output_dir)

        # Initialize figures dictionary
        figures = {}

        # Create separate figure for each attention map
        for name, attention_map in attention_maps.items():
            # Process attention map based on type
            if "_channel" in name:
                # Channel attention (C, 1, 1) - needs special handling
                # Convert to spatial map for visualization
                # Create a heatmap for each channel's attention weight
                channel_weights = attention_map.squeeze().cpu().numpy()

                # Sort channels by attention weight for better visualization
                sorted_indices = np.argsort(channel_weights)[::-1]  # Descending order
                sorted_weights = channel_weights[sorted_indices]

                # Create bar chart figure
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.bar(
                    range(len(sorted_weights)),
                    sorted_weights,
                    color=plt.cm.get_cmap(colormap)(
                        np.linspace(0, 1, len(sorted_weights))
                    ),
                )
                ax.set_xlabel("Channel Index (sorted by attention weight)")
                ax.set_ylabel("Attention Weight")
                ax.set_title(f"Channel Attention Weights: {name}")

                # Save figure
                if output_dir:
                    fig.savefig(
                        output_dir / f"{name}_channel_weights.png", bbox_inches="tight"
                    )

                figures[f"{name}_weights"] = fig

                # Create a figure showing the effect of channel attention on features
                # This is a simplified approximation since we don't have direct access to feature maps
                fig, axes = plt.subplots(1, 2, figsize=figsize)

                # Original image
                axes[0].imshow(original_image)
                axes[0].set_title("Original Image")
                axes[0].axis("off")

                # Create a visualization of channel attention effect
                # This is an approximation since we don't have intermediate feature maps
                # Here we just show the attention weights applied to RGB channels for demonstration
                # In a real model, there would be many more channels

                # Create a colored overlay based on attention weights for first 3 channels (if available)
                num_vis_channels = min(3, len(channel_weights))
                channel_attention_vis = np.zeros_like(original_image, dtype=float)

                for i in range(num_vis_channels):
                    if i < 3:  # Only process RGB channels for visualization
                        channel_attention_vis[:, :, i] = (
                            channel_weights[i] * original_image[:, :, i]
                        )

                # Normalize
                if channel_attention_vis.max() > 0:
                    channel_attention_vis = (
                        channel_attention_vis / channel_attention_vis.max()
                    )

                # Show channel attention visualization
                axes[1].imshow(channel_attention_vis)
                axes[1].set_title("Channel Attention Effect (RGB approx)")
                axes[1].axis("off")

                plt.tight_layout()

                # Save figure
                if output_dir:
                    fig.savefig(
                        output_dir / f"{name}_channel_effect.png", bbox_inches="tight"
                    )

                figures[f"{name}_effect"] = fig

            elif "_spatial" in name:
                # Spatial attention (1, 1, H, W)
                # Resize to match input size for visualization
                spatial_map = F.interpolate(
                    attention_map,
                    size=self.input_size,
                    mode="bilinear",
                    align_corners=False,
                )
                spatial_map = spatial_map.squeeze().cpu().numpy()

                # Create figure
                fig, axes = plt.subplots(1, 3, figsize=figsize)

                # Original image
                axes[0].imshow(original_image)
                axes[0].set_title("Original Image")
                axes[0].axis("off")

                # Attention map
                im = axes[1].imshow(spatial_map, cmap=colormap)
                axes[1].set_title("Spatial Attention Map")
                axes[1].axis("off")
                plt.colorbar(im, ax=axes[1])

                # Overlay
                cmap = plt.get_cmap(colormap)
                heatmap = cmap(spatial_map)[:, :, :3]
                # Apply alpha blending
                overlaid_image = original_image * (1 - alpha) + heatmap * 255 * alpha
                overlaid_image = np.clip(overlaid_image, 0, 255).astype(np.uint8)

                axes[2].imshow(overlaid_image)
                axes[2].set_title("Attention Overlay")
                axes[2].axis("off")

                plt.tight_layout()

                # Save figure
                if output_dir:
                    fig.savefig(output_dir / f"{name}_spatial.png", bbox_inches="tight")

                figures[f"{name}_spatial"] = fig

        # Create a summary figure of all spatial attention maps
        spatial_maps = [
            (name, attention_map)
            for name, attention_map in attention_maps.items()
            if "_spatial" in name
        ]

        if spatial_maps:
            # Determine grid size
            n = len(spatial_maps)
            nrows = int(np.ceil(np.sqrt(n + 1)))  # +1 for the original image
            ncols = int(np.ceil((n + 1) / nrows))

            # Create figure
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))

            # Flatten axes array (handles the case when nrows or ncols is 1)
            if nrows * ncols == 1:
                axes = np.array([axes])
            axes = axes.flatten()

            # Original image
            axes[0].imshow(original_image)
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            # Spatial attention maps
            for i, (name, attention_map) in enumerate(spatial_maps, 1):
                if i < len(axes):
                    # Resize to match input size for visualization
                    spatial_map = F.interpolate(
                        attention_map,
                        size=self.input_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                    spatial_map = spatial_map.squeeze().cpu().numpy()

                    # Apply colormap
                    cmap = plt.get_cmap(colormap)
                    heatmap = cmap(spatial_map)[:, :, :3]

                    # Apply alpha blending
                    overlaid_image = (
                        original_image * (1 - alpha) + heatmap * 255 * alpha
                    )
                    overlaid_image = np.clip(overlaid_image, 0, 255).astype(np.uint8)

                    # Show overlay
                    axes[i].imshow(overlaid_image)
                    axes[i].set_title(name.split("_spatial")[0])
                    axes[i].axis("off")

            # Turn off any unused axes
            for i in range(len(spatial_maps) + 1, len(axes)):
                axes[i].axis("off")

            plt.tight_layout()

            # Save figure
            if output_dir:
                fig.savefig(
                    output_dir / "all_spatial_attention_maps.png", bbox_inches="tight"
                )

            figures["all_spatial"] = fig

        return figures

    def analyze_channel_attention(
        self,
        image_input: Union[str, Image.Image, np.ndarray, torch.Tensor],
        output_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Analyze channel attention weights for an input image.

        Args:
            image_input: Image input as path string, PIL Image, numpy array, or tensor
            output_path: Path to save analysis results

        Returns:
            Dictionary of channel attention statistics
        """
        # Get attention maps
        attention_maps = self.get_attention_maps(image_input)

        # Extract channel attention maps
        channel_attention_maps = {
            name: attention_map
            for name, attention_map in attention_maps.items()
            if "_channel" in name
        }

        # Initialize results
        analysis = {}

        # Analyze each channel attention map
        for name, attention_map in channel_attention_maps.items():
            # Extract channel weights
            channel_weights = attention_map.squeeze().cpu().numpy()

            # Calculate statistics
            stats = {
                "mean": np.mean(channel_weights),
                "std": np.std(channel_weights),
                "min": np.min(channel_weights),
                "max": np.max(channel_weights),
                "median": np.median(channel_weights),
                "top_5_channels": np.argsort(channel_weights)[-5:].tolist(),
                "top_5_weights": np.sort(channel_weights)[-5:].tolist(),
            }

            # Store results
            analysis[name] = stats

        # Save results if output path provided
        if output_path:
            output_path = Path(output_path)
            ensure_dir(output_path.parent)

            with open(output_path, "w") as f:
                f.write(json.dumps(analysis, indent=2))

        return analysis

    def cleanup(self):
        """Remove hooks when done to prevent memory leaks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        logger.info("CBAMVisualizer hooks removed")

    def __del__(self):
        """Clean up hooks when instance is deleted."""
        try:
            for hook in self.hooks:
                hook.remove()
            self.hooks = []
        except Exception:
            pass  # Hooks might already be removed or model might be gone


def visualize_cbam_attention(
    model: nn.Module,
    image_path: str,
    output_dir: Optional[Union[str, Path]] = None,
    layer_filter: Optional[str] = None,
) -> Dict[str, plt.Figure]:
    """
    Convenience function to visualize CBAM attention maps for a single image.

    Args:
        model: PyTorch model containing CBAM attention modules
        image_path: Path to the input image
        output_dir: Directory to save visualizations
        layer_filter: Filter string to select specific attention modules

    Returns:
        Dictionary of figure objects (name -> figure)
    """
    # Initialize visualizer
    visualizer = CBAMVisualizer(model)

    # Generate visualizations
    figures = visualizer.visualize_attention(
        image_path, output_dir=output_dir, layer_filter=layer_filter
    )

    # Clean up
    visualizer.cleanup()

    return figures


def run_cbam_attention_visualization(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_samples: int = 5,
    output_dir: Optional[Union[str, Path]] = None,
    class_names: Optional[List[str]] = None,
) -> None:
    """
    Run CBAM attention visualization for multiple samples from a dataset.

    Args:
        model: PyTorch model containing CBAM attention modules
        dataloader: DataLoader for evaluation data
        num_samples: Number of sample images to visualize
        output_dir: Directory to save visualizations
        class_names: List of class names
    """
    # Set model to evaluation mode
    model.eval()

    # Create output directory
    if output_dir:
        output_dir = Path(output_dir)
        ensure_dir(output_dir)

    # Initialize visualizer
    visualizer = CBAMVisualizer(model)

    # Collect sample images
    samples_processed = 0

    logger.info(f"Visualizing CBAM attention for {num_samples} samples")

    for batch in dataloader:
        if samples_processed >= num_samples:
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

        # Process each image in batch
        for i in range(len(images)):
            if samples_processed >= num_samples:
                break

            # Get image and class info
            image = images[i]
            target = targets[i].item()
            target_name = (
                class_names[target]
                if class_names and target < len(class_names)
                else f"Class {target}"
            )

            # Create sample-specific output directory
            sample_dir = output_dir / f"sample_{samples_processed:02d}_{target_name}"
            ensure_dir(sample_dir)

            # Visualize attention maps
            try:
                logger.info(
                    f"Visualizing attention for sample {samples_processed + 1}/{num_samples}: {target_name}"
                )
                _ = visualizer.visualize_attention(image, output_dir=sample_dir)

                # Save original image for reference
                if isinstance(image, torch.Tensor):
                    # Denormalize and convert to PIL
                    img_np = image.permute(1, 2, 0).cpu().numpy()
                    for c in range(3):
                        img_np[:, :, c] = (
                            img_np[:, :, c] * visualizer.std[c] + visualizer.mean[c]
                        )
                    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
                    Image.fromarray(img_np).save(sample_dir / "original.png")

                samples_processed += 1
            except Exception as e:
                logger.error(
                    f"Error visualizing attention for sample {samples_processed}: {e}"
                )

    # Clean up
    visualizer.cleanup()

    logger.info(
        f"CBAM attention visualization completed for {samples_processed} samples"
    )
