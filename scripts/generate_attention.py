#!/usr/bin/env python
"""
Generate attention visualizations for CBAM model layers.
This script extracts and visualizes attention maps from CBAM attention modules
in the model and saves them to the attention_visualizations directory.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from core.models import get_model
from utils.logger import get_logger
from utils.paths import ensure_dir

logger = get_logger(__name__)


class AttentionExtractor:
    """
    Extract and visualize attention maps from CBAM modules in a model.

    Args:
        model: The PyTorch model containing CBAM modules
        device: Device to run the model on ('cpu', 'cuda', 'mps')
        output_dir: Directory to save visualizations
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        output_dir: Optional[Path] = None,
    ):
        self.model = model
        self.device = device
        self.output_dir = output_dir or Path("attention_visualizations")
        ensure_dir(self.output_dir)

        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()

        # Dictionary to store attention maps
        self.attention_maps = {}

        # Register hooks to capture attention maps
        self.hooks = self._register_hooks()

    def _register_hooks(self) -> List:
        """
        Register forward hooks to capture attention maps from CBAM modules.
        """
        hooks = []

        def get_activation(name):
            def hook(module, input, output):
                # Save activations to the attention_maps dictionary
                if isinstance(output, tuple):
                    # Some modules return multiple outputs
                    self.attention_maps[name] = output[0].detach().cpu()
                else:
                    self.attention_maps[name] = output.detach().cpu()

            return hook

        # Try multiple approaches to find attention modules
        attention_found = False

        # Approach 0: First try to directly access CBAM modules by checking for ca/sa attributes
        # This is specific to our implementation structure
        for name, module in self.model.named_modules():
            # Check for BasicBlock with ca/sa attributes (our CBAM implementation pattern)
            if hasattr(module, "ca") and hasattr(module, "sa"):
                logger.info(f"Found CBAM block with ca/sa: {name}")

                # Register hooks for channel attention
                ca_module = getattr(module, "ca")
                if isinstance(ca_module, nn.Module):
                    hooks.append(
                        ca_module.register_forward_hook(
                            get_activation(f"{name}_channel")
                        )
                    )
                    attention_found = True

                # Register hooks for spatial attention
                sa_module = getattr(module, "sa")
                if isinstance(sa_module, nn.Module):
                    hooks.append(
                        sa_module.register_forward_hook(
                            get_activation(f"{name}_spatial")
                        )
                    )
                    attention_found = True

        # Approach 1: Look for standard CBAM modules with channel and spatial attention
        if not attention_found:
            for name, module in self.model.named_modules():
                if hasattr(module, "channel_attention") and hasattr(
                    module, "spatial_attention"
                ):
                    logger.info(f"Found CBAM module: {name}")

                    # Register hooks for channel attention
                    if isinstance(module.channel_attention, nn.Module):
                        hooks.append(
                            module.channel_attention.register_forward_hook(
                                get_activation(f"{name}_channel")
                            )
                        )
                        attention_found = True

                    # Register hooks for spatial attention
                    if isinstance(module.spatial_attention, nn.Module):
                        hooks.append(
                            module.spatial_attention.register_forward_hook(
                                get_activation(f"{name}_spatial")
                            )
                        )
                        attention_found = True

        # Approach 2: Look for modules with "attention" in their name
        if not attention_found:
            for name, module in self.model.named_modules():
                if any(
                    keyword in name.lower() for keyword in ["attention", "cbam", "se"]
                ):
                    logger.info(f"Found potential attention module: {name}")
                    hooks.append(module.register_forward_hook(get_activation(name)))
                    attention_found = True

        # Approach 3: Look for common module types that might be attention-related
        if not attention_found:
            for name, module in self.model.named_modules():
                # Look for specific layer types that are commonly used in attention
                is_attention_candidate = False

                # Check if it's a sequential module containing both conv and sigmoid
                if isinstance(module, nn.Sequential):
                    has_conv = any(isinstance(m, nn.Conv2d) for m in module.children())
                    has_sigmoid = any(
                        isinstance(m, nn.Sigmoid) for m in module.children()
                    )
                    if has_conv and has_sigmoid:
                        is_attention_candidate = True

                # Check for modules that have global pooling or self.avgpool attribute
                elif hasattr(module, "avgpool") or hasattr(module, "global_pool"):
                    is_attention_candidate = True

                # Register hooks for attention candidates
                if is_attention_candidate:
                    logger.info(f"Found potential attention structure: {name}")
                    hooks.append(
                        module.register_forward_hook(
                            get_activation(f"attention_{name}")
                        )
                    )
                    attention_found = True

        # Approach 4: If all else fails, monitor feature layers of ResNet backbone
        if not attention_found:
            logger.warning(
                "No attention modules found. Monitoring backbone feature layers instead."
            )

            # Try to access directly if model has backbone attribute
            if hasattr(self.model, "backbone"):
                backbone = self.model.backbone
                logger.info("Found backbone attribute in model")

                # Look for layer blocks
                for idx in range(1, 5):  # Layers 1-4 in ResNet
                    layer_name = f"layer{idx}"
                    if hasattr(backbone, layer_name):
                        layer = getattr(backbone, layer_name)
                        # Hook the output of each layer
                        hooks.append(
                            layer.register_forward_hook(
                                get_activation(f"backbone_{layer_name}")
                            )
                        )
                        logger.info(f"Monitoring backbone {layer_name}")
            else:
                # Look for bottleneck or key feature layers
                bottleneck_keywords = ["layer3", "layer4", "bottleneck", "features"]
                for name, module in self.model.named_modules():
                    if any(keyword in name.lower() for keyword in bottleneck_keywords):
                        if isinstance(module, (nn.Conv2d, nn.Sequential)):
                            logger.info(f"Monitoring bottleneck feature layer: {name}")
                            hooks.append(
                                module.register_forward_hook(
                                    get_activation(f"features_{name}")
                                )
                            )

        if not hooks:
            logger.warning("Could not find any suitable modules to visualize")

            # Last resort - just capture any Conv2d modules
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Conv2d) and "layer" in name.lower():
                    logger.info(f"Monitoring Conv2d layer: {name}")
                    hooks.append(
                        module.register_forward_hook(get_activation(f"conv_{name}"))
                    )
                    # Limit to a few key layers to avoid too many visualizations
                    if len(hooks) >= 5:
                        break

        return hooks

    def _preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess an image for the model.

        Args:
            image_path: Path to the image

        Returns:
            Preprocessed tensor
        """
        # Load image
        img = Image.open(image_path).convert("RGB")

        # Define transform
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Apply transform and add batch dimension
        img_tensor = preprocess(img).unsqueeze(0).to(self.device)

        return img_tensor

    def extract_attention_maps(self, image_path: str) -> Dict[str, torch.Tensor]:
        """
        Extract attention maps for an image.

        Args:
            image_path: Path to the image

        Returns:
            Dictionary with attention maps
        """
        # Clear previous attention maps
        self.attention_maps = {}

        # Preprocess image
        img_tensor = self._preprocess_image(image_path)

        # Forward pass
        with torch.no_grad():
            _ = self.model(img_tensor)

        if not self.attention_maps:
            logger.warning("No attention maps were captured during forward pass")

        return self.attention_maps

    def visualize_attention_maps(
        self,
        image_path: str,
        attention_maps: Optional[Dict[str, torch.Tensor]] = None,
        output_path: Optional[str] = None,
    ) -> None:
        """
        Visualize attention maps for an image.

        Args:
            image_path: Path to the image
            attention_maps: Dictionary with attention maps
            output_path: Path to save the visualization
        """
        # Use provided attention maps or extract them
        maps = attention_maps or self.extract_attention_maps(image_path)

        if not maps:
            logger.error("No attention maps to visualize")
            return

        try:
            # Load original image
            original_img = Image.open(image_path).convert("RGB")
            original_img = original_img.resize((224, 224))

            # Determine number of maps to display and figure layout
            n_maps = min(len(maps), 9)  # Limit to 9 maps maximum
            n_cols = min(3, n_maps)
            n_rows = (n_maps + n_cols - 1) // n_cols + 1  # +1 for original image

            # Create figure
            fig = plt.figure(figsize=(n_cols * 4, n_rows * 4))
            fig.suptitle("Attention Visualizations", fontsize=16)

            # Plot original image
            plt.subplot(n_rows, n_cols, 1)
            plt.imshow(original_img)
            plt.title("Original Image")
            plt.axis("off")

            # Process and plot each attention map
            for i, (name, attention) in enumerate(list(maps.items())[:n_maps], 2):
                plt.subplot(n_rows, n_cols, i)

                try:
                    # Preprocess attention map based on its type
                    attention_tensor = attention

                    # Handle 4D tensors (B, C, H, W)
                    if attention_tensor.dim() == 4:
                        attention_tensor = attention_tensor[0]  # Take first batch

                    # For channel attention (1D), make it a bar chart
                    if attention_tensor.dim() == 1 or (
                        attention_tensor.dim() == 3
                        and (
                            attention_tensor.size(1) == 1
                            and attention_tensor.size(2) == 1
                        )
                    ):
                        if attention_tensor.dim() == 3:
                            attention_tensor = attention_tensor.squeeze()

                        # Plot as bar chart
                        plt.bar(range(len(attention_tensor)), attention_tensor.numpy())
                        plt.title(f"{name}\n(Channel Attention)")

                    # For 3D tensors with multiple channels
                    elif attention_tensor.dim() == 3:
                        # Different handling based on channel count
                        if attention_tensor.size(0) > 3:
                            # For multichannel, average across channels to get heatmap
                            attention_map = attention_tensor.mean(0).numpy()
                            plt.imshow(attention_map, cmap="hot")
                            plt.colorbar(fraction=0.046, pad=0.04)
                            plt.title(f"{name}\n(Avg Across Channels)")
                        else:
                            # For tensors with 1-3 channels, try to visualize as RGB
                            if attention_tensor.size(0) == 1:
                                # Single channel - use as grayscale
                                attention_map = attention_tensor[0].numpy()
                                plt.imshow(attention_map, cmap="gray")
                            elif attention_tensor.size(0) == 3:
                                # RGB channels
                                attention_map = attention_tensor.permute(
                                    1, 2, 0
                                ).numpy()
                                attention_map = (
                                    attention_map - attention_map.min()
                                ) / (attention_map.max() - attention_map.min() + 1e-8)
                                plt.imshow(attention_map)
                            else:
                                # 2 channels - use first as R, second as G
                                attention_map = torch.zeros(
                                    (
                                        3,
                                        attention_tensor.size(1),
                                        attention_tensor.size(2),
                                    )
                                )
                                attention_map[0] = attention_tensor[0]  # R
                                attention_map[1] = attention_tensor[1]  # G
                                attention_map = attention_map.permute(1, 2, 0).numpy()
                                plt.imshow(attention_map)

                            plt.title(f"{name}\n(Feature Map)")

                    # For 2D tensors (H, W)
                    elif attention_tensor.dim() == 2:
                        # This is a classic spatial attention map
                        attention_map = attention_tensor.numpy()
                        plt.imshow(attention_map, cmap="hot")
                        plt.colorbar(fraction=0.046, pad=0.04)
                        plt.title(f"{name}\n(Spatial Attention)")

                    # Fallback for other dimensions
                    else:
                        plt.text(
                            0.5,
                            0.5,
                            f"Can't visualize\n{attention_tensor.shape}",
                            ha="center",
                            va="center",
                            fontsize=12,
                        )
                        plt.title(f"{name}\n(Unsupported Format)")

                except Exception as e:
                    logger.error(f"Error visualizing attention map {name}: {e}")
                    plt.text(
                        0.5,
                        0.5,
                        f"Error: {str(e)[:30]}...",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="red",
                    )
                    plt.title(f"{name}\n(Error)")

                plt.axis("off")

            # Save figure
            plt.tight_layout()

            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                logger.info(f"Saved attention visualization to {output_path}")

            plt.close()

        except Exception as e:
            logger.error(f"Error in attention visualization: {e}")
            # Create a simple error image
            fig = plt.figure(figsize=(6, 3))
            plt.text(
                0.5,
                0.5,
                f"Visualization Error: {str(e)}",
                ha="center",
                va="center",
                fontsize=12,
                color="red",
            )
            plt.axis("off")

            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches="tight")

            plt.close()

    def apply_attention_to_image(
        self,
        image_path: str,
        attention_maps: Optional[Dict[str, torch.Tensor]] = None,
        output_path: Optional[str] = None,
    ) -> None:
        """
        Apply attention maps as overlays on the original image.

        Args:
            image_path: Path to the image
            attention_maps: Dictionary with attention maps
            output_path: Path to save the visualization
        """
        # Use provided attention maps or extract them
        maps = attention_maps or self.extract_attention_maps(image_path)

        if not maps:
            logger.error("No attention maps to apply")
            return

        try:
            # Load original image
            original_img = Image.open(image_path).convert("RGB")
            original_img = original_img.resize((224, 224))
            original_array = np.array(original_img)

            # Filter maps to include only those suitable for spatial overlay
            # (Channel attention maps or non-2D maps don't make sense for overlay)
            spatial_maps = {}

            for name, attention in maps.items():
                attention_tensor = attention

                # Skip channel attention maps
                if "channel" in name.lower():
                    continue

                # Handle 4D tensor (B, C, H, W)
                if attention_tensor.dim() == 4:
                    attention_tensor = attention_tensor[0]  # Take first batch

                # For 3D tensor (C, H, W), take first channel or average
                if attention_tensor.dim() == 3:
                    if attention_tensor.size(0) == 1:
                        attention_tensor = attention_tensor[0]  # Single channel
                    else:
                        attention_tensor = attention_tensor.mean(0)  # Average channels

                # 2D tensor (H, W) is ready for overlay
                if attention_tensor.dim() == 2:
                    spatial_maps[name] = attention_tensor

            if not spatial_maps:
                logger.warning("No spatial attention maps found for overlay")
                # Create a simple warning image
                fig = plt.figure(figsize=(6, 3))
                plt.text(
                    0.5,
                    0.5,
                    "No suitable spatial attention maps found",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                plt.axis("off")

                if output_path:
                    plt.savefig(output_path, dpi=300, bbox_inches="tight")

                plt.close()
                return

            # Determine layout based on number of maps
            n_maps = min(len(spatial_maps), 8)  # Limit to original + 7 maps maximum
            n_cols = min(4, n_maps + 1)  # +1 for original
            n_rows = (n_maps + n_cols) // n_cols

            # Create figure
            fig = plt.figure(figsize=(n_cols * 4, n_rows * 4))
            fig.suptitle("Attention Overlays", fontsize=16)

            # Plot original image
            plt.subplot(n_rows, n_cols, 1)
            plt.imshow(original_img)
            plt.title("Original Image")
            plt.axis("off")

            # Create overlays
            for i, (name, attention) in enumerate(spatial_maps.items(), 2):
                if i > n_rows * n_cols:
                    break  # Skip if we've filled the grid

                plt.subplot(n_rows, n_cols, i)

                try:
                    # Normalize attention map
                    attn_map = attention.numpy()
                    attn_map = (attn_map - attn_map.min()) / (
                        attn_map.max() - attn_map.min() + 1e-8
                    )

                    # Resize attention map to match image size
                    from scipy.ndimage import zoom

                    height_ratio = original_array.shape[0] / attn_map.shape[0]
                    width_ratio = original_array.shape[1] / attn_map.shape[1]
                    attn_map = zoom(attn_map, (height_ratio, width_ratio), order=1)

                    # Create heatmap
                    heatmap = plt.cm.hot(attn_map)[:, :, :3]

                    # Create overlay with alpha blending
                    alpha = 0.6
                    overlay = original_array * (1 - alpha) + heatmap * 255 * alpha
                    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

                    # Display overlay
                    plt.imshow(overlay)
                    plt.title(f"{name}\nOverlay")
                    plt.axis("off")

                except Exception as e:
                    logger.error(f"Error creating overlay for {name}: {e}")
                    plt.imshow(original_img)  # Show original as fallback
                    plt.title(f"{name}\n(Error)")
                    plt.axis("off")

            # Save figure
            plt.tight_layout()

            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                logger.info(f"Saved attention overlay to {output_path}")

            plt.close()

        except Exception as e:
            logger.error(f"Error creating attention overlays: {e}")
            # Create a simple error image
            fig = plt.figure(figsize=(6, 3))
            plt.text(
                0.5,
                0.5,
                f"Overlay Error: {str(e)}",
                ha="center",
                va="center",
                fontsize=12,
                color="red",
            )
            plt.axis("off")

            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches="tight")

            plt.close()

    def process_image(self, image_path: str, output_dir: Optional[Path] = None) -> None:
        """
        Process an image to extract and visualize attention maps.

        Args:
            image_path: Path to the image
            output_dir: Directory to save visualizations
        """
        save_dir = output_dir or self.output_dir
        ensure_dir(save_dir)

        # Extract attention maps
        attention_maps = self.extract_attention_maps(image_path)

        if not attention_maps:
            logger.warning(f"No attention maps extracted for {image_path}")
            return

        # Create filenames for visualizations
        image_name = Path(image_path).stem
        vis_path = save_dir / f"{image_name}_attention_maps.png"
        overlay_path = save_dir / f"{image_name}_attention_overlay.png"

        # Visualize attention maps
        self.visualize_attention_maps(image_path, attention_maps, str(vis_path))

        # Visualize attention overlays
        self.apply_attention_to_image(image_path, attention_maps, str(overlay_path))

    def cleanup(self) -> None:
        """
        Remove hooks and clean up resources.
        """
        for hook in self.hooks:
            hook.remove()

        self.hooks = []
        self.attention_maps = {}


def load_model(model_path: Path, model_name: str = "cbam_resnet18") -> nn.Module:
    """
    Load a model from a checkpoint.

    Args:
        model_path: Path to the model checkpoint
        model_name: Name of the model architecture

    Returns:
        Loaded model
    """
    try:
        # Use the new get_model function from core.models
        model = get_model(
            model_name=model_name, checkpoint_path=model_path, device="cpu"
        )

        logger.info(f"Loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise


def find_image_files(directory: Path) -> List[Path]:
    """
    Find all image files in a directory.

    Args:
        directory: Directory to search

    Returns:
        List of image file paths
    """
    image_extensions = [".jpg", ".jpeg", ".png"]
    image_files = []

    for ext in image_extensions:
        image_files.extend(list(directory.glob(f"**/*{ext}")))

    return image_files


def generate_attention_visualizations(
    experiment_dir: Path,
    num_images: int = 5,
    device: str = "cpu",
    model: Optional[nn.Module] = None,
    create_combined_visualization: bool = True,
    theme: str = "plantdoc",
) -> None:
    """
    Generate attention visualizations for a model.

    Args:
        experiment_dir: Path to the experiment directory
        num_images: Maximum number of images to process
        device: Device to run the model on
        model: Optional pre-loaded model (if not provided, will try to load from checkpoint)
    """
    # Create output directory
    output_dir = experiment_dir / "attention_visualizations"
    ensure_dir(output_dir)

    # Use provided model or load it from checkpoint
    if model is None:
        # Look for model checkpoint
        checkpoint_dir = experiment_dir / "checkpoints"
        model_path = None

        if checkpoint_dir.exists():
            # Try to find best model first
            best_model_path = checkpoint_dir / "best_model.pth"
            if best_model_path.exists():
                model_path = best_model_path
            else:
                # Look for any checkpoint
                checkpoints = list(checkpoint_dir.glob("*.pth"))
                if checkpoints:
                    model_path = checkpoints[0]

        if model_path is None:
            logger.error("No model checkpoint found in experiment directory")
            return

        # Try to determine model name from config or experiment name
        model_name = "cbam_resnet18"  # Default
        config_path = experiment_dir / "config.yaml"
        if config_path.exists():
            try:
                import yaml

                with open(config_path) as f:
                    config = yaml.safe_load(f)
                if "model" in config and "name" in config["model"]:
                    model_name = config["model"]["name"]
            except Exception as e:
                logger.error(f"Error loading config: {e}")

        try:
            # Load model using the new get_model function
            model = get_model(
                model_name=model_name, checkpoint_path=model_path, device=device
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return

    # Initialize attention extractor
    extractor = AttentionExtractor(
        model=model,
        device=device,
        output_dir=output_dir,
    )

    # Try to find sample images
    images_found = False
    image_files = []

    # Approach 1: Look in potential image directories
    image_dirs = [
        experiment_dir / "sample_images",
        experiment_dir / "test_images",
        Path("data/raw"),
        Path("data/processed"),
        Path("data"),
    ]

    for img_dir in image_dirs:
        if img_dir.exists():
            potential_images = find_image_files(img_dir)

            if potential_images:
                images_found = True
                logger.info(f"Found {len(potential_images)} images in {img_dir}")
                image_files.extend(potential_images)
                if (
                    len(image_files) >= num_images * 3
                ):  # Get more than we need so we can select diverse samples
                    break

    # Approach 2: Use test dataset if available
    if not images_found or len(image_files) < num_images:
        try:
            from scripts.generate_visualizations import create_test_dataset

            logger.info("Attempting to load dataset for samples")
            dataset = create_test_dataset(experiment_dir)

            if dataset is not None:
                # Get random samples from dataset
                import random

                import numpy as np
                import torch
                from torchvision.utils import save_image

                logger.info(f"Created dataset with {len(dataset)} samples")

                # Try to get diverse samples (pick from different classes if possible)
                sample_indices = []

                # Attempt to get a spread of classes
                if hasattr(dataset, "classes") and len(dataset.classes) > 0:
                    # Get samples distributed across classes
                    class_samples = {}

                    # First organize by class
                    for i in range(len(dataset)):
                        if hasattr(dataset, "samples") and len(dataset.samples) > i:
                            _, class_idx = dataset.samples[i]
                        elif hasattr(dataset, "targets") and len(dataset.targets) > i:
                            class_idx = dataset.targets[i]
                        else:
                            # If we can't determine class, just use random
                            sample_indices = random.sample(
                                range(len(dataset)), min(num_images, len(dataset))
                            )
                            break

                        if class_idx not in class_samples:
                            class_samples[class_idx] = []
                        class_samples[class_idx].append(i)

                    # Then select samples from each class
                    classes = list(class_samples.keys())
                    random.shuffle(classes)  # Randomize class order

                    samples_per_class = max(1, num_images // len(classes))
                    for class_idx in classes:
                        if len(sample_indices) >= num_images:
                            break

                        # Get samples from this class
                        class_indices = class_samples[class_idx]
                        selected = random.sample(
                            class_indices, min(samples_per_class, len(class_indices))
                        )
                        sample_indices.extend(selected)

                    # If we still need more samples, add random ones
                    if len(sample_indices) < num_images:
                        remaining = num_images - len(sample_indices)
                        all_indices = list(range(len(dataset)))
                        # Remove already selected indices
                        remaining_indices = [
                            i for i in all_indices if i not in sample_indices
                        ]
                        if remaining_indices:
                            additional = random.sample(
                                remaining_indices,
                                min(remaining, len(remaining_indices)),
                            )
                            sample_indices.extend(additional)
                else:
                    # Just select random samples
                    sample_indices = random.sample(
                        range(len(dataset)), min(num_images, len(dataset))
                    )

                # Get actual samples and save as images
                temp_dir = output_dir / "dataset_samples"
                ensure_dir(temp_dir)

                logger.info(f"Extracting {len(sample_indices)} samples from dataset")
                for i, idx in enumerate(sample_indices):
                    try:
                        # Handle different dataset return types
                        sample = dataset[idx]

                        if isinstance(sample, dict):
                            # Dataset returns a dictionary
                            image = sample.get("image", None)
                        elif isinstance(sample, tuple) and len(sample) >= 1:
                            # Dataset returns a tuple (image, label)
                            image = sample[0]
                        else:
                            # Assume sample is the image
                            image = sample

                        if image is not None:
                            # Save the sample as an image file
                            img_path = temp_dir / f"sample_{i:03d}.png"

                            if isinstance(image, torch.Tensor):
                                # Handle tensor to PIL conversion
                                if image.dim() == 3:
                                    # Handle normalization if needed
                                    try:
                                        # Try to save directly
                                        save_image(image, img_path)
                                    except Exception:
                                        # If that fails, try to denormalize
                                        mean = torch.tensor([0.485, 0.456, 0.406]).view(
                                            3, 1, 1
                                        )
                                        std = torch.tensor([0.229, 0.224, 0.225]).view(
                                            3, 1, 1
                                        )
                                        image = image * std + mean
                                        image = torch.clamp(image, 0, 1)
                                        save_image(image, img_path)
                                else:
                                    logger.warning(
                                        f"Unexpected image tensor shape: {image.shape}"
                                    )
                                    continue
                            else:
                                # If it's not a tensor, try to save through PIL
                                from PIL import Image as PILImage

                                if isinstance(image, np.ndarray):
                                    PILImage.fromarray(
                                        (image * 255).astype(np.uint8)
                                    ).save(img_path)
                                elif isinstance(image, PILImage.Image):
                                    image.save(img_path)
                                else:
                                    logger.warning(
                                        f"Unexpected image type: {type(image)}"
                                    )
                                    continue

                            image_files.append(img_path)

                    except Exception as e:
                        logger.error(f"Error extracting sample {idx}: {e}")
                        continue

                if image_files:
                    images_found = True
                    logger.info(
                        f"Successfully extracted {len(image_files)} images from dataset"
                    )

        except Exception as e:
            logger.error(f"Error trying to use dataset for samples: {e}")

    # If we found images, process them
    if images_found and image_files:
        # Select a subset of images
        if len(image_files) > num_images:
            # Prioritize image diversity if possible
            try:
                # Try to select images from different directories to get diversity
                dirs = {}
                for img_path in image_files:
                    parent = img_path.parent
                    if parent not in dirs:
                        dirs[parent] = []
                    dirs[parent].append(img_path)

                # Select images from different directories
                selected_images = []
                dirs_list = list(dirs.keys())

                while len(selected_images) < num_images and dirs_list:
                    for dir_path in dirs_list.copy():
                        if dirs[dir_path]:
                            # Take one image from each directory
                            img = dirs[dir_path].pop(0)
                            selected_images.append(img)
                            if len(selected_images) >= num_images:
                                break
                        else:
                            # Remove empty directory
                            dirs_list.remove(dir_path)

                if len(selected_images) < num_images:
                    # Fall back to random selection if needed
                    remaining = [img for sublist in dirs.values() for img in sublist]
                    selected_images.extend(
                        random.sample(
                            remaining,
                            min(num_images - len(selected_images), len(remaining)),
                        )
                    )

                image_files = selected_images[:num_images]
            except Exception:
                # Fall back to simple random selection
                image_files = random.sample(image_files, num_images)

        # Process each image
        for img_path in image_files:
            logger.info(f"Processing {img_path}")
            try:
                extractor.process_image(str(img_path), output_dir)
            except Exception as e:
                logger.error(f"Error processing image {img_path}: {e}")

    # If no images found or processing failed, create random images
    if not images_found or not image_files:
        logger.warning("No suitable image files found. Creating random images.")

        for i in range(num_images):
            # Create a random RGB image
            random_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(random_img)

            # Save the image
            random_path = output_dir / f"random_image_{i}.png"
            img.save(random_path)

            # Process the image
            logger.info(f"Processing random image {i}")
            extractor.process_image(str(random_path), output_dir)

    # Clean up
    extractor.cleanup()
    logger.info(f"Attention visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate attention visualizations")
    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        required=True,
        help="Path to experiment directory",
    )
    parser.add_argument(
        "--model-name",
        "-m",
        type=str,
        default="cbam_resnet18",
        help="Model architecture name",
    )
    parser.add_argument(
        "--num-images", "-n", type=int, default=5, help="Number of images to process"
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run the model on",
    )

    args = parser.parse_args()

    # Resolve experiment directory
    experiment_dir = Path(args.experiment)
    if not experiment_dir.is_absolute():
        experiment_dir = Path.cwd() / experiment_dir

    if not experiment_dir.exists():
        logger.error(f"Experiment directory {experiment_dir} does not exist")
        return

    # Generate visualizations
    generate_attention_visualizations(
        experiment_dir=experiment_dir,
        num_images=args.num_images,
        device=args.device,
    )


if __name__ == "__main__":
    main()
