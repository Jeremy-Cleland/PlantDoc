"""
Visualization tools for CBAM attention maps and model interpretability.

This module provides functions to visualize attention maps from CBAM modules,
overlay them on input images, and generate comprehensive attention visualization reports.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from utils.logger import get_logger

logger = get_logger(__name__)


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a tensor to a numpy array.

    Args:
        tensor: Input tensor

    Returns:
        Numpy array
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()
    if tensor.requires_grad:
        tensor = tensor.detach()
    return tensor.numpy()


def plot_attention_heatmap(
    attention_map: torch.Tensor,
    title: str = "Attention Map",
    cmap: str = "viridis",
    colorbar: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot an attention map as a heatmap.

    Args:
        attention_map: Attention map tensor (H, W)
        title: Title for the plot
        cmap: Colormap to use
        colorbar: Whether to include a colorbar
        ax: Existing axes to plot on
        figsize: Figure size if creating a new figure

    Returns:
        Tuple of (figure, axes)
    """
    # Convert to numpy if needed
    if isinstance(attention_map, torch.Tensor):
        attention_map = _to_numpy(attention_map)

    # Ensure 2D
    if attention_map.ndim > 2:
        attention_map = attention_map.squeeze()

    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Plot heatmap
    im = ax.imshow(attention_map, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")

    if colorbar:
        fig.colorbar(im, ax=ax)

    return fig, ax


def visualize_attention_maps(
    attention_maps: Dict[str, torch.Tensor],
    layer_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 10),
    cmap: str = "viridis",
    output_path: Optional[str] = None,
    suptitle: str = "CBAM Attention Maps",
) -> plt.Figure:
    """
    Visualize multiple attention maps in a grid.

    Args:
        attention_maps: Dictionary of attention maps
        layer_names: List of layer names to visualize (if None, use all)
        figsize: Figure size
        cmap: Colormap to use
        output_path: Optional path to save the figure
        suptitle: Super title for the figure

    Returns:
        Matplotlib figure
    """
    # Filter layers if specified
    if layer_names is not None:
        filtered_maps = {
            k: v
            for k, v in attention_maps.items()
            if any(ln in k for ln in layer_names)
        }
        if not filtered_maps:
            logger.warning(
                f"No attention maps found matching layer names: {layer_names}"
            )
            filtered_maps = attention_maps  # Fallback to all maps
    else:
        filtered_maps = attention_maps

    num_maps = len(filtered_maps)
    if num_maps == 0:
        logger.warning("No attention maps to visualize")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No attention maps available", ha="center", va="center")
        return fig

    # Calculate grid dimensions
    ncols = min(4, num_maps)
    nrows = (num_maps + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows * ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Plot each attention map
    for i, (name, attn_map) in enumerate(filtered_maps.items()):
        if i >= len(axes):
            break

        # Get attention map
        if isinstance(attn_map, torch.Tensor):
            attn_map = _to_numpy(attn_map)

        # Ensure 2D
        if attn_map.ndim > 2:
            # For channel attention (B, C, 1, 1), reshape to a single row
            if attn_map.shape[-1] == 1 and attn_map.shape[-2] == 1:
                attn_map = attn_map.squeeze(-1).squeeze(-1)
                if attn_map.ndim > 1:  # Handle batch dimension
                    attn_map = attn_map[0]
                attn_map = attn_map.reshape(1, -1)
            else:
                attn_map = attn_map.squeeze()
                if attn_map.ndim > 2:  # Handle batch dimension
                    attn_map = attn_map[0]

        # Plot
        axes[i].imshow(attn_map, cmap=cmap)
        axes[i].set_title(name)
        axes[i].axis("off")

    # Hide unused axes
    for i in range(num_maps, len(axes)):
        axes[i].axis("off")

    plt.suptitle(suptitle)
    plt.tight_layout()

    # Save if requested
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=150)

    return fig


def visualize_attention_overlay(
    image: Union[torch.Tensor, np.ndarray, Image.Image],
    attention_map: torch.Tensor,
    alpha: float = 0.7,
    cmap: str = "jet",
    figsize: Tuple[int, int] = (10, 5),
    output_path: Optional[str] = None,
    title: str = "Attention Overlay",
) -> plt.Figure:
    """
    Visualize an attention map overlaid on an input image.

    Args:
        image: Input image (tensor, numpy array, or PIL Image)
        attention_map: Attention map tensor
        alpha: Alpha value for overlay
        cmap: Colormap for attention
        figsize: Figure size
        output_path: Optional path to save the figure
        title: Title for the figure

    Returns:
        Matplotlib figure
    """
    # Convert image to numpy if needed
    if isinstance(image, torch.Tensor):
        # Move channel dimension to the end for plotting
        image = _to_numpy(image)
        if image.shape[0] == 3:  # CHW format
            image = np.transpose(image, (1, 2, 0))
        # Normalize if needed
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
    elif isinstance(image, Image.Image):
        image = np.array(image)

    # Convert attention map to numpy
    if isinstance(attention_map, torch.Tensor):
        attention_map = _to_numpy(attention_map)

    # Ensure 2D
    if attention_map.ndim > 2:
        attention_map = attention_map.squeeze()
        if attention_map.ndim > 2:  # Handle batch dimension
            attention_map = attention_map[0]

    # Resize attention map to match image size
    if attention_map.shape != image.shape[:2]:
        # Create temporary PIL image
        attn_pil = Image.fromarray((attention_map * 255).astype(np.uint8))
        attn_pil = attn_pil.resize((image.shape[1], image.shape[0]), Image.BICUBIC)
        attention_map = np.array(attn_pil) / 255.0

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Plot original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Plot attention map
    im = axes[1].imshow(attention_map, cmap=cmap)
    axes[1].set_title("Attention Map")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Plot overlay
    axes[2].imshow(image)
    im = axes[2].imshow(attention_map, cmap=cmap, alpha=alpha)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.suptitle(title)
    plt.tight_layout()

    # Save if requested
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=150)

    return fig


def visualize_layer_activations(
    model: nn.Module,
    image: torch.Tensor,
    layer_name: str,
    num_filters: int = 16,
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = "viridis",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize activations of a specific layer in a model.

    Args:
        model: PyTorch model
        image: Input image tensor (should be preprocessed)
        layer_name: Name of the layer to visualize
        num_filters: Number of filters to show
        figsize: Figure size
        cmap: Colormap to use
        output_path: Optional path to save the figure

    Returns:
        Matplotlib figure
    """
    # Ensure model is in eval mode
    model.eval()

    # Forward hooks to capture activations
    activations = {}

    def get_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()

        return hook

    # Find the layer
    target_layer = None
    for name, module in model.named_modules():
        if layer_name in name:
            target_layer = module
            break

    if target_layer is None:
        logger.error(f"Layer {layer_name} not found in model")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Layer {layer_name} not found", ha="center", va="center")
        return fig

    # Register hook
    handle = target_layer.register_forward_hook(get_activation(layer_name))

    # Forward pass
    with torch.no_grad():
        _ = model(image.unsqueeze(0))  # Add batch dimension

    # Remove hook
    handle.remove()

    # Get activations
    if layer_name not in activations:
        logger.error(f"No activations captured for layer {layer_name}")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"No activations for {layer_name}", ha="center", va="center")
        return fig

    act = activations[layer_name][0]  # Remove batch dimension

    # Plot activations
    num_channels = min(num_filters, act.size(0))
    ncols = min(4, num_channels)
    nrows = (num_channels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows * ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(num_channels):
        if i >= len(axes):
            break
        axes[i].imshow(act[i].cpu().numpy(), cmap=cmap)
        axes[i].set_title(f"Filter {i}")
        axes[i].axis("off")

    # Hide unused axes
    for i in range(num_channels, len(axes)):
        axes[i].axis("off")

    plt.suptitle(f"Activations for {layer_name}")
    plt.tight_layout()

    # Save if requested
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=150)

    return fig


def plot_attention_comparison(
    images: List[torch.Tensor],
    channel_maps: List[torch.Tensor],
    spatial_maps: List[torch.Tensor],
    titles: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 12),
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a comparison of channel and spatial attention maps for multiple images.

    Args:
        images: List of input images
        channel_maps: List of channel attention maps
        spatial_maps: List of spatial attention maps
        titles: Optional list of titles for each image
        figsize: Figure size
        output_path: Optional path to save the figure

    Returns:
        Matplotlib figure
    """
    num_images = len(images)
    if titles is None:
        titles = [f"Image {i+1}" for i in range(num_images)]

    fig, axes = plt.subplots(num_images, 3, figsize=figsize)

    # Handle single image case
    if num_images == 1:
        axes = np.array([axes])

    for i in range(num_images):
        # Convert image to numpy
        img = _to_numpy(images[i])
        if img.shape[0] == 3:  # CHW format
            img = np.transpose(img, (1, 2, 0))
        # Normalize if needed
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)

        # Plot original image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(titles[i])
        axes[i, 0].axis("off")

        # Plot channel attention
        ch_map = _to_numpy(channel_maps[i])
        if ch_map.ndim > 2:
            ch_map = ch_map.squeeze()
        # For channel attention, display as a bar chart if it's a 1D array
        if ch_map.ndim == 1 or (ch_map.ndim == 2 and ch_map.shape[0] == 1):
            if ch_map.ndim == 2:
                ch_map = ch_map.squeeze(0)
            axes[i, 1].bar(range(len(ch_map)), ch_map)
            axes[i, 1].set_title("Channel Attention")
        else:
            # If it's 2D, display as image
            axes[i, 1].imshow(ch_map, cmap="viridis")
            axes[i, 1].set_title("Channel Attention")
            axes[i, 1].axis("off")

        # Plot spatial attention
        sp_map = _to_numpy(spatial_maps[i])
        if sp_map.ndim > 2:
            sp_map = sp_map.squeeze()

        # Resize spatial map if needed
        if sp_map.shape != img.shape[:2]:
            sp_pil = Image.fromarray((sp_map * 255).astype(np.uint8))
            sp_pil = sp_pil.resize((img.shape[1], img.shape[0]), Image.BICUBIC)
            sp_map = np.array(sp_pil) / 255.0

        # Overlay spatial attention on image
        axes[i, 2].imshow(img)
        im = axes[i, 2].imshow(sp_map, cmap="jet", alpha=0.7)
        axes[i, 2].set_title("Spatial Attention")
        axes[i, 2].axis("off")

    plt.tight_layout()

    # Save if requested
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=150)

    return fig


def generate_attention_report(
    model: nn.Module,
    image: torch.Tensor,
    output_dir: str,
    layer_names: Optional[List[str]] = None,
    title_prefix: str = "CBAM Attention",
    resize_image: bool = True,
) -> str:
    """
    Generate a comprehensive attention visualization report.

    Args:
        model: PyTorch model with CBAM attention
        image: Input image tensor (should be preprocessed)
        output_dir: Directory to save the report
        layer_names: Optional list of layer names to visualize
        title_prefix: Prefix for the report title
        resize_image: Whether to resize the input image to match model input size

    Returns:
        Path to the report HTML file
    """
    from datetime import datetime

    import torch.nn.functional as F

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure model is in eval mode
    model.eval()

    # Resize image if needed
    if resize_image and hasattr(model, "input_size"):
        input_size = model.input_size
        if isinstance(input_size, (list, tuple)) and len(input_size) == 2:
            # Check if image needs resizing
            if image.shape[-2:] != tuple(input_size):
                logger.info(f"Resizing image from {image.shape[-2:]} to {input_size}")
                image = F.interpolate(
                    image.unsqueeze(0),
                    size=input_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

    # Get attention maps
    with torch.no_grad():
        # Check if model has a get_attention_maps method
        if hasattr(model, "get_attention_maps"):
            _ = model(image.unsqueeze(0))  # Forward pass
            attention_maps = model.get_attention_maps()
        # Try backbone if available
        elif hasattr(model, "backbone") and hasattr(
            model.backbone, "get_attention_maps"
        ):
            _ = model(image.unsqueeze(0))  # Forward pass
            attention_maps = model.backbone.get_attention_maps()
        else:
            logger.error("Model does not support attention map extraction")
            return ""

    # Separate channel and spatial attention maps
    channel_maps = {k: v for k, v in attention_maps.items() if "channel" in k}
    spatial_maps = {k: v for k, v in attention_maps.items() if "spatial" in k}

    # Extract attention maps for different layers
    if layer_names is None:
        # Extract unique layer names from keys
        layer_names = set()
        for key in attention_maps.keys():
            parts = key.split("_")
            if len(parts) >= 2:
                layer_names.add(parts[0])
        layer_names = sorted(list(layer_names))

    # Create HTML report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"attention_report_{timestamp}.html"

    with open(report_file, "w") as f:
        f.write(f"<html>\n<head>\n<title>{title_prefix} Report</title>\n")
        f.write("<style>\n")
        f.write(
            "body { font-family: Arial, sans-serif; margin: 0 auto; max-width: 1200px; padding: 20px; }\n"
        )
        f.write("h1, h2, h3 { color: #333; }\n")
        f.write(".figure { margin: 20px 0; text-align: center; }\n")
        f.write(
            ".figure img { max-width: 100%; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }\n"
        )
        f.write(
            ".section { margin: 40px 0; border-top: 1px solid #eee; padding-top: 20px; }\n"
        )
        f.write("</style>\n</head>\n<body>\n")

        # Header
        f.write(f"<h1>{title_prefix} Report</h1>\n")
        f.write(f"<p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n")

        # Original image
        f.write("<div class='section'>\n")
        f.write("<h2>Input Image</h2>\n")

        # Save and display original image
        img_path = output_dir / "input_image.png"
        plt.figure(figsize=(8, 8))
        img_np = _to_numpy(image)
        if img_np.shape[0] == 3:  # CHW format
            img_np = np.transpose(img_np, (1, 2, 0))
        # Normalize if needed
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        plt.imshow(img_np)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(img_path, bbox_inches="tight", dpi=150)
        plt.close()

        f.write("<div class='figure'>\n")
        f.write(f"<img src='{os.path.relpath(img_path, output_dir.parent)}'>\n")
        f.write("</div>\n")
        f.write("</div>\n")

        # Overview of all attention maps
        f.write("<div class='section'>\n")
        f.write("<h2>Overview of Attention Maps</h2>\n")

        # Channel attention overview
        if channel_maps:
            ch_path = output_dir / "channel_attention_overview.png"
            visualize_attention_maps(
                channel_maps,
                figsize=(15, 10),
                output_path=ch_path,
                suptitle="Channel Attention Maps Overview",
            )

            f.write("<h3>Channel Attention</h3>\n")
            f.write("<div class='figure'>\n")
            f.write(f"<img src='{os.path.relpath(ch_path, output_dir.parent)}'>\n")
            f.write("</div>\n")

        # Spatial attention overview
        if spatial_maps:
            sp_path = output_dir / "spatial_attention_overview.png"
            visualize_attention_maps(
                spatial_maps,
                figsize=(15, 10),
                output_path=sp_path,
                suptitle="Spatial Attention Maps Overview",
            )

            f.write("<h3>Spatial Attention</h3>\n")
            f.write("<div class='figure'>\n")
            f.write(f"<img src='{os.path.relpath(sp_path, output_dir.parent)}'>\n")
            f.write("</div>\n")

        f.write("</div>\n")

        # Detailed attention analysis by layer
        f.write("<div class='section'>\n")
        f.write("<h2>Layer-by-Layer Attention Analysis</h2>\n")

        for layer in layer_names:
            f.write(f"<h3>Layer: {layer}</h3>\n")

            # Get channel and spatial attention for this layer
            layer_ch_maps = {
                k: v for k, v in channel_maps.items() if k.startswith(layer)
            }
            layer_sp_maps = {
                k: v for k, v in spatial_maps.items() if k.startswith(layer)
            }

            # Need at least one map for this layer
            if not layer_ch_maps and not layer_sp_maps:
                continue

            # Create separate figures
            for block_idx in range(10):  # Assuming maximum 10 blocks per layer
                ch_key = f"{layer}_{block_idx}_channel"
                sp_key = f"{layer}_{block_idx}_spatial"

                if ch_key in layer_ch_maps and sp_key in layer_sp_maps:
                    # Create overlay visualization
                    overlay_path = output_dir / f"{layer}_block{block_idx}_overlay.png"
                    visualize_attention_overlay(
                        image=img_np,
                        attention_map=layer_sp_maps[sp_key][0],  # First item in batch
                        output_path=overlay_path,
                        title=f"{layer} Block {block_idx} Attention",
                    )

                    f.write(f"<h4>Block {block_idx}</h4>\n")
                    f.write("<div class='figure'>\n")
                    f.write(
                        f"<img src='{os.path.relpath(overlay_path, output_dir.parent)}'>\n"
                    )
                    f.write("</div>\n")

        f.write("</div>\n")

        # Comparison across layers
        if len(layer_names) > 1:
            f.write("<div class='section'>\n")
            f.write("<h2>Comparison Across Layers</h2>\n")

            # Create comparison visualization using last block of each layer
            layers_to_compare = []
            ch_maps_to_compare = []
            sp_maps_to_compare = []

            for layer in layer_names:
                # Find the last block for this layer
                block_idx = -1
                for key in attention_maps.keys():
                    if key.startswith(layer):
                        parts = key.split("_")
                        if len(parts) >= 2:
                            try:
                                idx = int(parts[1])
                                block_idx = max(block_idx, idx)
                            except ValueError:
                                pass

                if block_idx >= 0:
                    ch_key = f"{layer}_{block_idx}_channel"
                    sp_key = f"{layer}_{block_idx}_spatial"

                    if ch_key in attention_maps and sp_key in attention_maps:
                        layers_to_compare.append(layer)
                        ch_maps_to_compare.append(attention_maps[ch_key][0])
                        sp_maps_to_compare.append(attention_maps[sp_key][0])

            if layers_to_compare:
                comparison_path = output_dir / "layer_comparison.png"
                # Create copies of the input image for each layer
                images = [image.clone() for _ in range(len(layers_to_compare))]

                plot_attention_comparison(
                    images=images,
                    channel_maps=ch_maps_to_compare,
                    spatial_maps=sp_maps_to_compare,
                    titles=layers_to_compare,
                    figsize=(15, 10),
                    output_path=comparison_path,
                )

                f.write("<div class='figure'>\n")
                f.write(
                    f"<img src='{os.path.relpath(comparison_path, output_dir.parent)}'>\n"
                )
                f.write("</div>\n")

            f.write("</div>\n")

        # Footer
        f.write("</body>\n</html>")

    logger.info(f"Attention report generated at: {report_file}")
    return str(report_file)
