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
from utils.paths import ensure_dir

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
    cmap: str = "YlOrRd",
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
    cmap: str = "YlOrRd",
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
        plt.savefig(output_path, bbox_inches="tight", dpi=400)

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
        plt.savefig(output_path, bbox_inches="tight", dpi=400)

    return fig


def visualize_layer_activations(
    model: nn.Module,
    image: torch.Tensor,
    layer_name: str,
    num_filters: int = 16,
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = "YlOrRd",
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
        plt.savefig(output_path, bbox_inches="tight", dpi=400)

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
        titles = [f"Image {i + 1}" for i in range(num_images)]

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
            axes[i, 1].imshow(ch_map, cmap="YlOrRd")
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
        plt.savefig(output_path, bbox_inches="tight", dpi=400)

    return fig


def generate_attention_report(
    model: nn.Module,
    image: torch.Tensor,
    output_dir: str,
    layer_names: Optional[List[str]] = None,
    title_prefix: str = "CBAM Attention",
    resize_image: bool = True,
    filename_prefix: Optional[str] = None,
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
        filename_prefix: Optional prefix for generated filenames

    Returns:
        Path to the report HTML file
    """
    # Ensure output directory exists
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    # Extract model name from the model instance
    model_name = model.__class__.__name__

    # Apply dark theme for visualizations
    from core.visualization.base_visualization import DEFAULT_THEME, apply_dark_theme

    theme = DEFAULT_THEME.copy()
    apply_dark_theme(theme)

    # Create an attention visualizer
    visualizer = CBAMVisualizer(model)

    # Normalize and preprocess the input image if needed
    if resize_image:
        image = visualizer.preprocess_image(image, normalize=True)

    # Get the file prefix for saving visualizations
    prefix = filename_prefix or "attention"

    # Get attention maps
    attention_maps = visualizer.get_attention_maps(image)

    # Generate visualizations of attention maps (spatial and channel)
    spatial_maps = {
        name: attention_map
        for name, attention_map in attention_maps.items()
        if "_spatial" in name
    }
    channel_maps = {
        name: attention_map
        for name, attention_map in attention_maps.items()
        if "_channel" in name
    }

    # Create output paths
    spatial_viz_path = output_dir / f"{prefix}_spatial.png"
    channel_viz_path = output_dir / f"{prefix}_channel.png"
    overlay_viz_path = output_dir / f"{prefix}_overlay.png"

    # Filter attention maps by layer_names if provided
    if layer_names:
        spatial_maps = {
            k: v
            for k, v in spatial_maps.items()
            if any(name in k for name in layer_names)
        }
        channel_maps = {
            k: v
            for k, v in channel_maps.items()
            if any(name in k for name in layer_names)
        }

    # Generate visualization of spatial attention maps
    if spatial_maps:
        # Convert tensor to numpy for visualization
        original_image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        # Denormalize image
        original_image = np.clip(original_image * 0.225 + 0.45, 0, 1)

        # Set theme and styling
        plt.style.use("dark_background")

        # Determine grid size
        n = len(spatial_maps)
        nrows = 3  # Fixed number of rows for more consistent layout
        ncols = 3  # Fixed number of columns

        # Create figure with dark theme
        fig = plt.figure(figsize=(15, 15), facecolor=theme["background_color"])

        # Main title
        fig.suptitle(
            "Attention Overlays",
            fontsize=22,
            fontweight="bold",
            color=theme["text_color"],
        )
        # Subtitle
        plt.figtext(
            0.5,
            0.92,
            "Areas the model focuses on for classification",
            ha="center",
            fontsize=14,
            color=theme["text_color"],
        )

        # Generate the subplot layout
        grid = plt.GridSpec(nrows, ncols, figure=fig, wspace=0.2, hspace=0.4)

        # Original image - make it larger and centered in the first row
        ax_original = fig.add_subplot(grid[0, 0])
        ax_original.set_facecolor(theme["background_color"])
        ax_original.imshow(original_image)
        ax_original.set_title(
            "Original Image",
            fontsize=16,
            pad=10,
            fontweight="medium",
            color=theme["text_color"],
        )
        ax_original.axis("off")

        # Add borders to axes
        for spine in ax_original.spines.values():
            spine.set_visible(True)
            spine.set_color(theme["grid_color"])
            spine.set_linewidth(1)

        # Add attention overlays
        for i, (name, attention_map) in enumerate(spatial_maps.items()):
            # Skip if we've filled the grid
            if i >= nrows * ncols - 1:  # -1 for the original image
                break

            # Calculate position - fill first row then continue to next rows
            row = (i + 1) // ncols
            col = (i + 1) % ncols

            # Create subplot
            ax = fig.add_subplot(grid[row, col])
            ax.set_facecolor(theme["background_color"])

            try:
                # Normalize attention map
                attn_map = attention_map.squeeze().cpu().numpy()
                attn_map = (attn_map - attn_map.min()) / (
                    attn_map.max() - attn_map.min() + 1e-8
                )

                # Create heatmap
                heatmap = plt.cm.get_cmap("hot")(attn_map)
                heatmap = heatmap[:, :, :3]  # Remove alpha channel

                # Create alpha blend overlay
                alpha = 0.7  # Slightly higher alpha for better visibility in dark theme
                overlay = original_image * (1 - alpha) + heatmap * alpha
                overlay = np.clip(overlay, 0, 1)

                # Show overlay
                ax.imshow(overlay)

                # Format the name for better readability
                formatted_name = name.replace("backbone.backbone.", "")
                formatted_name = formatted_name.replace("_spatial", "")
                formatted_name = formatted_name.replace("_", " ")

                ax.set_title(formatted_name, fontsize=12, color=theme["text_color"])
                ax.axis("off")

                # Add subtle borders to distinguish plots
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color(theme["grid_color"])
                    spine.set_linewidth(0.5)

            except Exception as e:
                logger.error(f"Error visualizing attention map {name}: {e}")
                ax.text(
                    0.5,
                    0.5,
                    "Visualization Error",
                    ha="center",
                    va="center",
                    color="red",
                )
                ax.axis("off")

        # Save the figure
        plt.savefig(
            spatial_viz_path,
            dpi=400,
            bbox_inches="tight",
            facecolor=theme["background_color"],
        )
        logger.info(f"Saved spatial attention visualization to {spatial_viz_path}")

    # Generate visualization of channel attention maps
    if channel_maps:
        plt.figure(figsize=(12, 8), facecolor=theme["background_color"])
        plt.style.use("dark_background")

        for name, attention_map in channel_maps.items():
            # Process channel attention (typically shape [C, 1, 1])
            channel_weights = attention_map.squeeze().cpu().numpy()

            # Sort channels by attention weight (descending)
            sorted_indices = np.argsort(channel_weights)[::-1]
            sorted_weights = channel_weights[sorted_indices]

            # Create bar chart with dark theme
            fig, ax = plt.subplots(figsize=(12, 8), facecolor=theme["background_color"])
            ax.set_facecolor(theme["background_color"])

            # Format name for better readability
            formatted_name = name.replace("backbone.backbone.", "")
            formatted_name = formatted_name.replace("_channel", "")
            formatted_name = formatted_name.replace("_", " ")

            # Create color gradient for bars
            cmap = plt.cm.get_cmap("plasma")
            colors = cmap(np.linspace(0, 1, len(sorted_weights)))

            # Create bars with enhanced styling
            bars = ax.bar(
                range(len(sorted_weights)),
                sorted_weights,
                color=colors,
                width=0.8,
                edgecolor=theme["grid_color"],
                linewidth=0.5,
            )

            # Add subtle grid and styling
            ax.grid(
                True, axis="y", linestyle="--", alpha=0.3, color=theme["grid_color"]
            )
            ax.set_facecolor(theme["background_color"])
            ax.set_xlabel(
                "Channel Index (sorted by weight)",
                color=theme["text_color"],
                fontsize=12,
            )
            ax.set_ylabel("Attention Weight", color=theme["text_color"], fontsize=12)
            ax.set_title(
                f"Channel Attention Weights: {formatted_name}",
                color=theme["text_color"],
                fontsize=14,
                pad=20,
            )

            # Set tick colors
            ax.tick_params(axis="x", colors=theme["text_color"])
            ax.tick_params(axis="y", colors=theme["text_color"])

            # Add value annotations for the top channels
            for i, bar in enumerate(bars[:10]):  # Annotate just the top 10
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.002,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    color=theme["text_color"],
                    fontsize=8,
                )

            # Highlight the most important channels
            for i, bar in enumerate(bars[:5]):  # Highlight top 5 channels
                bar.set_edgecolor(theme["main_color"])
                bar.set_linewidth(1.5)

            # Add subtle border to the entire plot
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(theme["grid_color"])
                spine.set_linewidth(0.5)

            # Save the figure for this channel attention map
            plt.tight_layout()
            plt.savefig(
                channel_viz_path,
                dpi=400,
                bbox_inches="tight",
                facecolor=theme["background_color"],
            )
            logger.info(f"Saved channel attention visualization to {channel_viz_path}")
            plt.close()

            # We only process the first channel attention map
            break

    # Create HTML report
    from datetime import datetime

    # Create HTML report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use filename_prefix if provided
    if filename_prefix:
        report_file = output_dir / f"{filename_prefix}_attention_report.html"
    else:
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
        image_name = "input_image.png"
        if filename_prefix:
            image_name = f"{filename_prefix}_input_image.png"
        img_path = output_dir / image_name

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
        plt.savefig(img_path, bbox_inches="tight", dpi=400)
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
            ch_name = "channel_attention_overview.png"
            if filename_prefix:
                ch_name = f"{filename_prefix}_channel_attention_overview.png"
            ch_path = output_dir / ch_name

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
            sp_name = "spatial_attention_overview.png"
            if filename_prefix:
                sp_name = f"{filename_prefix}_spatial_attention_overview.png"
            sp_path = output_dir / sp_name

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
                    overlay_name = f"{layer}_block{block_idx}_overlay.png"
                    if filename_prefix:
                        overlay_name = (
                            f"{filename_prefix}_{layer}_block{block_idx}_overlay.png"
                        )
                    overlay_path = output_dir / overlay_name

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
                comparison_name = "layer_comparison.png"
                if filename_prefix:
                    comparison_name = f"{filename_prefix}_layer_comparison.png"
                comparison_path = output_dir / comparison_name

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


class CBAMVisualizer:
    """
    Visualizer for CBAM attention maps.
    """

    def __init__(self, model: nn.Module):
        """
        Initialize the CBAM attention visualizer.

        Args:
            model: PyTorch model with CBAM layers
        """
        self.model = model
        self.device = next(model.parameters()).device
        self.model.eval()

        # Set up normalization parameters
        self.mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    def preprocess_image(
        self, image: torch.Tensor, normalize: bool = True
    ) -> torch.Tensor:
        """
        Preprocess an image for model input.

        Args:
            image: Input image tensor (C, H, W)
            normalize: Whether to normalize using ImageNet stats

        Returns:
            Preprocessed image tensor
        """
        # Ensure image is on the correct device
        image = image.to(self.device)

        # Add batch dimension if missing
        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Resize if model has input_size attribute
        if hasattr(self.model, "input_size"):
            try:
                input_size = self.model.input_size
                if isinstance(input_size, (list, tuple)) and len(input_size) == 2:
                    if image.shape[2:] != tuple(input_size):
                        logger.info(
                            f"Resizing image from {image.shape[2:]} to {input_size}"
                        )
                        import torch.nn.functional as F

                        image = F.interpolate(
                            image, size=input_size, mode="bilinear", align_corners=False
                        )
            except Exception as e:
                logger.warning(f"Error resizing image: {e}")

        # Normalize if requested
        if normalize:
            if image.max() > 1.0:
                image = image / 255.0

            mean = self.mean.to(image.device)
            std = self.std.to(image.device)

            image = (image - mean) / std

        return image

    def get_attention_maps(
        self, image_input: Union[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Get attention maps from the model for the given input image.

        Args:
            image_input: Input image (path or tensor)

        Returns:
            Dictionary of attention maps by layer name
        """
        # Convert file path to tensor if needed
        if isinstance(image_input, str):
            from PIL import Image
            from torchvision import transforms

            img = Image.open(image_input).convert("RGB")
            transform = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.ToTensor(),
                ]
            )
            image = transform(img).unsqueeze(0)
        else:
            image = image_input

        # Ensure image is preprocessed
        image = self.preprocess_image(image)

        # Get attention maps from the model
        with torch.no_grad():
            # Forward pass
            _ = self.model(image)

            # Check if model has a get_attention_maps method
            if hasattr(self.model, "get_attention_maps"):
                attention_maps = self.model.get_attention_maps(image)
            # Try backbone if available
            elif hasattr(self.model, "backbone") and hasattr(
                self.model.backbone, "get_attention_maps"
            ):
                attention_maps = self.model.backbone.get_attention_maps(image)
            else:
                logger.error("Model does not support attention map extraction")
                return {}

        return attention_maps
