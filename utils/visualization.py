"""
Visualization utilities for model analysis and reporting.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from PIL import Image
from sklearn.metrics import confusion_matrix

from utils.logger import get_logger
from utils.paths import ensure_dir

logger = get_logger(__name__)

# Default dark theme settings
DEFAULT_THEME = {
    "background_color": "#121212",
    "text_color": "#f5f5f5",
    "grid_color": "#404040",
    "main_color": "#34d399",
    "bar_colors": ["#a78bfa", "#22d3ee", "#34d399", "#d62728", "#e27c7c"],
    "cmap": "viridis",
}


def apply_dark_theme(theme: Optional[Dict] = None) -> None:
    """
    Apply dark theme to matplotlib plots.

    Args:
        theme: Theme settings dictionary (uses DEFAULT_THEME if None)
    """
    theme = theme or DEFAULT_THEME

    # Set the style
    plt.style.use("dark_background")

    # Configure plot settings
    plt.rcParams["figure.facecolor"] = theme["background_color"]
    plt.rcParams["axes.facecolor"] = theme["background_color"]
    plt.rcParams["text.color"] = theme["text_color"]
    plt.rcParams["axes.labelcolor"] = theme["text_color"]
    plt.rcParams["xtick.color"] = theme["text_color"]
    plt.rcParams["ytick.color"] = theme["text_color"]
    plt.rcParams["grid.color"] = theme["grid_color"]
    plt.rcParams["axes.edgecolor"] = theme["grid_color"]
    plt.rcParams["savefig.facecolor"] = theme["background_color"]
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", theme["bar_colors"])

    # Configure seaborn
    sns.set_palette(theme["bar_colors"])
    sns.set_style(
        "darkgrid",
        {
            "axes.facecolor": theme["background_color"],
            "grid.color": theme["grid_color"],
        },
    )
    sns.set_context("paper", font_scale=1.5)

    logger.info(f"Applied dark theme with {theme['background_color']} background")


def plot_training_history(
    history: Dict[str, List[float]],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 10),
    metrics: Optional[List[str]] = None,
    theme: Optional[Dict] = None,
) -> Figure:
    """
    Plot training and validation metrics history.

    Args:
        history: Dictionary with metrics history
        output_path: Path to save the figure
        figsize: Figure size
        metrics: List of metrics to plot (default: loss, accuracy)
        theme: Theme settings to apply (uses DEFAULT_THEME if None)

    Returns:
        Matplotlib figure
    """
    # Apply theme
    apply_dark_theme(theme)

    metrics = metrics or ["loss", "accuracy"]

    # Filter metrics to include only those present in history
    present_metrics = []
    for metric in metrics:
        train_key = metric
        val_key = f"val_{metric}"

        if train_key in history or val_key in history:
            present_metrics.append(metric)

    if not present_metrics:
        logger.warning("No metrics found in history to plot")
        return None

    # Create figure with subplots for each metric
    fig, axes = plt.subplots(len(present_metrics), 1, figsize=figsize, sharex=True)

    # If only one metric, ensure axes is a list
    if len(present_metrics) == 1:
        axes = [axes]

    theme = theme or DEFAULT_THEME
    main_color = theme["main_color"]
    bar_colors = theme["bar_colors"]

    for i, metric in enumerate(present_metrics):
        ax = axes[i]

        # Train metric
        train_key = metric
        if train_key in history:
            ax.plot(
                history[train_key],
                marker="o",
                markersize=4,
                linestyle="-",
                label=f"Train {metric.capitalize()}",
                color=bar_colors[0],
            )

        # Validation metric
        val_key = f"val_{metric}"
        if val_key in history:
            ax.plot(
                history[val_key],
                marker="s",
                markersize=4,
                linestyle="--",
                label=f"Val {metric.capitalize()}",
                color=bar_colors[1],
            )

        # Add title, labels, legend
        ax.set_title(f"{metric.capitalize()} vs. Epoch", color=theme["text_color"])
        ax.set_ylabel(metric.capitalize(), color=theme["text_color"])
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3, color=theme["grid_color"])
        ax.tick_params(colors=theme["text_color"])

    # Add epoch label to bottom subplot
    axes[-1].set_xlabel("Epoch", color=theme["text_color"])

    plt.tight_layout()

    # Save figure if output path provided
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            facecolor=theme["background_color"],
        )
        logger.info(f"Saved training history plot to {output_path}")

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 12),
    normalize: bool = True,
    cmap: Optional[str] = None,
    title: str = "Confusion Matrix",
    theme: Optional[Dict] = None,
) -> Figure:
    """
    Plot confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix
        class_names: List of class names
        output_path: Path to save the figure
        figsize: Figure size
        normalize: Whether to normalize by row
        cmap: Colormap (defaults to theme cmap if None)
        title: Plot title
        theme: Theme settings to apply (uses DEFAULT_THEME if None)

    Returns:
        Matplotlib figure
    """
    # Apply theme
    theme = theme or DEFAULT_THEME
    apply_dark_theme(theme)

    # Use the theme's colormap if none specified
    if cmap is None:
        cmap = theme["cmap"]

    # Create a normalized version of the confusion matrix
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.zeros_like(cm, dtype=float)
        np.divide(cm, row_sums, out=cm_norm, where=row_sums != 0)
    else:
        cm_norm = cm

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor=theme["background_color"])
    ax.set_facecolor(theme["background_color"])

    # Shorten class names if needed
    shortened_names = []
    for name in class_names:
        if len(name) > 20:
            shortened_names.append(name[:18] + "...")
        else:
            shortened_names.append(name)

    # Plot heatmap
    im = ax.imshow(cm_norm, cmap=cmap)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(
        "Proportion" if normalize else "Count",
        rotation=-90,
        va="bottom",
        color=theme["text_color"],
    )
    cbar.ax.tick_params(colors=theme["text_color"])

    # Set title and labels
    ax.set_title(title, color=theme["text_color"])
    ax.set_xlabel("Predicted Label", color=theme["text_color"])
    ax.set_ylabel("True Label", color=theme["text_color"])

    # Add ticks
    ax.set_xticks(np.arange(len(shortened_names)))
    ax.set_yticks(np.arange(len(shortened_names)))
    ax.set_xticklabels(
        shortened_names,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        color=theme["text_color"],
    )
    ax.set_yticklabels(shortened_names, color=theme["text_color"])

    # Add grid lines
    ax.set_xticks(np.arange(len(class_names) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(class_names) + 1) - 0.5, minor=True)
    ax.grid(
        which="minor",
        color=theme["grid_color"],
        linestyle="-",
        linewidth=0.5,
        alpha=0.2,
    )
    ax.tick_params(which="minor", bottom=False, left=False, colors=theme["text_color"])

    # Add text annotations (only for small matrices)
    if len(class_names) <= 30:
        fmt = ".2f" if normalize else "d"
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                value = cm_norm[i, j]
                threshold = cm_norm.max() / 2.0
                color = "white" if value > threshold else theme["text_color"]
                ax.text(j, i, format(value, fmt), ha="center", va="center", color=color)

    plt.tight_layout()

    # Save figure if output path provided
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            facecolor=theme["background_color"],
        )
        logger.info(f"Saved confusion matrix plot to {output_path}")

    return fig


def plot_class_metrics(
    metrics: Dict[str, float],
    class_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 10),
    metrics_to_plot: List[str] = ["accuracy", "precision", "recall", "f1"],
    sort_by: Optional[str] = "f1",
    theme: Optional[Dict] = None,
) -> Figure:
    """
    Plot class-wise metrics.

    Args:
        metrics: Dictionary of class metrics
        class_names: List of class names
        output_path: Path to save the figure
        figsize: Figure size
        metrics_to_plot: List of metrics to plot
        sort_by: Metric to sort by
        theme: Theme settings to apply (uses DEFAULT_THEME if None)

    Returns:
        Matplotlib figure
    """
    # Apply theme
    theme = theme or DEFAULT_THEME
    apply_dark_theme(theme)

    # Extract class metrics
    class_metrics = {}

    for class_idx, class_name in enumerate(class_names):
        class_metrics[class_name] = {}
        for metric in metrics_to_plot:
            metric_key = f"class_{class_name.replace(' ', '_')}_{metric}"
            if metric_key in metrics:
                class_metrics[class_name][metric] = metrics[metric_key]

    # Convert to DataFrame
    df = pd.DataFrame(class_metrics).T

    # Safety check: ensure we have at least one metric
    if df.empty:
        logger.warning("No class metrics found in metrics dictionary")
        # Create a blank figure as a placeholder
        fig, ax = plt.subplots(figsize=figsize, facecolor=theme["background_color"])
        ax.set_facecolor(theme["background_color"])
        ax.text(
            0.5,
            0.5,
            "No class metrics available",
            ha="center",
            va="center",
            fontsize=14,
            color=theme["text_color"],
        )
        ax.axis("off")

        if output_path:
            output_path = Path(output_path)
            ensure_dir(output_path.parent)
            plt.savefig(
                output_path,
                dpi=300,
                bbox_inches="tight",
                facecolor=theme["background_color"],
            )

        return fig

    # Sort if requested
    if sort_by and sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=False)
    elif sort_by:
        logger.warning(
            f"Requested sorting by '{sort_by}' but it's not available. "
            f"Available metrics: {list(df.columns)}"
        )
        # Try to sort by an alternative metric if available
        for alt_metric in ["f1", "precision", "recall", "accuracy"]:
            if alt_metric in df.columns:
                logger.info(f"Sorting by '{alt_metric}' instead")
                df = df.sort_values(by=alt_metric, ascending=False)
                break

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor=theme["background_color"])
    ax.set_facecolor(theme["background_color"])

    # Plot heatmap - use theme colors
    sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap=theme["cmap"],
        linewidths=0.5,
        cbar=True,
        ax=ax,
        annot_kws={"color": "white"},
        cbar_kws={"label": "Value", "shrink": 0.8},
    )

    # Set title and labels with theme colors
    ax.set_title("Performance Metrics by Class", color=theme["text_color"])
    ax.set_ylabel("Class", color=theme["text_color"])
    ax.set_xlabel("Metric", color=theme["text_color"])
    ax.tick_params(colors=theme["text_color"])

    plt.tight_layout()

    # Save figure if output path provided
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            facecolor=theme["background_color"],
        )
        logger.info(f"Saved class metrics plot to {output_path}")

    return fig


def plot_training_time(
    epoch_times: List[float],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 8),
    theme: Optional[Dict] = None,
) -> Figure:
    """
    Plot training time per epoch.

    Args:
        epoch_times: List of times per epoch in seconds
        output_path: Path to save the figure
        figsize: Figure size
        theme: Theme settings to apply (uses DEFAULT_THEME if None)

    Returns:
        Matplotlib figure
    """
    # Apply theme
    theme = theme or DEFAULT_THEME
    apply_dark_theme(theme)

    # Create figure with two subplots - with theme background
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, sharex=True, facecolor=theme["background_color"]
    )
    ax1.set_facecolor(theme["background_color"])
    ax2.set_facecolor(theme["background_color"])

    # Plot epoch times
    epochs = np.arange(1, len(epoch_times) + 1)
    ax1.plot(
        epochs,
        epoch_times,
        marker="o",
        linestyle="-",
        markersize=4,
        color=theme["bar_colors"][0],
    )
    ax1.set_ylabel("Time (seconds)", color=theme["text_color"])
    ax1.set_title("Training Time per Epoch", color=theme["text_color"])
    ax1.grid(True, alpha=0.3, color=theme["grid_color"])
    ax1.tick_params(colors=theme["text_color"])

    # Plot cumulative time
    cum_times = np.cumsum(epoch_times)
    hours = cum_times / 3600
    ax2.plot(
        epochs,
        hours,
        marker="s",
        linestyle="-",
        markersize=4,
        color=theme["bar_colors"][1],
    )
    ax2.set_ylabel("Cumulative Time (hours)", color=theme["text_color"])
    ax2.set_xlabel("Epoch", color=theme["text_color"])
    ax2.set_title("Cumulative Training Time", color=theme["text_color"])
    ax2.grid(True, alpha=0.3, color=theme["grid_color"])
    ax2.tick_params(colors=theme["text_color"])

    plt.tight_layout()

    # Save figure if output path provided
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            facecolor=theme["background_color"],
        )
        logger.info(f"Saved training time plot to {output_path}")

    return fig


def plot_model_comparison(
    metrics_list: List[Dict[str, float]],
    model_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 8),
    metrics_to_compare: List[str] = ["accuracy", "precision", "recall", "f1"],
    theme: Optional[Dict] = None,
) -> Figure:
    """
    Plot comparison of multiple models.

    Args:
        metrics_list: List of metrics dictionaries for each model
        model_names: List of model names
        output_path: Path to save the figure
        figsize: Figure size
        metrics_to_compare: List of metrics to compare
        theme: Theme settings to apply (uses DEFAULT_THEME if None)

    Returns:
        Matplotlib figure
    """
    # Apply theme
    theme = theme or DEFAULT_THEME
    apply_dark_theme(theme)

    if len(metrics_list) != len(model_names):
        logger.error("Number of metrics dictionaries must match number of model names")
        return None

    # Extract metrics for each model
    comparison_data = []
    for model_idx, metrics in enumerate(metrics_list):
        model_data = {"Model": model_names[model_idx]}
        for metric in metrics_to_compare:
            if metric in metrics:
                model_data[metric] = metrics[metric]
            else:
                model_data[metric] = None
        comparison_data.append(model_data)

    # Convert to DataFrame
    df = pd.DataFrame(comparison_data)

    # Set Model as index
    df.set_index("Model", inplace=True)

    # Create figure with theme background
    fig, ax = plt.subplots(figsize=figsize, facecolor=theme["background_color"])
    ax.set_facecolor(theme["background_color"])

    # Plot bar chart with theme colors
    df.plot(kind="bar", ax=ax, color=theme["bar_colors"])

    # Set title and labels with theme colors
    ax.set_title("Model Comparison", color=theme["text_color"])
    ax.set_ylabel("Score", color=theme["text_color"])
    ax.set_ylim([0, 1])
    ax.tick_params(colors=theme["text_color"])

    # Add value annotations with color that stands out on dark background
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, color=theme["text_color"])

    # Add legend with theme colors
    legend = ax.legend(title="Metric")
    legend.get_title().set_color(theme["text_color"])
    for text in legend.get_texts():
        text.set_color(theme["text_color"])

    plt.tight_layout()

    # Save figure if output path provided
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            facecolor=theme["background_color"],
        )
        logger.info(f"Saved model comparison plot to {output_path}")

    return fig


def plot_learning_rate(
    lr_history: List[float],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Learning Rate Schedule",
    theme: Optional[Dict] = None,
) -> Figure:
    """
    Plot learning rate schedule.

    Args:
        lr_history: List of learning rates
        output_path: Path to save the figure
        figsize: Figure size
        title: Plot title
        theme: Theme settings to apply (uses DEFAULT_THEME if None)

    Returns:
        Matplotlib figure
    """
    # Apply theme
    theme = theme or DEFAULT_THEME
    apply_dark_theme(theme)

    # Create figure with theme background
    fig, ax = plt.subplots(figsize=figsize, facecolor=theme["background_color"])
    ax.set_facecolor(theme["background_color"])

    # Plot learning rate with theme color
    epochs = np.arange(1, len(lr_history) + 1)
    ax.plot(
        epochs,
        lr_history,
        marker="o",
        linestyle="-",
        markersize=4,
        color=theme["main_color"],
    )

    # Set log scale for y-axis
    ax.set_yscale("log")

    # Set title and labels with theme colors
    ax.set_title(title, color=theme["text_color"])
    ax.set_xlabel("Epoch", color=theme["text_color"])
    ax.set_ylabel("Learning Rate", color=theme["text_color"])
    ax.grid(True, alpha=0.3, color=theme["grid_color"])
    ax.tick_params(colors=theme["text_color"])

    plt.tight_layout()

    # Save figure if output path provided
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            facecolor=theme["background_color"],
        )
        logger.info(f"Saved learning rate plot to {output_path}")

    return fig


def get_model_size(model: torch.nn.Module) -> float:
    """
    Calculate model size in MB.

    Args:
        model: PyTorch model

    Returns:
        Model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    # Convert to MB
    size_mb = (param_size + buffer_size) / (1024**2)
    return size_mb


def count_model_parameters(model: torch.nn.Module) -> int:
    """
    Count total parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
