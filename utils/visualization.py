"""
Visualization utilities for model analysis and reporting.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix

from utils.logger import get_logger
from utils.paths import ensure_dir

logger = get_logger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)


def plot_training_history(
    history: Dict[str, List[float]],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 10),
    metrics: Optional[List[str]] = None,
) -> Figure:
    """
    Plot training and validation metrics history.
    
    Args:
        history: Dictionary with metrics history
        output_path: Path to save the figure
        figsize: Figure size
        metrics: List of metrics to plot (default: loss, accuracy)
        
    Returns:
        Matplotlib figure
    """
    metrics = metrics or ['loss', 'accuracy']
    
    # Filter metrics to include only those present in history
    present_metrics = []
    for metric in metrics:
        train_key = metric
        val_key = f'val_{metric}'
        
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
    
    for i, metric in enumerate(present_metrics):
        ax = axes[i]
        
        # Train metric
        train_key = metric
        if train_key in history:
            ax.plot(history[train_key], marker='o', markersize=4, linestyle='-', 
                   label=f'Train {metric.capitalize()}')
        
        # Validation metric
        val_key = f'val_{metric}'
        if val_key in history:
            ax.plot(history[val_key], marker='s', markersize=4, linestyle='--', 
                   label=f'Val {metric.capitalize()}')
        
        # Add title, labels, legend
        ax.set_title(f'{metric.capitalize()} vs. Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    # Add epoch label to bottom subplot
    axes[-1].set_xlabel('Epoch')
    
    plt.tight_layout()
    
    # Save figure if output path provided
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training history plot to {output_path}")
    
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 12),
    normalize: bool = True,
    cmap: str = "Blues",
    title: str = "Confusion Matrix",
) -> Figure:
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        output_path: Path to save the figure
        figsize: Figure size
        normalize: Whether to normalize by row
        cmap: Colormap
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Create a normalized version of the confusion matrix
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.zeros_like(cm, dtype=float)
        np.divide(cm, row_sums, out=cm_norm, where=row_sums != 0)
    else:
        cm_norm = cm
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Shorten class names if needed
    shortened_names = []
    for name in class_names:
        if len(name) > 20:
            shortened_names.append(name[:18] + '...')
        else:
            shortened_names.append(name)
    
    # Plot heatmap
    im = ax.imshow(cm_norm, cmap=cmap)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Proportion" if normalize else "Count", rotation=-90, va="bottom")
    
    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    # Add ticks
    ax.set_xticks(np.arange(len(shortened_names)))
    ax.set_yticks(np.arange(len(shortened_names)))
    ax.set_xticklabels(shortened_names, rotation=45, ha='right', rotation_mode='anchor')
    ax.set_yticklabels(shortened_names)
    
    # Add grid lines
    ax.set_xticks(np.arange(len(class_names) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(class_names) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    # Add text annotations (only for small matrices)
    if len(class_names) <= 30:
        fmt = '.2f' if normalize else 'd'
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                value = cm_norm[i, j]
                threshold = cm_norm.max() / 2.
                color = "white" if value > threshold else "black"
                ax.text(j, i, format(value, fmt),
                        ha="center", va="center",
                        color=color)
    
    plt.tight_layout()
    
    # Save figure if output path provided
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix plot to {output_path}")
    
    return fig


def plot_class_metrics(
    metrics: Dict[str, float],
    class_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 10),
    metrics_to_plot: List[str] = ['accuracy', 'precision', 'recall', 'f1'],
    sort_by: Optional[str] = 'f1',
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
        
    Returns:
        Matplotlib figure
    """
    # Extract class metrics
    class_metrics = {}
    
    for class_idx, class_name in enumerate(class_names):
        class_metrics[class_name] = {}
        for metric in metrics_to_plot:
            metric_key = f"class_{class_name.replace(' ', '_')}_{metric}"
            if metric_key in metrics:
                class_metrics[class_name][metric] = metrics[metric_key]
    
    # Convert to DataFrame
    import pandas as pd
    df = pd.DataFrame(class_metrics).T
    
    # Sort if requested
    if sort_by and sort_by in metrics_to_plot:
        df = df.sort_values(by=sort_by, ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        df, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues',
        linewidths=0.5,
        cbar=True,
        ax=ax
    )
    
    # Set title and labels
    ax.set_title('Performance Metrics by Class')
    ax.set_ylabel('Class')
    ax.set_xlabel('Metric')
    
    plt.tight_layout()
    
    # Save figure if output path provided
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved class metrics plot to {output_path}")
    
    return fig


def plot_training_time(
    epoch_times: List[float],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> Figure:
    """
    Plot training time per epoch.
    
    Args:
        epoch_times: List of times per epoch in seconds
        output_path: Path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Plot epoch times
    epochs = np.arange(1, len(epoch_times) + 1)
    ax1.plot(epochs, epoch_times, marker='o', linestyle='-', markersize=4)
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Training Time per Epoch')
    ax1.grid(True, alpha=0.3)
    
    # Plot cumulative time
    cum_times = np.cumsum(epoch_times)
    hours = cum_times / 3600
    ax2.plot(epochs, hours, marker='s', linestyle='-', markersize=4)
    ax2.set_ylabel('Cumulative Time (hours)')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Cumulative Training Time')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if output path provided
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training time plot to {output_path}")
    
    return fig


def plot_model_comparison(
    metrics_list: List[Dict[str, float]],
    model_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 8),
    metrics_to_compare: List[str] = ['accuracy', 'precision', 'recall', 'f1'],
) -> Figure:
    """
    Plot comparison of multiple models.
    
    Args:
        metrics_list: List of metrics dictionaries for each model
        model_names: List of model names
        output_path: Path to save the figure
        figsize: Figure size
        metrics_to_compare: List of metrics to compare
        
    Returns:
        Matplotlib figure
    """
    if len(metrics_list) != len(model_names):
        logger.error("Number of metrics dictionaries must match number of model names")
        return None
    
    # Extract metrics for each model
    comparison_data = []
    for model_idx, metrics in enumerate(metrics_list):
        model_data = {'Model': model_names[model_idx]}
        for metric in metrics_to_compare:
            if metric in metrics:
                model_data[metric] = metrics[metric]
            else:
                model_data[metric] = None
        comparison_data.append(model_data)
    
    # Convert to DataFrame
    import pandas as pd
    df = pd.DataFrame(comparison_data)
    
    # Set Model as index
    df.set_index('Model', inplace=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot bar chart
    df.plot(kind='bar', ax=ax)
    
    # Set title and labels
    ax.set_title('Model Comparison')
    ax.set_ylabel('Score')
    ax.set_ylim([0, 1])
    
    # Add value annotations
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)
    
    # Add legend
    ax.legend(title='Metric')
    
    plt.tight_layout()
    
    # Save figure if output path provided
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved model comparison plot to {output_path}")
    
    return fig


def plot_learning_rate(
    lr_history: List[float],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Learning Rate Schedule",
) -> Figure:
    """
    Plot learning rate schedule.
    
    Args:
        lr_history: List of learning rates
        output_path: Path to save the figure
        figsize: Figure size
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot learning rate
    epochs = np.arange(1, len(lr_history) + 1)
    ax.plot(epochs, lr_history, marker='o', linestyle='-', markersize=4)
    
    # Set log scale for y-axis
    ax.set_yscale('log')
    
    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if output path provided
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
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
    size_mb = (param_size + buffer_size) / (1024 ** 2)
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