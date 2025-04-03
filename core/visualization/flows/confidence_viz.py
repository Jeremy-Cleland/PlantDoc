"""
Confidence visualization module.

This module provides tools to visualize model confidence metrics across classes
and training epochs, helping identify classes that the model struggles with.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from utils.logger import get_logger
from utils.paths import ensure_dir

logger = get_logger(__name__)


def plot_confidence_distribution_by_class(
    class_confidences: Dict[int, float],
    class_counts: Dict[int, int],
    class_names: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Mean Confidence by Class",
    accuracy: Optional[float] = None,
    figsize: Tuple[int, int] = (12, 8),
    threshold: float = 0.7,
    min_confidence: float = 0.0,
    max_confidence: float = 1.0,
    sort_by: str = "index",  # "index", "confidence", "count"
    use_dark_theme: bool = True,
    fig_background_color: str = "#121212",
    text_color: str = "#FFFFFF",
) -> plt.Figure:
    """
    Visualize model confidence by class.

    Args:
        class_confidences: Dictionary of class index to mean confidence
        class_counts: Dictionary of class index to count of samples
        class_names: Optional list of class names (if None, will use indices)
        output_path: Optional path to save the visualization
        title: Title for the plot
        accuracy: Optional overall accuracy to display
        figsize: Figure size as (width, height)
        threshold: Confidence threshold to highlight (typically 0.7)
        min_confidence: Minimum confidence for y-axis
        max_confidence: Maximum confidence for y-axis
        sort_by: How to sort classes ("index", "confidence", "count")
        use_dark_theme: Whether to use dark theme
        fig_background_color: Background color for the figure
        text_color: Text color

    Returns:
        Matplotlib figure object
    """
    # Set theme
    if use_dark_theme:
        plt.style.use("dark_background")
        plt.rcParams.update(
            {
                "figure.facecolor": fig_background_color,
                "axes.facecolor": fig_background_color,
                "text.color": text_color,
                "axes.labelcolor": text_color,
                "xtick.color": text_color,
                "ytick.color": text_color,
                "legend.facecolor": "gray",
                "legend.edgecolor": "white",
            }
        )

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Extract data
    classes = sorted(class_confidences.keys())
    confidences = [class_confidences[c] for c in classes]
    counts = [class_counts[c] for c in classes]

    # Create class labels
    if class_names is not None:
        labels = [
            class_names[c] if c < len(class_names) else f"Class {c}" for c in classes
        ]
    else:
        labels = [f"Class {c}" for c in classes]

    # Sort data if requested
    if sort_by == "confidence":
        sorted_indices = np.argsort(confidences)
    elif sort_by == "count":
        sorted_indices = np.argsort(counts)
    else:  # Default to index
        sorted_indices = np.arange(len(classes))

    # Apply sorting
    classes = [classes[i] for i in sorted_indices]
    confidences = [confidences[i] for i in sorted_indices]
    counts = [counts[i] for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]

    # Set colors based on confidence level
    cmap = plt.cm.get_cmap("RdYlGn")
    colors = [cmap(conf) for conf in confidences]

    # Create colormap for counts
    count_norm = plt.Normalize(min(counts), max(counts))
    count_colors = plt.cm.viridis(count_norm(counts))

    # Create bars with gradient color based on confidence
    bars = ax.bar(
        range(len(classes)),
        confidences,
        color=colors,
        alpha=0.8,
        edgecolor="gray",
        linewidth=0.5,
    )

    # Add horizontal line for threshold
    ax.axhline(y=threshold, linestyle="--", color="white", alpha=0.7, linewidth=1)
    ax.text(
        len(classes) - 1,
        threshold,
        f"Threshold: {threshold:.1f}",
        ha="right",
        va="bottom",
        color="white",
        fontsize=10,
    )

    # Add labels and title
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Mean Confidence", fontsize=12)
    ax.set_ylim(min_confidence, max_confidence)

    if accuracy is not None:
        title = f"{title} (Accuracy: {accuracy:.2%})"
    ax.set_title(title, fontsize=16)

    # Set x-axis ticks
    if len(classes) > 30:
        # Too many classes, show a subset of ticks
        tick_step = max(1, len(classes) // 20)
        ax.set_xticks(range(0, len(classes), tick_step))
        ax.set_xticklabels(
            [labels[i] for i in range(0, len(classes), tick_step)],
            rotation=45,
            ha="right",
        )
    else:
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(labels, rotation=45, ha="right")

    # Add count labels above bars
    for i, (count, conf) in enumerate(zip(counts, confidences)):
        text_color = "white" if conf < 0.6 else "black"
        ax.text(
            i,
            conf + 0.02,
            str(count),
            ha="center",
            va="bottom",
            color=text_color,
            fontsize=8,
            rotation=0,
            fontstyle="italic",
        )

    # Add legend for count scale
    from matplotlib.cm import ScalarMappable

    sm = ScalarMappable(cmap=plt.cm.viridis, norm=count_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01, shrink=0.5)
    cbar.set_label("Sample Count")

    # Adjust layout
    plt.tight_layout()

    # Save figure if output path specified
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved confidence distribution plot to {output_path}")

    return fig


def plot_confidence_timeline(
    history: List[Dict],
    metric_key: str = "mean_confidence",
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Confidence Evolution During Training",
    figsize: Tuple[int, int] = (10, 6),
    threshold: float = 0.7,
    use_dark_theme: bool = True,
    show_accuracy: bool = True,
    fig_background_color: str = "#121212",
    text_color: str = "#FFFFFF",
) -> plt.Figure:
    """
    Plot the evolution of confidence metrics over training epochs.

    Args:
        history: List of dictionaries containing metrics for each epoch
        metric_key: Key for the confidence metric to plot
        output_path: Optional path to save the visualization
        title: Title for the plot
        figsize: Figure size as (width, height)
        threshold: Confidence threshold to highlight
        use_dark_theme: Whether to use dark theme
        show_accuracy: Whether to show accuracy on the same plot
        fig_background_color: Background color for the figure
        text_color: Text color

    Returns:
        Matplotlib figure object
    """
    # Set theme
    if use_dark_theme:
        plt.style.use("dark_background")
        plt.rcParams.update(
            {
                "figure.facecolor": fig_background_color,
                "axes.facecolor": fig_background_color,
                "text.color": text_color,
                "axes.labelcolor": text_color,
                "xtick.color": text_color,
                "ytick.color": text_color,
                "legend.facecolor": "gray",
                "legend.edgecolor": "white",
            }
        )

    # Create figure
    fig, ax1 = plt.subplots(figsize=figsize)

    # Extract confidence data
    epochs = [entry.get("epoch", i + 1) for i, entry in enumerate(history)]
    confidence_values = [entry.get(metric_key, None) for entry in history]

    # Filter out None values
    valid_points = [(e, c) for e, c in zip(epochs, confidence_values) if c is not None]
    if not valid_points:
        logger.warning(f"No valid confidence data found for metric: {metric_key}")
        return fig

    valid_epochs, valid_confidence = zip(*valid_points)

    # Plot confidence
    confidence_line = ax1.plot(
        valid_epochs,
        valid_confidence,
        "o-",
        color="#3498db",
        linewidth=2,
        markersize=6,
        label="Mean Confidence",
    )
    ax1.set_ylabel("Confidence", fontsize=12, color="#3498db")
    ax1.set_ylim(0, 1.05)

    # Add threshold line
    ax1.axhline(y=threshold, linestyle="--", color="white", alpha=0.7, linewidth=1)
    ax1.text(
        max(valid_epochs),
        threshold,
        f"Threshold: {threshold:.1f}",
        ha="right",
        va="bottom",
        color="white",
        fontsize=10,
    )

    # Plot accuracy on secondary axis if requested
    if show_accuracy and all("accuracy" in entry for entry in history):
        accuracy_values = [entry.get("accuracy", None) for entry in history]
        valid_acc_points = [
            (e, a) for e, a in zip(epochs, accuracy_values) if a is not None
        ]

        if valid_acc_points:
            valid_acc_epochs, valid_accuracy = zip(*valid_acc_points)
            ax2 = ax1.twinx()
            accuracy_line = ax2.plot(
                valid_acc_epochs,
                valid_accuracy,
                "o-",
                color="#e74c3c",
                linewidth=2,
                markersize=6,
                label="Accuracy",
            )
            ax2.set_ylabel("Accuracy", fontsize=12, color="#e74c3c")
            ax2.set_ylim(0, 1.05)

            # Add combined legend
            lines = confidence_line + accuracy_line
            labels = [line.get_label() for line in lines]
            ax1.legend(lines, labels, loc="lower right")
        else:
            ax1.legend(loc="lower right")
    else:
        ax1.legend(loc="lower right")

    # Add labels and title
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_title(title, fontsize=16)

    # Set integer ticks for epochs
    ax1.set_xticks(valid_epochs)
    ax1.set_xlim(min(valid_epochs) - 0.5, max(valid_epochs) + 0.5)

    # Add grid
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Save figure if output path specified
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved confidence timeline plot to {output_path}")

    return fig


def plot_ece_by_class(
    confidence_data: Dict[str, Dict],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Expected Calibration Error by Class",
    figsize: Tuple[int, int] = (12, 8),
    top_n: int = 20,
    use_dark_theme: bool = True,
    fig_background_color: str = "#121212",
    text_color: str = "#FFFFFF",
) -> plt.Figure:
    """
    Visualize Expected Calibration Error (ECE) by class.

    Args:
        confidence_data: Dictionary containing calibration data
        output_path: Optional path to save the visualization
        title: Title for the plot
        figsize: Figure size as (width, height)
        top_n: Number of top classes to display
        use_dark_theme: Whether to use dark theme
        fig_background_color: Background color for the figure
        text_color: Text color

    Returns:
        Matplotlib figure object
    """
    # Set theme
    if use_dark_theme:
        plt.style.use("dark_background")
        plt.rcParams.update(
            {
                "figure.facecolor": fig_background_color,
                "axes.facecolor": fig_background_color,
                "text.color": text_color,
                "axes.labelcolor": text_color,
                "xtick.color": text_color,
                "ytick.color": text_color,
                "legend.facecolor": "gray",
                "legend.edgecolor": "white",
            }
        )

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Extract calibration data
    classes = []
    ece_values = []
    sample_counts = []

    for class_name, data in confidence_data.items():
        # Skip classes with no samples
        if data["sample_count"] == 0:
            continue

        classes.append(class_name)
        ece_values.append(data["ece"])
        sample_counts.append(data["sample_count"])

    # Sort by ECE (descending)
    sorted_indices = np.argsort(ece_values)[::-1]

    # Limit to top_n classes
    if len(sorted_indices) > top_n:
        sorted_indices = sorted_indices[:top_n]

    # Extract sorted data
    sorted_classes = [classes[i] for i in sorted_indices]
    sorted_ece = [ece_values[i] for i in sorted_indices]
    sorted_counts = [sample_counts[i] for i in sorted_indices]

    # Create colormap for ECE values
    cmap = plt.cm.get_cmap("coolwarm_r")
    colors = [cmap(ece) for ece in sorted_ece]

    # Plot ECE bars
    bars = ax.bar(
        range(len(sorted_classes)),
        sorted_ece,
        color=colors,
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
    )

    # Add counts as text on bars
    for i, (count, ece) in enumerate(zip(sorted_counts, sorted_ece)):
        text_color = "white" if ece > 0.25 else "black"
        ax.text(
            i,
            ece + 0.01,
            str(count),
            ha="center",
            va="bottom",
            color=text_color,
            fontsize=8,
            rotation=0,
            fontstyle="italic",
        )

    # Add labels and title
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Expected Calibration Error (ECE)", fontsize=12)
    ax.set_title(title, fontsize=16)

    # Set x-axis ticks
    ax.set_xticks(range(len(sorted_classes)))
    ax.set_xticklabels(sorted_classes, rotation=45, ha="right")

    # Add horizontal grid lines
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)

    # Add overall ECE line if available
    if "overall_ece" in confidence_data:
        overall_ece = confidence_data["overall_ece"]
        ax.axhline(
            y=overall_ece, linestyle="--", color="yellow", alpha=0.7, linewidth=1
        )
        ax.text(
            len(sorted_classes) - 1,
            overall_ece,
            f"Overall ECE: {overall_ece:.3f}",
            ha="right",
            va="bottom",
            color="yellow",
            fontsize=10,
        )

    # Adjust layout
    plt.tight_layout()

    # Save figure if output path specified
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved ECE by class plot to {output_path}")

    return fig


def save_confidence_visualizations(
    history: List[Dict],
    class_names: Optional[List[str]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    ece_data: Optional[Dict] = None,
    use_dark_theme: bool = True,
    latest_epoch_only: bool = False,
) -> List[Path]:
    """
    Generate and save confidence visualization plots.

    Args:
        history: List of dictionaries containing metrics for each epoch
        class_names: Optional list of class names
        output_dir: Directory to save visualizations
        ece_data: Optional ECE data by class
        use_dark_theme: Whether to use dark theme
        latest_epoch_only: If True, only plot the most recent epoch

    Returns:
        List of paths to the generated plots
    """
    if not history:
        logger.warning("No confidence history data provided")
        return []

    if output_dir is None:
        logger.warning("No output directory specified for confidence visualizations")
        return []

    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    generated_plots = []

    # Create confidence timeline plot
    timeline_path = output_dir / "confidence_timeline.png"
    plot_confidence_timeline(
        history=history,
        metric_key="mean_confidence",
        output_path=timeline_path,
        title="Confidence Evolution During Training",
        use_dark_theme=use_dark_theme,
        show_accuracy=True,
    )
    generated_plots.append(timeline_path)

    # Get data for latest epoch
    latest_epoch = None
    if history:
        if latest_epoch_only:
            # Only plot the latest epoch
            latest_epoch = history[-1]
        else:
            # Find epoch with highest accuracy
            latest_epoch = max(history, key=lambda x: x.get("accuracy", 0))

    if latest_epoch:
        # Extract class confidence data
        class_confidences = {}
        class_counts = {}

        for key, value in latest_epoch.items():
            if key.startswith("class_") and key.endswith("_mean_conf"):
                class_idx = int(key.split("_")[1])
                class_confidences[class_idx] = value
            elif key.startswith("class_") and key.endswith("_count"):
                class_idx = int(key.split("_")[1])
                class_counts[class_idx] = value

        if class_confidences and class_counts:
            # Create confidence distribution plot
            dist_path = output_dir / "confidence_by_class.png"
            epoch_num = latest_epoch.get("epoch", len(history))
            accuracy = latest_epoch.get("accuracy", None)

            plot_confidence_distribution_by_class(
                class_confidences=class_confidences,
                class_counts=class_counts,
                class_names=class_names,
                output_path=dist_path,
                title=f"Mean Confidence by Class (Epoch {epoch_num})",
                accuracy=accuracy,
                use_dark_theme=use_dark_theme,
                sort_by="confidence",  # Sort by confidence level (ascending)
            )
            generated_plots.append(dist_path)

    # Plot ECE by class if available
    if ece_data:
        ece_path = output_dir / "ece_by_class.png"
        plot_ece_by_class(
            confidence_data=ece_data,
            output_path=ece_path,
            title="Expected Calibration Error by Class",
            use_dark_theme=use_dark_theme,
        )
        generated_plots.append(ece_path)

    logger.info(
        f"Generated {len(generated_plots)} confidence visualization plots in {output_dir}"
    )
    return generated_plots


def generate_confidence_report(
    confidence_history_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    class_names: Optional[List[str]] = None,
    use_dark_theme: bool = True,
) -> Dict:
    """
    Generate confidence visualization report from history JSON file.

    Args:
        confidence_history_path: Path to confidence history JSON file
        output_dir: Directory to save visualization plots
        class_names: Optional list of class names
        use_dark_theme: Whether to use dark theme

    Returns:
        Dictionary with paths to generated plots
    """
    confidence_history_path = Path(confidence_history_path)

    if not confidence_history_path.exists():
        logger.error(f"Confidence history file not found: {confidence_history_path}")
        return {}

    if output_dir is None:
        output_dir = confidence_history_path.parent / "confidence_plots"

    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    # Load confidence history
    try:
        with open(confidence_history_path, "r") as f:
            history = json.load(f)

        if not isinstance(history, list):
            logger.error("Invalid confidence history format (expected list)")
            return {}

        logger.info(
            f"Loaded confidence history with {len(history)} epochs from {confidence_history_path}"
        )
    except Exception as e:
        logger.error(f"Failed to load confidence history: {e}")
        return {}

    # Generate visualizations
    plot_paths = save_confidence_visualizations(
        history=history,
        class_names=class_names,
        output_dir=output_dir,
        use_dark_theme=use_dark_theme,
    )

    result = {
        "output_dir": str(output_dir),
        "plots": [str(p) for p in plot_paths],
        "num_epochs": len(history),
    }

    # Save a summary
    summary_path = output_dir / "confidence_summary.json"
    try:
        with open(summary_path, "w") as f:
            json.dump(result, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save confidence summary: {e}")

    return result
