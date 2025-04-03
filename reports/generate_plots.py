"""
Generate plots for training reports.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from core.visualization.base_visualization import (
    DEFAULT_THEME,
    apply_dark_theme,
    plot_class_metrics,
    plot_learning_rate,
    plot_training_history,
    plot_training_time,
)
from utils.logger import get_logger
from utils.paths import ensure_dir

# Import enhanced visualization functions
try:
    from core.visualization.visualization import (
        create_augmentation_grid,
        create_classification_examples_grid,
        plot_categorical,
        plot_class_performance,
        plot_confidence_distribution,
        plot_enhanced_confusion_matrix,
        plot_feature_space,
        plot_histogram,
        plot_precision_recall_curve,
        plot_roc_curve,
        plot_tsne_visualization,
        plot_umap_visualization,
        visualize_augmentations,
    )

    ENHANCED_VISUALIZATION_AVAILABLE = True
except ImportError:
    ENHANCED_VISUALIZATION_AVAILABLE = False
    from core.visualization.base_visualization import (
        plot_confusion_matrix as plot_enhanced_confusion_matrix,
    )

logger = get_logger(__name__)


def load_json(file_path: Union[str, Path]) -> Dict:
    """
    Load data from a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Dictionary of loaded data
    """
    try:
        with open(file_path) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON from {file_path}: {e}")
        return {}


def load_history(experiment_dir: Union[str, Path]) -> Dict:
    """
    Load training history from JSON file.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        Dictionary with training history data
    """
    # Try multiple possible locations in order
    possible_paths = [
        Path(experiment_dir) / "metrics" / "history.json",
        Path(experiment_dir) / "history.json",
    ]

    for history_path in possible_paths:
        if history_path.exists():
            logger.info(f"Found history file at {history_path}")
            return load_json(history_path)

    logger.warning(f"No history file found in {experiment_dir}")
    return {}


def load_metrics(experiment_dir: Union[str, Path]) -> Dict:
    """
    Load evaluation metrics from JSON file.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        Dictionary with metrics data
    """
    # Try multiple possible locations in order
    possible_paths = [
        Path(experiment_dir) / "metrics" / "metrics.json",
        Path(experiment_dir) / "metrics.json",
    ]

    for metrics_path in possible_paths:
        if metrics_path.exists():
            logger.info(f"Found metrics file at {metrics_path}")
            return load_json(metrics_path)

    logger.warning(f"No metrics file found in {experiment_dir}")
    return {}


def load_confusion_matrix(experiment_dir: Union[str, Path]) -> Optional[np.ndarray]:
    """
    Load confusion matrix from an experiment directory.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        Confusion matrix as numpy array, or None if not found
    """
    experiment_dir = Path(experiment_dir)

    # Try different possible locations for the confusion matrix
    possible_paths = [
        experiment_dir / "evaluation_artifacts" / "confusion_matrix.npy",
    ]

    # Log all paths being checked
    logger.info("Looking for confusion matrix in the following locations:")
    for path in possible_paths:
        logger.info(f"  - {path}")
        if path.exists():
            logger.info(f"Found confusion matrix at {path}")
            try:
                return np.load(path)
            except Exception as e:
                logger.error(f"Failed to load confusion matrix from {path}: {e}")

    # If we get here, no valid confusion matrix was found
    logger.warning(f"No confusion matrix file found in {experiment_dir}")
    return None


def load_class_names(experiment_dir: Union[str, Path]) -> List[str]:
    """
    Load class names from an experiment directory.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        List of class names
    """
    class_names_path = Path(experiment_dir) / "class_names.txt"

    if not class_names_path.exists():
        logger.warning(f"No class_names.txt found in {experiment_dir}")
        return []

    try:
        with open(class_names_path) as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Failed to load class names from {class_names_path}: {e}")
        return []


def load_visualization_theme(experiment_dir: Union[str, Path]) -> Dict:
    """
    Load visualization theme from config file.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        Dictionary with theme settings
    """
    # Try to load config.yaml
    config_path = Path(experiment_dir) / "config.yaml"

    if not config_path.exists():
        logger.warning(
            f"No config.yaml found in {experiment_dir}, using default dark theme"
        )
        return DEFAULT_THEME

    try:
        cfg = OmegaConf.load(config_path)
        if (
            "prepare_data" in cfg
            and "visualization_theme" in cfg.prepare_data
            and cfg.prepare_data.visualization_theme
        ):
            # Extract theme from config
            theme = OmegaConf.to_container(cfg.prepare_data.visualization_theme)
            logger.info("Loaded custom visualization theme from config")
            return theme
        else:
            logger.info(
                "No visualization_theme found in config, using default dark theme"
            )
            return DEFAULT_THEME
    except Exception as e:
        logger.error(f"Error loading theme from config: {e}")
        return DEFAULT_THEME


def plot_training_history_from_file(
    history: Dict[str, List[float]],
    output_path: Union[str, Path],
    theme: Optional[Dict] = None,
) -> None:
    """
    Plot training history from a history dictionary.

    Args:
        history: Dictionary of training history
        output_path: Path to save the plot
        theme: Theme settings to apply
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    # Filter out non-scalar values
    scalar_history = {}
    for key, values in history.items():
        if isinstance(values, list) and all(
            isinstance(v, (int, float)) for v in values
        ):
            scalar_history[key] = values

    # Identify key metrics to plot
    metrics_to_plot = []
    for metric in ["loss", "accuracy", "precision", "recall", "f1"]:
        # Check if we have both train and val versions
        train_key = f"train_{metric}" if f"train_{metric}" in scalar_history else metric
        val_key = f"val_{metric}"

        if train_key in scalar_history and val_key in scalar_history:
            metrics_to_plot.append(metric)

    if not metrics_to_plot:
        # Try direct metrics if no train/val versions
        metrics_to_plot = [
            m
            for m in ["loss", "accuracy", "precision", "recall", "f1"]
            if m in scalar_history
        ]

    if not metrics_to_plot:
        logger.warning("No suitable metrics found for plotting training history")
        return

    # Plot training history with theme
    plot_training_history(
        history=scalar_history,
        output_path=output_path,
        metrics=metrics_to_plot,
        theme=theme,
    )

    logger.info(f"Saved training history plot to {output_path}")


def plot_confusion_matrix_from_file(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Union[str, Path],
    theme: Optional[Dict] = None,
) -> None:
    """
    Plot confusion matrix from a numpy array.

    Args:
        cm: Confusion matrix
        class_names: List of class names
        output_path: Path to save the plot
        theme: Theme settings to apply
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    # Use enhanced confusion matrix if available
    if ENHANCED_VISUALIZATION_AVAILABLE:
        logger.info("Using enhanced confusion matrix visualization")

        # Plot confusion matrix with theme
        plot_enhanced_confusion_matrix(
            cm=cm,
            class_names=class_names,
            output_path=output_path,
            normalize=True,
            theme=theme,
            cmap="YlOrRd",
            title="Confusion Matrix",
        )
    else:
        # Fall back to standard visualization
        plot_enhanced_confusion_matrix(
            cm=cm,
            class_names=class_names,
            output_path=output_path,
            normalize=True,
            theme=theme,
        )

    logger.info(f"Saved confusion matrix plot to {output_path}")


def plot_training_time_from_history(
    history: Dict[str, List[float]],
    output_path: Union[str, Path],
    theme: Optional[Dict] = None,
) -> None:
    """
    Plot training time from history.

    Args:
        history: Dictionary of training history
        output_path: Path to save the plot
        theme: Theme settings to apply
    """
    # Check if we have time per epoch
    time_key = "time" if "time" in history else "train_time"

    if time_key not in history:
        logger.warning("No training time data found in history")
        return

    times = history[time_key]

    # Plot training time with theme
    plot_training_time(
        epoch_times=times,
        output_path=output_path,
        theme=theme,
    )

    logger.info(f"Saved training time plot to {output_path}")


def plot_class_metrics_from_metrics(
    metrics: Dict,
    class_names: List[str],
    output_path: Union[str, Path],
    theme: Optional[Dict] = None,
) -> None:
    """
    Plot class metrics from a metrics dictionary.

    Args:
        metrics: Dictionary of metrics
        class_names: List of class names
        output_path: Path to save the plot
        theme: Theme settings to apply
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    # Extract class metrics
    class_metrics = {}
    metrics_to_plot = ["accuracy", "precision", "recall", "f1"]

    for class_name in class_names:
        class_key = class_name.replace(" ", "_")
        class_metrics[class_name] = {}

        for metric in metrics_to_plot:
            metric_key = f"class_{class_key}_{metric}"
            if metric_key in metrics:
                class_metrics[class_name][metric] = metrics[metric_key]

    if not class_metrics:
        logger.warning("No class metrics found in metrics dictionary")
        return

    # Plot class metrics with theme
    plot_class_metrics(
        metrics=metrics,
        class_names=class_names,
        output_path=output_path,
        theme=theme,
    )

    logger.info(f"Saved class metrics plot to {output_path}")


def plot_learning_rate_from_history(
    history: Dict[str, List[float]],
    output_path: Union[str, Path],
    theme: Optional[Dict] = None,
) -> None:
    """
    Plot learning rate from history.

    Args:
        history: Dictionary of training history
        output_path: Path to save the plot
        theme: Theme settings to apply
    """
    # Check if we have learning rate data
    lr_key = "lr" if "lr" in history else "learning_rate"

    if lr_key not in history:
        logger.warning("No learning rate data found in history")
        return

    lr_history = history[lr_key]

    # Plot learning rate with theme
    plot_learning_rate(
        lr_history=lr_history,
        output_path=output_path,
        theme=theme,
    )

    logger.info(f"Saved learning rate plot to {output_path}")


def normalize_image_for_display(image_array: np.ndarray) -> np.ndarray:
    """
    Convert various image formats to a consistent numpy array format for display.

    Args:
        image_array: Image data in various formats

    Returns:
        Image as a numpy array in HWC format with values in [0, 255] as uint8
    """
    # Check if the input is valid
    if image_array is None or not isinstance(image_array, np.ndarray):
        logger.error(f"Invalid image data type: {type(image_array)}")
        return np.zeros((224, 224, 3), dtype=np.uint8)

    # Check for empty arrays
    if image_array.size == 0:
        logger.error("Empty image array")
        return np.zeros((224, 224, 3), dtype=np.uint8)

    # Handle channel dimensions
    if len(image_array.shape) == 4:
        if image_array.shape[0] == 1:  # Batch dimension of 1
            image_array = image_array[0]
        else:
            logger.warning(
                f"Unexpected batch dimension: {image_array.shape}, taking first image"
            )
            image_array = image_array[0]

    # Convert from CHW to HWC if needed
    if len(image_array.shape) == 3 and image_array.shape[0] in [1, 3, 4]:
        image_array = np.transpose(image_array, (1, 2, 0))

    # Ensure three channels (convert grayscale to RGB)
    if len(image_array.shape) == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    elif len(image_array.shape) == 3 and image_array.shape[2] == 1:
        image_array = np.concatenate([image_array] * 3, axis=2)

    # Handle 4 channels (RGBA) - discard alpha
    if len(image_array.shape) == 3 and image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]

    # Normalize values to 0-255 range
    if image_array.dtype == np.float32 or image_array.dtype == np.float64:
        # Check if normalized between 0 and 1
        if np.max(image_array) <= 1.0:
            image_array = (image_array * 255).astype(np.uint8)
        # Check if normalized with ImageNet stats
        elif np.min(image_array) < 0:
            # Approximately reverse ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])

            # Handle both HWC and CHW formats
            if image_array.shape[2] == 3:  # HWC
                for i in range(3):
                    image_array[:, :, i] = image_array[:, :, i] * std[i] + mean[i]
            else:  # Assume CHW
                for i in range(3):
                    image_array[i] = image_array[i] * std[i] + mean[i]

            image_array = np.clip(image_array, 0, 1)
            image_array = (image_array * 255).astype(np.uint8)
        else:
            # Just clip and convert to uint8
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)

    # Ensure uint8 type for cv2/PIL compatibility
    if image_array.dtype != np.uint8:
        image_array = image_array.astype(np.uint8)

    return image_array


def plot_top_misclassifications(
    images: np.ndarray,
    true_labels: np.ndarray,
    predictions: np.ndarray,
    class_names: List[str],
    misclassified_indices: np.ndarray,
    output_path: Union[str, Path],
    max_examples: int = 20,
    theme: Optional[Dict] = None,
) -> None:
    """
    Plot top misclassifications with highest confidence.

    Args:
        images: Array of images
        true_labels: Array of true labels (either class indices or one-hot encoded)
        predictions: Array of model predictions (logits or probabilities)
        class_names: List of class names
        misclassified_indices: Array of indices of misclassified examples
        output_path: Path to save the plot
        max_examples: Maximum number of examples to display
        theme: Visualization theme configuration
    """
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    # Apply theme
    apply_theme(theme)

    # Convert labels to indices if one-hot encoded
    if true_labels.ndim > 1:
        true_indices = np.argmax(true_labels, axis=1)
    else:
        true_indices = true_labels

    # Convert predictions to indices and confidences
    if predictions.ndim > 1:
        pred_indices = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)
    else:
        pred_indices = np.round(predictions).astype(int)
        confidences = np.abs(predictions - 0.5) + 0.5  # Convert to confidence

    # Check if we have any misclassified samples
    if len(misclassified_indices) == 0:
        logger.warning("No misclassified examples to visualize")
        return

    # Get misclassification confidence
    misclassified_confidences = confidences[misclassified_indices]

    # Sort by confidence (high to low)
    sorted_indices = np.argsort(-misclassified_confidences)
    top_indices = sorted_indices[:max_examples]

    # Get actual indices in the original data
    top_misclassified = misclassified_indices[top_indices]

    # Determine grid size
    n_examples = min(len(top_misclassified), max_examples)

    # Calculate grid dimensions
    cols = min(5, n_examples)
    rows = (n_examples + cols - 1) // cols

    # Create figure
    fig = plt.figure(figsize=(cols * 3, rows * 3))

    # Create gridspec for better control over spacing
    gs = gridspec.GridSpec(rows, cols, figure=fig, wspace=0.1, hspace=0.3)

    # Plot each misclassified example
    for i, idx in enumerate(top_misclassified):
        if i >= max_examples:
            break

        try:
            # Get image and labels
            image = images[idx]

            # Normalize image to consistent format
            image = normalize_image_for_display(image)

            true_idx = true_indices[idx]
            pred_idx = pred_indices[idx]
            conf = confidences[idx]

            # Get class names
            true_name = (
                class_names[true_idx]
                if true_idx < len(class_names)
                else f"Class {true_idx}"
            )
            pred_name = (
                class_names[pred_idx]
                if pred_idx < len(class_names)
                else f"Class {pred_idx}"
            )

            # Create subplot
            ax = fig.add_subplot(gs[i])

            # Display image
            ax.imshow(image)

            # Add labels
            ax.set_title(
                f"True: {true_name}\nPred: {pred_name}\nConf: {conf:.2f}",
                fontsize=8,
                pad=2,
            )

            # Remove axes
            ax.axis("off")
        except Exception as e:
            logger.error(f"Error plotting misclassified example {idx}: {e}")
            # Create empty subplot
            ax = fig.add_subplot(gs[i])
            ax.text(
                0.5,
                0.5,
                "Error loading image",
                horizontalalignment="center",
                verticalalignment="center",
            )
            ax.axis("off")

    plt.suptitle("Top Misclassifications", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Top misclassifications plot saved to {output_path}")


def plot_class_difficulties(
    per_class_metrics: Dict,
    output_path: Union[str, Path],
    theme: Optional[Dict] = None,
) -> None:
    """
    Create a scatter plot showing precision vs recall for each class,
    with point size representing class frequency (support).

    Args:
        per_class_metrics: Dictionary of per-class metrics
        output_path: Path to save the figure
        theme: Visualization theme
    """
    # Extract metrics for plotting
    classes = []
    precisions = []
    recalls = []
    supports = []

    for class_name, metrics in per_class_metrics.items():
        if class_name in ["accuracy", "macro avg", "weighted avg"]:
            continue

        classes.append(class_name)
        precisions.append(metrics["precision"])
        recalls.append(metrics["recall"])
        supports.append(metrics["support"])

    # Apply theme if provided
    if theme:
        apply_theme(theme)

    # Create figure
    plt.figure(figsize=(12, 10), facecolor="#121212")

    # Calculate size based on support (normalize to a reasonable size range)
    max_support = max(supports)
    sizes = [100 + 900 * (s / max_support) for s in supports]

    # Create scatter plot
    scatter = plt.scatter(
        recalls, precisions, s=sizes, c=recalls, cmap="viridis", alpha=0.7
    )

    # Add color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label("Recall", color="white")

    # Add diagonal line for precision = recall
    plt.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)

    # Add labels for each point
    for i, class_name in enumerate(classes):
        # Truncate long class names
        if len(class_name) > 20:
            display_name = class_name[:17] + "..."
        else:
            display_name = class_name

        plt.annotate(
            display_name,
            (recalls[i], precisions[i]),
            fontsize=8,
            color="white",
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.title("Class Performance Map: Precision vs Recall", color="white", fontsize=16)
    plt.xlabel("Recall", color="white")
    plt.ylabel("Precision", color="white")
    plt.grid(True, alpha=0.2)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)

    # Add legend for bubble size
    sizes_legend = [min(supports), max(supports)]
    labels_legend = [f"Min Support: {min(supports)}", f"Max Support: {max(supports)}"]

    # Create a legend proxy
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=labels_legend[i],
            markerfacecolor="gray",
            markersize=np.sqrt(100 + 900 * (s / max_support)) / 2,
        )
        for i, s in enumerate(sizes_legend)
    ]
    plt.legend(handles=legend_elements, loc="lower left", frameon=False)

    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="#121212")
    plt.close()
    logger.info(f"Saved class difficulty plot to {output_path}")


def plot_calibration_curves(
    calibration_data: Dict[int, Dict[str, np.ndarray]],
    output_path: Union[str, Path],
    max_classes: int = 10,
    theme: Optional[Dict] = None,
) -> None:
    """
    Plot calibration curves for each class (reliability diagram).

    Args:
        calibration_data: Dictionary containing calibration data for each class
        output_path: Path to save the plot
        max_classes: Maximum number of classes to plot
        theme: Visualization theme configuration
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # Set plot style
    apply_theme(theme)

    plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 1, height_ratios=[3, 1])

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    # Add reference diagonal line
    ax1.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")

    # Calculate calibration gap (ECE) for each class
    calibration_gaps = {}
    for class_idx, class_data in calibration_data.items():
        # Extract data
        prob_true = class_data["prob_true"]
        prob_pred = class_data["prob_pred"]

        # Calculate expected calibration error (ECE)
        bin_proportion = class_data["bin_count"] / np.sum(class_data["bin_count"])
        calibration_gap = np.sum(bin_proportion * np.abs(prob_true - prob_pred))
        calibration_gaps[class_idx] = calibration_gap

    # Sort classes by calibration gap (highest to lowest)
    sorted_classes = sorted(
        calibration_gaps.keys(), key=lambda x: calibration_gaps[x], reverse=True
    )

    # Plot only the top N classes with highest calibration gap
    for i, class_idx in enumerate(sorted_classes[:max_classes]):
        class_data = calibration_data[class_idx]

        # Plot calibration curve
        ax1.plot(
            class_data["prob_pred"],
            class_data["prob_true"],
            marker="o",
            linestyle="-",
            label=f"Class {class_idx} (ECE: {calibration_gaps[class_idx]:.3f})",
        )

        # Plot calibration gap (bar chart in the second subplot)
        ax2.bar(i, calibration_gaps[class_idx], alpha=0.7)

    # Set axis labels and title
    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Fraction of Positives")
    ax1.set_title("Calibration Curves (Reliability Diagram)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize="small")

    ax2.set_title("Expected Calibration Error (ECE) by Class")
    ax2.set_xlabel("Class Index")
    ax2.set_ylabel("ECE")
    ax2.set_xticks(range(min(max_classes, len(sorted_classes))))
    ax2.set_xticklabels([str(idx) for idx in sorted_classes[:max_classes]])
    ax2.grid(True, alpha=0.3)

    # Calculate overall ECE
    overall_ece = np.mean(list(calibration_gaps.values()))
    plt.figtext(0.5, 0.01, f"Overall ECE: {overall_ece:.4f}", ha="center", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Calibration curves saved to {output_path}")


def plot_embeddings(
    embedding_data: Dict[str, np.ndarray],
    class_names: List[str],
    output_path: Union[str, Path],
    theme: Optional[Dict] = None,
) -> None:
    """
    Plot 2D embeddings visualization using t-SNE or UMAP.

    Args:
        embedding_data: Dictionary containing embeddings and labels
        class_names: List of class names
        output_path: Path to save the plot
        theme: Visualization theme configuration
    """
    import matplotlib.pyplot as plt

    # Set plot style
    apply_theme(theme)

    # Extract data
    embeddings = embedding_data["embeddings"]
    labels = embedding_data["labels"]
    method = embedding_data.get("method", "UMAP")  # Default to UMAP if not specified

    # Convert to numpy arrays if needed
    embeddings = np.array(embeddings)
    labels = np.array(labels)

    plt.figure(figsize=(12, 10))

    # Create a scatter plot with different colors for each class
    unique_labels = np.unique(labels)

    # Get colors from theme if available
    colors = get_theme_colors(theme, len(unique_labels))

    # Plot each class
    for i, label in enumerate(unique_labels):
        idx = labels == label

        # Check if we have class names
        if i < len(class_names):
            class_label = class_names[label]
        else:
            class_label = f"Class {label}"

        plt.scatter(
            embeddings[idx, 0],
            embeddings[idx, 1],
            c=[colors[i % len(colors)]],
            label=class_label,
            alpha=0.7,
            edgecolors="none",
            s=30,
        )

    # Set title and labels
    plt.title(f"Feature Space Visualization using {method}")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")

    # Add legend with small font size
    if len(unique_labels) > 10:
        plt.legend(
            fontsize="small",
            markerscale=1.5,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        )
    else:
        plt.legend(fontsize="medium", markerscale=1.5, loc="best")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Feature space visualization saved to {output_path}")


def apply_theme(theme: Optional[Dict] = None) -> None:
    """
    Apply visualization theme to matplotlib plots.

    Args:
        theme: Dictionary containing theme configuration
    """
    import matplotlib.pyplot as plt

    # Default dark theme if none specified
    if theme is None:
        theme = {
            "figure.facecolor": "#121212",
            "axes.facecolor": "#1E1E1E",
            "axes.edgecolor": "#404040",
            "axes.labelcolor": "#FFFFFF",
            "axes.titlecolor": "#FFFFFF",
            "xtick.color": "#FFFFFF",
            "ytick.color": "#FFFFFF",
            "grid.color": "#333333",
            "text.color": "#FFFFFF",
            "legend.facecolor": "#1E1E1E",
            "legend.edgecolor": "#333333",
            "legend.textcolor": "#FFFFFF",
        }

    # Set style based on theme
    plt.style.use("dark_background" if theme.get("dark", True) else "default")

    # Apply specific theme parameters
    for key, value in theme.items():
        if key in plt.rcParams:
            plt.rcParams[key] = value


def get_theme_colors(theme: Optional[Dict] = None, num_colors: int = 10) -> List[str]:
    """
    Get colors from theme or generate a color palette.

    Args:
        theme: Dictionary containing theme configuration
        num_colors: Number of colors needed

    Returns:
        List of colors as hex strings
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    # If theme has colors defined, use those
    if theme and "colors" in theme:
        colors = theme["colors"]
        # If not enough colors, cycle through them
        if len(colors) < num_colors:
            colors = colors * (num_colors // len(colors) + 1)
        return colors[:num_colors]

    # Otherwise use a predefined colormap
    cmap_name = "viridis"
    if theme and "colormap" in theme:
        cmap_name = theme["colormap"]

    # Generate colors from colormap
    cmap = plt.cm.get_cmap(cmap_name, num_colors)
    return [mcolors.rgb2hex(cmap(i)) for i in range(num_colors)]


def plot_attention_maps(
    attention_data: Dict[str, np.ndarray],
    output_path: Union[str, Path],
    class_names: Optional[List[str]] = None,
    max_samples: int = 10,
    theme: Optional[Dict] = None,
) -> None:
    """
    Visualize attention maps from a model.

    Args:
        attention_data: Dictionary containing attention maps and metadata
        output_path: Path to save the visualizations
        class_names: List of class names
        max_samples: Maximum number of samples to visualize
        theme: Visualization theme configuration
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    # Apply theme
    apply_theme(theme)

    # Create output directory
    output_dir = Path(output_path).parent
    ensure_dir(output_dir)

    # Extract data - check if we have the expected keys
    if not all(
        k in attention_data for k in ["attention_maps", "true_labels", "pred_labels"]
    ):
        logger.error(
            f"Missing required keys in attention_data: {attention_data.keys()}"
        )
        return

    attention_maps = attention_data["attention_maps"]
    true_labels = attention_data["true_labels"]
    pred_labels = attention_data["pred_labels"]
    confidences = attention_data.get("confidences", np.ones_like(true_labels))

    # Get original images if available
    original_images = attention_data.get("original_images")

    # Determine number of samples to visualize
    num_samples = min(max_samples, len(attention_maps))

    # Create class name mapping if available
    label_names = {}
    if class_names:
        for i in range(len(class_names)):
            label_names[i] = class_names[i]

    # Create subplots for each sample
    for i in range(num_samples):
        try:
            attention_map = attention_maps[i]
            true_label = true_labels[i]
            pred_label = pred_labels[i]
            confidence = confidences[i] if i < len(confidences) else 1.0

            # Get class names
            true_name = label_names.get(true_label, f"Class {true_label}")
            pred_name = label_names.get(pred_label, f"Class {pred_label}")

            # Determine if prediction was correct
            is_correct = true_label == pred_label
            title_color = "green" if is_correct else "red"

            # Create figure with two subplots side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            # Reshape attention map if needed
            if attention_map.ndim > 2:
                # For multi-channel attention maps, take average across channels
                if attention_map.shape[0] in (1, 3):  # Channel-first format
                    attention_map = np.mean(attention_map, axis=0)
                else:
                    attention_map = np.mean(attention_map, axis=-1)

            # Normalize attention map for visualization
            norm = Normalize(vmin=np.min(attention_map), vmax=np.max(attention_map))

            # Plot heatmap
            im = ax1.imshow(attention_map, cmap="viridis")
            ax1.set_title("Attention Map")
            ax1.axis("off")
            fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

            # Plot 3D surface

            ax2 = fig.add_subplot(1, 2, 2, projection="3d")

            # Create meshgrid for 3D plot
            h, w = attention_map.shape
            y, x = np.mgrid[0:h, 0:w]

            # Plot the surface
            surf = ax2.plot_surface(
                x,
                y,
                attention_map,
                cmap="viridis",
                linewidth=0,
                antialiased=True,
                rcount=100,
                ccount=100,
            )

            ax2.set_title("3D Attention Surface")

            # Add colorbar
            fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=5)

            # Set main title
            plt.suptitle(
                f"Sample {i + 1}: True={true_name}, Pred={pred_name}, Conf={confidence:.2f}",
                color=title_color,
                fontsize=14,
            )

            plt.tight_layout()

            # Save the figure
            sample_path = Path(output_path).parent / f"attention_map_{i + 1}.png"
            plt.savefig(sample_path, dpi=300, bbox_inches="tight")
            plt.close()
        except Exception as e:
            logger.error(f"Error generating attention map for sample {i}: {e}")

    # Create a combined visualization of all attention maps
    try:
        fig = plt.figure(figsize=(15, 10))

        # Calculate grid dimensions
        cols = min(5, num_samples)
        rows = (num_samples + cols - 1) // cols

        for i in range(num_samples):
            try:
                attention_map = attention_maps[i]
                true_label = true_labels[i]
                pred_label = pred_labels[i]

                # Reshape attention map if needed
                if attention_map.ndim > 2:
                    # For multi-channel attention maps, take average across channels
                    if attention_map.shape[0] in (1, 3):  # Channel-first format
                        attention_map = np.mean(attention_map, axis=0)
                    else:
                        attention_map = np.mean(attention_map, axis=-1)

                # Get class names
                true_name = label_names.get(true_label, f"Class {true_label}")
                pred_name = label_names.get(pred_label, f"Class {pred_label}")

                # Create subplot
                ax = fig.add_subplot(rows, cols, i + 1)

                # Plot heatmap
                im = ax.imshow(attention_map, cmap="viridis")

                # Add border color based on correctness
                is_correct = true_label == pred_label
                border_color = "green" if is_correct else "red"
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color(border_color)
                    spine.set_linewidth(2)

                # Add minimal title
                ax.set_title(f"T:{true_name}\nP:{pred_name}", fontsize=8)
                ax.axis("off")
            except Exception as e:
                logger.error(f"Error in attention map overview for sample {i}: {e}")
                # Create an empty subplot with error message
                ax = fig.add_subplot(rows, cols, i + 1)
                ax.text(0.5, 0.5, "Error", ha="center", va="center")
                ax.axis("off")

        plt.suptitle("Attention Map Overview", fontsize=16)
        plt.tight_layout()

        # Save the overview figure
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved attention map visualizations to {output_dir}")
    except Exception as e:
        logger.error(f"Error generating attention map overview: {e}")


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist and return Path object."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate_plots(
    experiment_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
) -> None:
    """
    Generate all plots for a training report.

    Args:
        experiment_dir: Path to experiment directory
        output_dir: Directory to save plots (default: experiment_dir/reports/plots)
    """
    experiment_dir = Path(experiment_dir)

    if not experiment_dir.exists():
        logger.error(f"Experiment directory {experiment_dir} does not exist")
        return

    if output_dir is None:
        output_dir = experiment_dir / "reports" / "plots"

    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    logger.info(f"Generating plots for experiment in {experiment_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Load data
    history = load_history(experiment_dir)
    metrics = load_metrics(experiment_dir)
    cm = load_confusion_matrix(experiment_dir)
    class_names = load_class_names(experiment_dir)

    # Define paths for all evaluation artifacts
    # First check direct evaluation_artifacts directory
    direct_evaluation_artifacts_dir = experiment_dir / "evaluation_artifacts"
    # Also check in metrics subdirectory
    metrics_evaluation_artifacts_dir = (
        experiment_dir / "metrics" / "evaluation_artifacts"
    )

    # Use the first directory that exists, or default to direct path
    evaluation_artifacts_dir = direct_evaluation_artifacts_dir
    if (
        not direct_evaluation_artifacts_dir.exists()
        and metrics_evaluation_artifacts_dir.exists()
    ):
        logger.info("Using evaluation artifacts from metrics subdirectory")
        evaluation_artifacts_dir = metrics_evaluation_artifacts_dir

    logger.info(f"Using evaluation artifacts directory: {evaluation_artifacts_dir}")

    # Define a function to find artifacts in multiple possible locations
    def find_artifact(filename):
        # Check in order: direct path, metrics path, in both evaluation_artifacts subdirs
        possible_paths = [
            direct_evaluation_artifacts_dir / filename,
            metrics_evaluation_artifacts_dir / filename,
            experiment_dir / filename,
        ]

        for path in possible_paths:
            if path.exists():
                logger.info(f"Found artifact {filename} at {path}")
                return path

        # Return default path even if it doesn't exist
        return evaluation_artifacts_dir / filename

    # Basic evaluation artifacts
    predictions_path = find_artifact("predictions.npy")
    features_path = find_artifact("features.npy")
    true_labels_path = find_artifact("true_labels.npy")
    scores_path = find_artifact("scores.npy")
    test_images_path = find_artifact("test_images.npy")

    # New evaluation artifacts
    misclassified_indices_path = find_artifact("misclassified_indices.npy")
    per_class_metrics_path = find_artifact("per_class_metrics.npy")
    calibration_path = find_artifact("calibration_data.npy")
    embeddings_path = find_artifact("embeddings.npy")

    # Other visualization directories
    augmentation_examples_path = experiment_dir / "reports" / "plots" / "augmentation"
    attention_visualizations_path = experiment_dir / "reports" / "plots" / "attention"

    # SHAP directories - Fix paths to be consistent with other visualizations
    shap_dir = (
        experiment_dir / "reports" / "plots" / "shap_analysis"
    )  # Where model-generated SHAP files are stored
    shap_output_dir = output_dir / "shap_analysis"  # Where we copy files for the report

    # Ensure the SHAP directories exist
    ensure_dir(shap_dir)
    ensure_dir(shap_output_dir)

    # Check which artifacts exist
    has_predictions = predictions_path.exists()
    has_features = features_path.exists()
    has_true_labels = true_labels_path.exists()
    has_scores = scores_path.exists()
    has_test_images = test_images_path.exists()
    has_misclassified_indices = misclassified_indices_path.exists()
    has_per_class_metrics = per_class_metrics_path.exists()
    has_calibration = calibration_path.exists()
    has_embeddings = embeddings_path.exists()
    has_augmentation_examples = (
        augmentation_examples_path.exists() and augmentation_examples_path.is_dir()
    )
    has_attention_visualizations = (
        attention_visualizations_path.exists()
        and attention_visualizations_path.is_dir()
    )
    has_shap_visualizations = shap_dir.exists() and shap_dir.is_dir()

    # Log which artifacts were found
    logger.info("Found the following artifacts:")
    for name, found in {
        "predictions": has_predictions,
        "features": has_features,
        "true_labels": has_true_labels,
        "scores": has_scores,
        "test_images": has_test_images,
        "misclassified_indices": has_misclassified_indices,
        "per_class_metrics": has_per_class_metrics,
        "calibration_data": has_calibration,
        "embeddings": has_embeddings,
    }.items():
        logger.info(f"  - {name}: {'✓' if found else '✗'}")

    # Load artifacts that exist
    predictions = np.load(predictions_path) if has_predictions else None
    features = np.load(features_path) if has_features else None
    true_labels = np.load(true_labels_path) if has_true_labels else None
    scores = np.load(scores_path) if has_scores else None
    test_images = np.load(test_images_path) if has_test_images else None
    misclassified_indices = (
        np.load(misclassified_indices_path) if has_misclassified_indices else None
    )
    per_class_metrics = (
        np.load(per_class_metrics_path, allow_pickle=True).item()
        if has_per_class_metrics
        else None
    )
    calibration_data = (
        np.load(calibration_path, allow_pickle=True).item() if has_calibration else None
    )
    embedding_data = (
        np.load(embeddings_path, allow_pickle=True).item() if has_embeddings else None
    )

    # Load theme from config
    theme = load_visualization_theme(experiment_dir)

    # Copy SHAP visualizations if available
    if has_shap_visualizations:
        ensure_dir(shap_output_dir)
        import shutil

        # Only copy relevant SHAP visualizations
        real_shap_patterns = [
            "shap_summary.png",
            "shap_example_*.png",
            "shap_class_*.png",
            "shap_feature_*.png",
            "shap_explanation_*.png",
        ]

        shap_plots = []
        for pattern in real_shap_patterns:
            for shap_file in shap_dir.glob(pattern):
                shutil.copy2(shap_file, shap_output_dir / shap_file.name)
                shap_plots.append(
                    str((shap_output_dir / shap_file.name).relative_to(output_dir))
                )
                logger.info(f"Copied SHAP visualization: {shap_file.name}")

        if shap_plots:
            logger.info(f"Found {len(shap_plots)} SHAP visualizations")
        else:
            logger.info("No matching SHAP visualizations found")

    # Generate basic plots
    try:
        # Always generate basic plots
        if history:
            # Plot training history
            plot_training_history_from_file(
                history=history,
                output_path=output_dir / "training_history.png",
                theme=theme,
            )

            # Plot training time if available
            if "epoch_times" in history:
                plot_training_time_from_history(
                    history=history,
                    output_path=output_dir / "training_time.png",
                    theme=theme,
                )

            # Plot learning rate if available
            if "learning_rate" in history:
                plot_learning_rate_from_history(
                    history=history,
                    output_path=output_dir / "learning_rate.png",
                    theme=theme,
                )

        # Plot confusion matrix if available
        if cm is not None and len(class_names) > 0:
            plot_confusion_matrix_from_file(
                cm=cm,
                class_names=class_names,
                output_path=output_dir / "confusion_matrix.png",
                theme=theme,
            )

        # Plot class metrics if available
        if metrics and "class_metrics" in metrics and len(class_names) > 0:
            plot_class_metrics_from_metrics(
                metrics=metrics,
                class_names=class_names,
                output_path=output_dir / "class_metrics.png",
                theme=theme,
            )

            # Generate performance by class bar chart
            plot_class_performance(
                metrics=metrics["class_metrics"],
                class_names=class_names,
                output_path=output_dir / "class_performance_bars.png",
                theme=theme,
            )

        # Enhanced visualizations if library is available
        if ENHANCED_VISUALIZATION_AVAILABLE:
            logger.info("Using enhanced visualization")

            # ROC curves
            if has_true_labels and has_scores and len(class_names) > 0:
                plot_roc_curve(
                    y_true=true_labels,
                    y_scores=scores,
                    class_names=class_names,
                    output_path=output_dir / "roc_curves.png",
                    theme=theme,
                )

            # Precision-Recall curves
            if has_true_labels and has_scores and len(class_names) > 0:
                plot_precision_recall_curve(
                    y_true=true_labels,
                    y_scores=scores,
                    class_names=class_names,
                    output_path=output_dir / "precision_recall_curves.png",
                    theme=theme,
                )

            # Confidence distribution
            if has_scores and has_true_labels:
                # Calculate confidence metrics
                confidences = np.max(scores, axis=1) if scores.ndim > 1 else scores

                # Determine which predictions were correct
                if true_labels.ndim > 1:  # One-hot encoded
                    pred_classes = np.argmax(scores, axis=1)
                    true_classes = np.argmax(true_labels, axis=1)
                    correctness = pred_classes == true_classes
                else:
                    pred_classes = (
                        np.argmax(scores, axis=1)
                        if scores.ndim > 1
                        else np.round(scores).astype(int)
                    )
                    correctness = pred_classes == true_labels

                # Plot confidence distributions
                plot_confidence_distribution(
                    confidences=confidences,
                    correctness=correctness,
                    output_path=output_dir / "confidence_distribution.png",
                    theme=theme,
                )

                # Plot additional visualization plots
                try:
                    # Confidence histogram
                    plot_histogram(
                        data=confidences,
                        title="Prediction Confidence Histogram",
                        xlabel="Confidence",
                        ylabel="Count",
                        output_path=output_dir / "confidence_histogram.png",
                        theme=theme,
                    )
                    logger.info("Generated confidence histogram")

                    # Prediction distribution
                    # Fix parameter names for plot_categorical
                    plot_categorical(
                        data=pred_classes,  # Change values to data
                        labels=class_names,
                        title="Prediction Distribution",
                        output_path=output_dir / "prediction_distribution.png",
                        theme=theme,
                    )
                    logger.info("Generated prediction distribution")

                    # Classification examples grid
                    if has_test_images:
                        n_examples = min(len(test_images), 20)
                        examples_indices = np.random.choice(
                            len(test_images), n_examples, replace=False
                        )

                        create_classification_examples_grid(
                            images=test_images[examples_indices],
                            true_labels=(
                                true_labels[examples_indices]
                                if len(true_labels) >= len(test_images)
                                else None
                            ),
                            pred_labels=(
                                pred_classes[examples_indices]
                                if len(pred_classes) >= len(test_images)
                                else None
                            ),
                            class_names=class_names,
                            output_path=output_dir / "classification_examples.png",
                            theme=theme,
                        )
                        logger.info("Generated classification examples grid")

                    # Augmentation visualization
                    if has_augmentation_examples and has_test_images:
                        visualize_augmentations(
                            original_image=test_images[0],
                            output_path=output_dir / "augmentation.png",
                            theme=theme,
                        )
                        logger.info("Generated augmentation visualization")

                except Exception as e:
                    logger.error(f"Error generating additional visualizations: {e}")
    except Exception as e:
        logger.error(f"Error generating basic plots: {e}")
        logger.exception("Detailed traceback:")

    # Plot misclassified examples if available
    if (
        has_misclassified_indices
        and has_predictions
        and has_test_images
        and has_true_labels
    ):
        try:
            # Check for size mismatch between test_images and true_labels
            if len(test_images) != len(true_labels):
                logger.warning(
                    f"Size mismatch: test_images ({len(test_images)}) vs true_labels ({len(true_labels)}). Using only the available images."
                )

                # Check if we have a subset of labels saved
                subset_labels_path = evaluation_artifacts_dir / "subset_true_labels.npy"
                if subset_labels_path.exists():
                    subset_labels = np.load(subset_labels_path)
                    logger.info(
                        f"Using subset_true_labels ({len(subset_labels)}) for visualization"
                    )
                    true_labels = subset_labels
                else:
                    # Create a subset on the fly if needed
                    subset_size = min(len(test_images), len(true_labels))
                    true_labels = true_labels[:subset_size]
                    logger.info(
                        f"Created subset of {subset_size} labels to match test_images"
                    )

                # Also limit predictions if needed
                if len(predictions) > len(true_labels):
                    predictions = predictions[: len(true_labels)]

                # Adjust misclassified indices to only include valid indices
                misclassified_indices = misclassified_indices[
                    misclassified_indices < len(true_labels)
                ]

            # Now plot with matching sizes
            plot_top_misclassifications(
                images=test_images,
                true_labels=true_labels[: len(test_images)],  # Ensure sizes match
                predictions=predictions[: len(test_images)],  # Ensure sizes match
                class_names=class_names,
                misclassified_indices=misclassified_indices,
                output_path=output_dir / "top_misclassifications.png",
                theme=theme,
            )
        except Exception as e:
            logger.error(f"Error generating misclassification plot: {e}")

    # Plot class difficulty map if per-class metrics are available
    if has_per_class_metrics:
        try:
            plot_class_difficulties(
                per_class_metrics=per_class_metrics,
                output_path=output_dir / "class_difficulty_map.png",
                theme=theme,
            )
        except Exception as e:
            logger.error(f"Error generating class difficulty plot: {e}")

    # Plot calibration curves if available
    if has_calibration:
        try:
            plot_calibration_curves(
                calibration_data=calibration_data,
                output_path=output_dir / "calibration_curves.png",
                theme=theme,
            )
        except Exception as e:
            logger.error(f"Error generating calibration curves: {e}")

    # Plot feature space if features and labels are available
    if (
        has_features
        and has_true_labels
        and features is not None
        and true_labels is not None
    ):
        try:
            feature_space_path = output_dir / "feature_space.png"

            # Extract labels - if true_labels is one-hot encoded, convert to indices
            if len(true_labels.shape) > 1 and true_labels.shape[1] > 1:
                labels = np.argmax(true_labels, axis=1)
            else:
                labels = true_labels.flatten()

            # Use either t-SNE or UMAP depending on what's available
            if has_embeddings and embedding_data is not None:
                # We already have embeddings generated during evaluation
                if embedding_data.get("method") == "umap":
                    logger.info("Using pre-computed UMAP embeddings")
                    plot_umap_visualization(
                        features=embedding_data["embeddings"],
                        labels=embedding_data["labels"],
                        class_names=class_names,
                        output_path=feature_space_path,
                        theme=theme,
                    )
                else:
                    logger.info("Using pre-computed t-SNE embeddings")
                    plot_tsne_visualization(
                        features=embedding_data["embeddings"],
                        labels=embedding_data["labels"],
                        class_names=class_names,
                        output_path=feature_space_path,
                        theme=theme,
                    )
            else:
                # No pre-computed embeddings, use plot_feature_space
                logger.info("Computing feature space visualization from raw features")
                # Fix the feature_space plot parameters (remove title if causing issues)
                plot_feature_space(
                    features=features,
                    labels=labels,
                    class_names=class_names,
                    output_path=feature_space_path,
                    theme=theme,
                    # title parameter removed as it might be causing the error
                )
            logger.info(
                f"Generated feature space visualization at {feature_space_path}"
            )
        except Exception as e:
            logger.error(f"Error generating feature space plot: {e}")

    # Create augmentation grid if samples are available in the augmentation directory
    augmentation_dir = output_dir / "augmentation"
    ensure_dir(augmentation_dir)

    try:
        # Try to load an example image from test_images
        if has_test_images and test_images is not None and len(test_images) > 0:
            original_image = test_images[0]

            # Create some basic augmentations for visualization
            import torchvision.transforms as T

            # Define augmentations with error handling for each type
            augmentations = [
                ("Horizontal Flip", T.RandomHorizontalFlip(p=1.0)),
                ("Vertical Flip", T.RandomVerticalFlip(p=1.0)),
                ("Rotate 30°", T.RandomRotation(degrees=30)),
                # Make sure Color Jitter only applies to 3-channel images
                (
                    "Color Jitter",
                    lambda x: T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)(x) 
                              if x.shape[0] == 3 else x
                ),
                # Grayscale works with most image formats
                ("Grayscale", T.Grayscale(num_output_channels=3)),
                # Make sure GaussianBlur only applies to valid image sizes
                (
                    "Blur", 
                    lambda x: T.GaussianBlur(kernel_size=5)(x) 
                              if min(x.shape[1:]) >= 5 else x
                ),
            ]

            # Generate augmented versions
            augmented_images = []
            augmentation_names = []

            import torch

            for name, aug in augmentations:
                # Convert numpy to tensor if needed
                if isinstance(original_image, np.ndarray):
                    # Normalize image to 0-1 range if needed
                    if original_image.max() > 1.0:
                        img_tensor = torch.from_numpy(
                            original_image.astype(np.float32) / 255.0
                        ).permute(2, 0, 1)
                    else:
                        img_tensor = torch.from_numpy(
                            original_image.astype(np.float32)
                        ).permute(2, 0, 1)
                else:
                    img_tensor = original_image

                # Apply augmentation
                try:
                    aug_tensor = aug(img_tensor)
                    # Convert back to numpy for visualization
                    aug_image = aug_tensor.permute(1, 2, 0).numpy()

                    augmented_images.append(aug_image)
                    augmentation_names.append(name)
                except Exception as e:
                    logger.warning(f"Error applying augmentation {name}: {e}")

            # Create the grid if we have any successful augmentations
            if augmented_images:
                aug_grid_path = augmentation_dir / "augmentation_grid.png"
                # Ensure original_image has the right format for visualization (HWC)
                if isinstance(original_image, np.ndarray):
                    # If original_image is already a numpy array, check the format
                    if len(original_image.shape) == 3 and original_image.shape[0] == 3:
                        # Convert from CHW format to HWC
                        viz_original = np.transpose(original_image, (1, 2, 0))
                    else:
                        # Already in HWC format
                        viz_original = original_image
                else:
                    # Convert tensor to numpy in HWC format
                    viz_original = original_image.permute(1, 2, 0).numpy()
                
                create_augmentation_grid(
                    original_image=viz_original,
                    augmented_images=augmented_images,
                    augmentation_names=augmentation_names,
                    output_path=aug_grid_path,
                    title="Data Augmentation Examples",
                )
                logger.info(f"Generated augmentation grid at {aug_grid_path}")
    except Exception as e:
        logger.error(f"Error generating augmentation grid: {e}")

    # Check for attention maps and visualize if available
    attention_maps_path = find_artifact("attention_maps.npy")
    if attention_maps_path.exists():
        try:
            # Load attention maps data
            attention_data = np.load(attention_maps_path, allow_pickle=True).item()

            # Create subdirectory for attention visualizations
            attention_dir = output_dir / "attention"
            ensure_dir(attention_dir)

            # Generate visualizations
            plot_attention_maps(
                attention_data=attention_data,
                output_path=attention_dir / "attention_overview.png",
                class_names=class_names,
                max_samples=10,
                theme=theme,
            )
            logger.info(f"Generated attention map visualizations in {attention_dir}")
        except Exception as e:
            logger.error(f"Error generating attention map visualizations: {e}")

    # Apply theme to all plots
    if theme:
        # Use apply_dark_theme if specified in the theme
        if theme.get("dark_mode", True):
            apply_dark_theme()
        else:
            apply_theme(theme)
    else:
        # Default to dark theme
        apply_dark_theme()

    logger.info("Finished generating plots for report")


def generate_plots_for_report(
    experiment_dir: Union[str, Path], output_dir: Optional[Union[str, Path]] = None
) -> None:
    """
    Generate plots for a training report.

    This is a wrapper function for generate_plots that properly handles paths
    and provides a clear API for the CLI.

    Args:
        experiment_dir: Path to experiment directory
        output_dir: Directory to save plots (default: experiment_dir/reports/plots)
    """
    # Resolve experiment directory
    experiment_dir = Path(experiment_dir)
    if not experiment_dir.is_absolute():
        # If path is relative, it's expected to be relative to the current directory
        experiment_dir = Path.cwd() / experiment_dir

    # Make sure the directory exists
    if not experiment_dir.exists():
        logger.error(f"Experiment directory {experiment_dir} does not exist")
        return

    # Generate plots
    generate_plots(
        experiment_dir=experiment_dir,
        output_dir=output_dir,
    )

    logger.info(f"Plots generated for experiment: {experiment_dir}")
    return


def main():
    """
    Main entry point for plot generation.
    """
    parser = argparse.ArgumentParser(description="Generate plots for training report")
    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        required=True,
        help="Name of the experiment or path to experiment directory",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output directory for the plots"
    )

    args = parser.parse_args()

    # Call the wrapper function
    generate_plots_for_report(
        experiment_dir=args.experiment,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()


def normalize_image_for_display(image_array: np.ndarray) -> np.ndarray:
    """
    Convert various image formats to a consistent numpy array format for display.

    Args:
        image_array: Image data in various formats

    Returns:
        Image as a numpy array in HWC format with values in [0, 255] as uint8
    """
    # Check if the input is valid
    if image_array is None or not isinstance(image_array, np.ndarray):
        logger.error(f"Invalid image data type: {type(image_array)}")
        return np.zeros((224, 224, 3), dtype=np.uint8)

    # Check for empty arrays
    if image_array.size == 0:
        logger.error("Empty image array")
        return np.zeros((224, 224, 3), dtype=np.uint8)

    # Handle channel dimensions
    if len(image_array.shape) == 4:
        if image_array.shape[0] == 1:  # Batch dimension of 1
            image_array = image_array[0]
        else:
            logger.warning(
                f"Unexpected batch dimension: {image_array.shape}, taking first image"
            )
            image_array = image_array[0]

    # Convert from CHW to HWC if needed
    if len(image_array.shape) == 3 and image_array.shape[0] in [1, 3, 4]:
        image_array = np.transpose(image_array, (1, 2, 0))

    # Ensure three channels (convert grayscale to RGB)
    if len(image_array.shape) == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    elif len(image_array.shape) == 3 and image_array.shape[2] == 1:
        image_array = np.concatenate([image_array] * 3, axis=2)

    # Handle 4 channels (RGBA) - discard alpha
    if len(image_array.shape) == 3 and image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]

    # Normalize values to 0-255 range
    if image_array.dtype == np.float32 or image_array.dtype == np.float64:
        # Check if normalized between 0 and 1
        if np.max(image_array) <= 1.0:
            image_array = (image_array * 255).astype(np.uint8)
        # Check if normalized with ImageNet stats
        elif np.min(image_array) < 0:
            # Approximately reverse ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])

            # Handle both HWC and CHW formats
            if image_array.shape[2] == 3:  # HWC
                for i in range(3):
                    image_array[:, :, i] = image_array[:, :, i] * std[i] + mean[i]
            else:  # Assume CHW
                for i in range(3):
                    image_array[i] = image_array[i] * std[i] + mean[i]

            image_array = np.clip(image_array, 0, 1)
            image_array = (image_array * 255).astype(np.uint8)
        else:
            # Just clip and convert to uint8
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)

    # Ensure uint8 type for cv2/PIL compatibility
    if image_array.dtype != np.uint8:
        image_array = image_array.astype(np.uint8)

    return image_array


def create_classification_examples_grid(
    images: np.ndarray,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    class_names: List[str],
    output_path: Union[str, Path],
    max_examples: int = 20,
    figsize: Tuple[int, int] = (15, 12),
    theme: Optional[Dict] = None,
) -> None:
    """
    Create a grid of image examples with classification results.

    Args:
        images: Array of images
        true_labels: Ground truth labels
        pred_labels: Predicted labels
        class_names: List of class names
        output_path: Path to save the visualization
        max_examples: Maximum number of examples to show
        figsize: Figure size as (width, height)
        theme: Visualization theme settings
    """
    # Apply theme if provided
    apply_theme(theme)

    # Handle None inputs
    if true_labels is None or pred_labels is None:
        logger.warning("Missing labels for classification grid, skipping")
        return

    # Convert labels to indices if one-hot encoded
    if true_labels.ndim > 1:
        true_indices = np.argmax(true_labels, axis=1)
    else:
        true_indices = true_labels

    if pred_labels.ndim > 1:
        pred_indices = np.argmax(pred_labels, axis=1)
    else:
        pred_indices = pred_labels

    # Check correctness of predictions
    correctness = pred_indices == true_indices

    # Determine samples to display
    # For better visualization, include both correct and incorrect examples
    correct_samples = np.where(correctness)[0]
    incorrect_samples = np.where(~correctness)[0]

    # Limit samples based on availability
    n_correct = min(max_examples // 2, len(correct_samples))
    n_incorrect = min(max_examples // 2, len(incorrect_samples))

    # Random selection
    if n_correct > 0:
        correct_selection = np.random.choice(correct_samples, n_correct, replace=False)
    else:
        correct_selection = np.array([], dtype=int)

    if n_incorrect > 0:
        incorrect_selection = np.random.choice(
            incorrect_samples, n_incorrect, replace=False
        )
    else:
        incorrect_selection = np.array([], dtype=int)

    # Combine samples
    selected_indices = np.concatenate([correct_selection, incorrect_selection])
    np.random.shuffle(selected_indices)  # Mix correct and incorrect

    # Limit to max examples
    n_samples = min(len(selected_indices), max_examples)
    selected_indices = selected_indices[:n_samples]

    # Create grid layout
    cols = min(5, n_samples)
    rows = (n_samples + cols - 1) // cols

    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Plot each example
    for i, idx in enumerate(selected_indices):
        if i >= len(axes):
            break

        try:
            # Get data
            image = images[idx]
            true_idx = true_indices[idx]
            pred_idx = pred_indices[idx]

            # Normalize image to consistent format
            norm_image = normalize_image_for_display(image)

            # Get class names
            true_name = (
                class_names[true_idx]
                if true_idx < len(class_names)
                else f"Class {true_idx}"
            )
            pred_name = (
                class_names[pred_idx]
                if pred_idx < len(class_names)
                else f"Class {pred_idx}"
            )

            # Truncate class names if too long
            if len(true_name) > 15:
                true_name = true_name[:12] + "..."
            if len(pred_name) > 15:
                pred_name = pred_name[:12] + "..."

            # Plot image
            axes[i].imshow(norm_image)

            # Add colored border based on correctness
            is_correct = true_idx == pred_idx
            border_color = "green" if is_correct else "red"

            for spine in axes[i].spines.values():
                spine.set_visible(True)
                spine.set_color(border_color)
                spine.set_linewidth(2)

            # Add title with labels
            axes[i].set_title(
                f"True: {true_name}\nPred: {pred_name}",
                fontsize=8,
                color="white" if is_correct else "red",
            )

            # Remove ticks
            axes[i].set_xticks([])
            axes[i].set_yticks([])

        except Exception as e:
            logger.error(f"Error creating example grid for index {idx}: {e}")
            axes[i].text(
                0.5,
                0.5,
                "Error",
                horizontalalignment="center",
                verticalalignment="center",
            )
            axes[i].axis("off")

    # Hide unused subplots
    for i in range(n_samples, len(axes)):
        axes[i].axis("off")

    # Add title
    plt.suptitle("Classification Examples", fontsize=16)

    # Create legend for correct/incorrect
    legend_elements = [
        plt.Line2D([0], [0], color="green", lw=4, label="Correct"),
        plt.Line2D([0], [0], color="red", lw=4, label="Incorrect"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2)

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Make room for legend
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Created classification examples grid with {n_samples} images")
