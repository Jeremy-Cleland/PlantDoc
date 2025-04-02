"""
Generate plots for training reports.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from omegaconf import OmegaConf

from core.visualization.base_visualization import (
    DEFAULT_THEME,
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
        plot_distribution,
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
    # First check in metrics directory
    metrics_dir = Path(experiment_dir) / "metrics"
    history_path = metrics_dir / "history.json"

    # Fall back to main directory
    if not history_path.exists():
        history_path = Path(experiment_dir) / "history.json"

    if not history_path.exists():
        logger.warning(f"No history file found in {experiment_dir}")
        return {}

    return load_json(history_path)


def load_metrics(experiment_dir: Union[str, Path]) -> Dict:
    """
    Load evaluation metrics from JSON file.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        Dictionary with metrics data
    """
    # First check in metrics directory
    metrics_dir = Path(experiment_dir) / "metrics"
    metrics_path = metrics_dir / "metrics.json"

    # Fall back to main directory
    if not metrics_path.exists():
        metrics_path = Path(experiment_dir) / "metrics.json"

    if not metrics_path.exists():
        logger.warning(f"No metrics file found in {experiment_dir}")
        return {}

    return load_json(metrics_path)


def load_confusion_matrix(experiment_dir: Union[str, Path]) -> Optional[np.ndarray]:
    """
    Load confusion matrix from an experiment directory.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        Confusion matrix as numpy array, or None if not found
    """
    cm_path = Path(experiment_dir) / "confusion_matrix.npy"

    if not cm_path.exists():
        logger.warning(f"No confusion matrix file found in {experiment_dir}")
        return None

    try:
        return np.load(cm_path)
    except Exception as e:
        logger.error(f"Failed to load confusion matrix from {cm_path}: {e}")
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


def generate_plots_for_report(
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

    # Check if we have additional data for enhanced visualizations
    predictions_path = experiment_dir / "predictions.npy"
    features_path = experiment_dir / "features.npy"
    true_labels_path = experiment_dir / "true_labels.npy"
    scores_path = experiment_dir / "scores.npy"
    test_images_path = experiment_dir / "test_images.npy"
    augmentation_examples_path = experiment_dir / "augmentation_examples"
    attention_visualizations_path = experiment_dir / "attention_visualizations"
    shap_visualizations_path = experiment_dir / "shap_visualizations"

    has_predictions = predictions_path.exists()
    has_features = features_path.exists()
    has_true_labels = true_labels_path.exists()
    has_scores = scores_path.exists()
    has_test_images = test_images_path.exists()
    has_augmentation_examples = (
        augmentation_examples_path.exists() and augmentation_examples_path.is_dir()
    )
    has_attention_visualizations = (
        attention_visualizations_path.exists()
        and attention_visualizations_path.is_dir()
    )
    has_shap_visualizations = (
        shap_visualizations_path.exists() and shap_visualizations_path.is_dir()
    )

    # Check for SHAP visualizations
    shap_plots = []
    shap_dir = experiment_dir / "shap_visualizations"
    shap_output_dir = output_dir / "shap"

    if shap_dir.exists() and any(shap_dir.glob("*.png")):
        # Create output directory for SHAP visualizations
        ensure_dir(shap_output_dir)

        # Only copy files that are actual SHAP visualizations (not regular plots)
        real_shap_patterns = [
            "shap_summary.png",
            "shap_example_*.png",
            "shap_class_*.png",
            "shap_feature_*.png",
            "shap_explanation_*.png",
        ]

        # Copy only SHAP visualization files that match the pattern
        import shutil

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

    # Load these files if they exist
    predictions = np.load(predictions_path) if has_predictions else None
    features = np.load(features_path) if has_features else None
    true_labels = np.load(true_labels_path) if has_true_labels else None
    scores = np.load(scores_path) if has_scores else None
    test_images = np.load(test_images_path) if has_test_images else None

    # Load theme from config
    theme = load_visualization_theme(experiment_dir)

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

        # Prediction confidence analysis if predictions are available
        if has_scores and has_true_labels:
            # Extract max confidence for each prediction
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

            # Plot confidence distribution
            plot_confidence_distribution(
                confidences=confidences,
                correctness=correctness,
                output_path=output_dir / "confidence_distribution.png",
                theme=theme,
            )

            # Plot histogram of prediction confidences
            plot_histogram(
                data=confidences,
                output_path=output_dir / "confidence_histogram.png",
                theme=theme,
                bins=20,
                title="Prediction Confidence Histogram",
                xlabel="Confidence",
                ylabel="Count",
            )

            # Plot distribution of class predictions
            class_counts = np.bincount(pred_classes, minlength=len(class_names))
            plot_categorical(
                data=class_counts,
                categories=class_names,
                output_path=output_dir / "prediction_distribution.png",
                theme=theme,
                title="Prediction Distribution",
                xlabel="Class",
                ylabel="Count",
                kind="bar",  # Use bar chart for discrete classes
            )

        # Classification examples grid if test images are available
        if (
            has_test_images
            and has_predictions
            and has_true_labels
            and len(class_names) > 0
        ):
            try:
                # Convert predictions to class indices if needed
                if predictions.ndim > 1 and predictions.shape[1] > 1:
                    pred_indices = np.argmax(predictions, axis=1)
                else:
                    pred_indices = predictions.astype(int)

                # Create classification examples grid
                create_classification_examples_grid(
                    images=test_images,
                    true_labels=true_labels,
                    pred_labels=predictions,
                    class_names=class_names,
                    output_path=output_dir / "classification_examples.png",
                    theme=theme,
                    max_examples=20,
                    separate_correct_incorrect=True,
                )
                logger.info("Generated classification examples grid")
            except Exception as e:
                logger.error(f"Failed to generate classification examples grid: {e}")

        # Augmentation visualization using the new function
        if has_test_images:
            try:
                # Select a sample image from the test set
                sample_idx = np.random.randint(0, len(test_images))
                original_img = test_images[sample_idx]

                # Create augmentation examples
                augmented_images = []
                augmentation_names = [
                    "Horizontal Flip",
                    "Vertical Flip",
                    "Rotation (30°)",
                    "Brightness +30%",
                    "Contrast +30%",
                    "Crop & Resize",
                    "Blur",
                    "Color Jitter",
                ]

                # Apply some basic augmentations using PIL
                import PIL.Image
                from PIL import ImageEnhance, ImageFilter

                # Convert to PIL image for easier manipulation
                if original_img.dtype == np.float32 or original_img.max() <= 1.0:
                    original_img_pil = PIL.Image.fromarray(
                        (original_img * 255).astype(np.uint8)
                    )
                else:
                    original_img_pil = PIL.Image.fromarray(
                        original_img.astype(np.uint8)
                    )

                # Create augmented versions
                # 1. Horizontal Flip
                h_flip = np.array(original_img_pil.transpose(PIL.Image.FLIP_LEFT_RIGHT))
                augmented_images.append(h_flip)

                # 2. Vertical Flip
                v_flip = np.array(original_img_pil.transpose(PIL.Image.FLIP_TOP_BOTTOM))
                augmented_images.append(v_flip)

                # 3. Rotation
                rotated = np.array(original_img_pil.rotate(30))
                augmented_images.append(rotated)

                # 4. Brightness enhancement
                bright = np.array(
                    ImageEnhance.Brightness(original_img_pil).enhance(1.3)
                )
                augmented_images.append(bright)

                # 5. Contrast enhancement
                contrast = np.array(
                    ImageEnhance.Contrast(original_img_pil).enhance(1.3)
                )
                augmented_images.append(contrast)

                # 6. Random crop and resize
                width, height = original_img_pil.size
                left = width // 4
                top = height // 4
                right = 3 * width // 4
                bottom = 3 * height // 4
                cropped = original_img_pil.crop((left, top, right, bottom))
                resized = np.array(cropped.resize((width, height)))
                augmented_images.append(resized)

                # 7. Blur
                blurred = np.array(
                    original_img_pil.filter(ImageFilter.GaussianBlur(radius=2))
                )
                augmented_images.append(blurred)

                # 8. Color jitter (adjust color balance)
                color = np.array(ImageEnhance.Color(original_img_pil).enhance(1.5))
                augmented_images.append(color)

                # Create grid visualization
                create_augmentation_grid(
                    original_image=original_img,
                    augmented_images=augmented_images,
                    augmentation_names=augmentation_names,
                    output_path=output_dir / "augmentation.png",
                    theme=theme,
                    title="Data Augmentation Examples",
                )
                logger.info("Generated augmentation visualization grid")
            except Exception as e:
                logger.error(f"Failed to generate augmentation visualization: {e}")

    logger.info("Finished generating plots for report")


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

    # Resolve experiment directory
    experiment_dir = Path(args.experiment)
    if not experiment_dir.is_absolute():
        # If path is relative, it's expected to be relative to the current directory
        experiment_dir = Path.cwd() / experiment_dir

    # Make sure the directory exists
    if not experiment_dir.exists():
        logger.error(f"Experiment directory {experiment_dir} does not exist")
        return

    # Generate plots
    generate_plots_for_report(
        experiment_dir=experiment_dir,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
