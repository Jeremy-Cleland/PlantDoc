"""
Generate plots for training reports.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

from utils.logger import get_logger
from utils.metrics import plot_confusion_matrix, plot_metrics_history
from utils.paths import ensure_dir
from utils.visualization import (
    plot_class_metrics,
    plot_confusion_matrix,
    plot_learning_rate,
    plot_training_history,
    plot_training_time,
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
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON from {file_path}: {e}")
        return {}


def load_history(experiment_dir: Union[str, Path]) -> Dict[str, List[float]]:
    """
    Load training history from an experiment directory.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        Dictionary of training history
    """
    history_path = Path(experiment_dir) / "history.json"

    if not history_path.exists():
        # Try metrics_logger.jsonl or other possibilities
        logger_path = Path(experiment_dir) / "metrics_logger.jsonl"
        if logger_path.exists():
            try:
                # Load JSONL file (one JSON object per line)
                with open(logger_path, "r") as f:
                    lines = f.readlines()

                history = {}
                for line in lines:
                    try:
                        entry = json.loads(line.strip())
                        for key, value in entry.items():
                            if isinstance(value, (int, float)):
                                if key not in history:
                                    history[key] = []
                                history[key].append(value)
                    except:
                        pass

                return history
            except Exception as e:
                logger.error(f"Failed to load metrics logger from {logger_path}: {e}")
                return {}
        else:
            logger.warning(f"No history file found in {experiment_dir}")
            return {}

    return load_json(history_path)


def load_metrics(experiment_dir: Union[str, Path]) -> Dict:
    """
    Load metrics from an experiment directory.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        Dictionary of metrics
    """
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
        with open(class_names_path, "r") as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Failed to load class names from {class_names_path}: {e}")
        return []


def plot_training_history_from_file(
    history: Dict[str, List[float]],
    output_path: Union[str, Path],
) -> None:
    """
    Plot training history from a history dictionary.

    Args:
        history: Dictionary of training history
        output_path: Path to save the plot
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

    # Plot training history
    plot_training_history(
        history=scalar_history,
        output_path=output_path,
        metrics=metrics_to_plot,
    )

    logger.info(f"Saved training history plot to {output_path}")


def plot_confusion_matrix_from_file(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Union[str, Path],
) -> None:
    """
    Plot confusion matrix from a numpy array.

    Args:
        cm: Confusion matrix
        class_names: List of class names
        output_path: Path to save the plot
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    # Plot confusion matrix
    plot_confusion_matrix(
        cm=cm,
        class_names=class_names,
        output_path=output_path,
        normalize=True,
    )

    logger.info(f"Saved confusion matrix plot to {output_path}")


def plot_training_time_from_history(
    history: Dict[str, List[float]],
    output_path: Union[str, Path],
) -> None:
    """
    Plot training time from history.

    Args:
        history: Dictionary of training history
        output_path: Path to save the plot
    """
    # Check if we have time per epoch
    time_key = "time" if "time" in history else "train_time"

    if time_key not in history:
        logger.warning("No training time data found in history")
        return

    times = history[time_key]

    # Plot training time
    plot_training_time(
        epoch_times=times,
        output_path=output_path,
    )

    logger.info(f"Saved training time plot to {output_path}")


def plot_class_metrics_from_metrics(
    metrics: Dict,
    class_names: List[str],
    output_path: Union[str, Path],
) -> None:
    """
    Plot class metrics from a metrics dictionary.

    Args:
        metrics: Dictionary of metrics
        class_names: List of class names
        output_path: Path to save the plot
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

    # Plot class metrics
    plot_class_metrics(
        metrics=metrics,
        class_names=class_names,
        output_path=output_path,
    )

    logger.info(f"Saved class metrics plot to {output_path}")


def plot_learning_rate_from_history(
    history: Dict[str, List[float]],
    output_path: Union[str, Path],
) -> None:
    """
    Plot learning rate from history.

    Args:
        history: Dictionary of training history
        output_path: Path to save the plot
    """
    # Check if we have learning rate data
    lr_key = "lr" if "lr" in history else "learning_rate"

    if lr_key not in history:
        logger.warning("No learning rate data found in history")
        return

    lr_history = history[lr_key]

    # Plot learning rate
    plot_learning_rate(
        lr_history=lr_history,
        output_path=output_path,
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

    # Plot training history
    if history:
        plot_training_history_from_file(
            history=history,
            output_path=output_dir / "training_history.png",
        )

    # Plot confusion matrix
    if cm is not None and class_names:
        plot_confusion_matrix_from_file(
            cm=cm,
            class_names=class_names,
            output_path=output_dir / "confusion_matrix.png",
        )

    # Plot training time
    if history and ("time" in history or "train_time" in history):
        plot_training_time_from_history(
            history=history,
            output_path=output_dir / "training_time.png",
        )

    # Plot class metrics
    if metrics and class_names:
        plot_class_metrics_from_metrics(
            metrics=metrics,
            class_names=class_names,
            output_path=output_dir / "class_metrics.png",
        )

    # Plot learning rate
    if history and ("lr" in history or "learning_rate" in history):
        plot_learning_rate_from_history(
            history=history,
            output_path=output_dir / "learning_rate.png",
        )

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
        # Try to find in outputs directory
        outputs_dir = Path(__file__).parents[2] / "outputs"
        experiment_dir = outputs_dir / experiment_dir

    # Generate plots
    generate_plots_for_report(
        experiment_dir=experiment_dir,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
