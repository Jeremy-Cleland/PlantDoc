# PATH: plantdl/core/evaluation/metrics.py

"""
Functions for computing various classification metrics and metric tracking utilities.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

from utils.logger import get_logger

logger = get_logger(__name__)


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate classification accuracy.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Accuracy score
    """
    return (y_true == y_pred).mean()


def calculate_precision_recall_f1(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = "weighted"
) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 score.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: Averaging method ('micro', 'macro', 'weighted', 'samples', None)

    Returns:
        Dictionary with precision, recall, and F1 scores
    """
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

    return {"precision": precision, "recall": recall, "f1": f1}


def calculate_per_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Calculate precision, recall, and F1 score for each class.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names

    Returns:
        Dictionary with per-class metrics
    """
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Use default class names if none provided
    if class_names is None:
        num_classes = len(precision)
        class_names = [f"Class {i}" for i in range(num_classes)]

    # Create dictionary of per-class metrics
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            "precision": precision[i],
            "recall": recall[i],
            "f1": f1[i],
        }

    return per_class_metrics


def calculate_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, normalize: Optional[str] = None
) -> np.ndarray:
    """
    Calculate confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        normalize: Normalization method ('true', 'pred', 'all', None)

    Returns:
        Confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)

    # Normalize confusion matrix if requested
    if normalize == "true":
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    elif normalize == "pred":
        cm = cm.astype("float") / cm.sum(axis=0)[np.newaxis, :]
    elif normalize == "all":
        cm = cm.astype("float") / cm.sum()

    return cm


def calculate_all_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Calculate all classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names

    Returns:
        Dictionary with all metrics
    """
    # Calculate overall metrics
    accuracy = calculate_accuracy(y_true, y_pred)
    overall_metrics = calculate_precision_recall_f1(y_true, y_pred, average="weighted")

    # Calculate per-class metrics
    per_class_metrics = calculate_per_class_metrics(y_true, y_pred, class_names)

    # Calculate confusion matrix
    cm = calculate_confusion_matrix(y_true, y_pred)

    # Combine all metrics
    all_metrics = {
        "accuracy": accuracy,
        "overall": overall_metrics,
        "per_class": per_class_metrics,
        "confusion_matrix": cm,
    }

    return all_metrics


class IncrementalMetricsCalculator:
    """
    Class for incrementally calculating classification metrics.
    This avoids storing all predictions and targets in memory at once.

    Args:
        num_classes: Number of classes in the dataset
        class_names: Optional list of class names
    """

    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class {i}" for i in range(num_classes)]
        self.reset()

    def reset(self):
        """
        Reset all accumulators for a new calculation.
        """
        # Initialize confusion matrix
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )

        # Initialize counters
        self.total_samples = 0
        self.correct_samples = 0

    def update(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
    ):
        """
        Update metrics with a new batch of predictions.

        Args:
            y_true: Ground truth labels (tensor or numpy array)
            y_pred: Predicted labels (tensor or numpy array)
        """
        # Convert tensors to numpy if needed with explicit data type handling
        if isinstance(y_true, torch.Tensor):
            # Handle mixed precision: ensure we convert to standard float32 for consistency
            if y_true.dtype == torch.float16 or y_true.dtype == torch.bfloat16:
                y_true = y_true.to(torch.float32)
            y_true = y_true.detach().cpu().numpy()

        if isinstance(y_pred, torch.Tensor):
            # Handle mixed precision: ensure we convert to standard float32 for consistency
            if y_pred.dtype == torch.float16 or y_pred.dtype == torch.bfloat16:
                y_pred = y_pred.to(torch.float32)
            y_pred = y_pred.detach().cpu().numpy()

        # Ensure arrays are 1D and have consistent data types
        y_true = y_true.flatten().astype(np.int64)
        y_pred = y_pred.flatten().astype(np.int64)

        # Update total and correct samples
        batch_size = len(y_true)
        self.total_samples += batch_size
        self.correct_samples += np.sum(y_true == y_pred)

        # Update confusion matrix
        for i in range(batch_size):
            self.confusion_matrix[y_true[i], y_pred[i]] += 1

    def compute(self) -> Dict[str, Any]:
        """
        Compute all metrics based on accumulated data.

        Returns:
            Dictionary with all metrics
        """
        # Calculate accuracy
        accuracy = (
            self.correct_samples / self.total_samples if self.total_samples > 0 else 0.0
        )

        # Extract true positives, false positives, false negatives for each class
        tp = np.diag(self.confusion_matrix)  # True positives are on the diagonal

        # Calculate per-class precision, recall, and F1
        precision_per_class = np.zeros(self.num_classes)
        recall_per_class = np.zeros(self.num_classes)
        f1_per_class = np.zeros(self.num_classes)

        for i in range(self.num_classes):
            # Precision: TP / (TP + FP)
            col_sum = np.sum(self.confusion_matrix[:, i])
            precision_per_class[i] = tp[i] / col_sum if col_sum > 0 else 0

            # Recall: TP / (TP + FN)
            row_sum = np.sum(self.confusion_matrix[i, :])
            recall_per_class[i] = tp[i] / row_sum if row_sum > 0 else 0

            # F1: 2 * (precision * recall) / (precision + recall)
            denominator = precision_per_class[i] + recall_per_class[i]
            f1_per_class[i] = (
                2 * precision_per_class[i] * recall_per_class[i] / denominator
                if denominator > 0
                else 0
            )

        # Calculate weighted average metrics
        class_counts = np.sum(self.confusion_matrix, axis=1)
        total_count = np.sum(class_counts)

        if total_count > 0:
            weights = class_counts / total_count
            weighted_precision = np.sum(precision_per_class * weights)
            weighted_recall = np.sum(recall_per_class * weights)
            weighted_f1 = np.sum(f1_per_class * weights)
        else:
            weighted_precision = 0
            weighted_recall = 0
            weighted_f1 = 0

        # Create per-class metrics dictionary
        per_class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            per_class_metrics[class_name] = {
                "precision": precision_per_class[i],
                "recall": recall_per_class[i],
                "f1": f1_per_class[i],
            }

        # Combine all metrics
        return {
            "accuracy": accuracy,
            "overall": {
                "precision": weighted_precision,
                "recall": weighted_recall,
                "f1": weighted_f1,
            },
            "per_class": per_class_metrics,
            "confusion_matrix": self.confusion_matrix.copy(),
        }

    def get_confusion_matrix(self) -> np.ndarray:
        """
        Return the current confusion matrix.

        Returns:
            NumPy array with the confusion matrix
        """
        return self.confusion_matrix.copy()

    def get_classification_report(self) -> Dict[str, Dict[str, float]]:
        """
        Return a classification report similar to sklearn's classification_report.

        Returns:
            Dictionary where keys are class names/indices and values are dictionaries
            with 'precision', 'recall', and 'f1-score' metrics
        """
        report = {}

        # Extract metrics for each class
        for i, class_name in enumerate(self.class_names):
            # Calculate metrics from confusion matrix
            tp = self.confusion_matrix[i, i]
            col_sum = np.sum(self.confusion_matrix[:, i])
            row_sum = np.sum(self.confusion_matrix[i, :])

            # Calculate precision, recall, and F1
            precision = tp / col_sum if col_sum > 0 else 0
            recall = tp / row_sum if row_sum > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            # Add to report using both class name and index as keys
            # Class names can sometimes be more convenient
            report[str(i)] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1-score": float(f1),
                "support": int(row_sum),
            }

            report[class_name] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1-score": float(f1),
                "support": int(row_sum),
            }

        return report


class ClassificationMetrics:
    """
    Calculate and store classification metrics.

    This class handles calculation of various classification metrics like
    accuracy, precision, recall, F1 score, and confusion matrix.

    Args:
        num_classes: Number of classes in the dataset
        class_names: List of class names (if available)
    """

    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
    ):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]

        # Initialize metrics storage
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.all_preds = []
        self.all_targets = []
        self.class_correct = np.zeros(self.num_classes)
        self.class_total = np.zeros(self.num_classes)
        self.confusion_mat = np.zeros((self.num_classes, self.num_classes))
        self.metrics = {}

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with batch predictions.

        Args:
            logits: Model output logits (B, C)
            targets: Ground truth labels (B,)
        """
        preds = torch.argmax(logits, dim=1)
        preds_np = preds.cpu().numpy()
        targets_np = targets.cpu().numpy()

        # Store predictions and targets for global metrics
        self.all_preds.append(preds_np)
        self.all_targets.append(targets_np)

        # Update class-wise accuracy counters
        for i in range(len(targets_np)):
            label = targets_np[i]
            self.class_total[label] += 1
            if preds_np[i] == label:
                self.class_correct[label] += 1

        # Update confusion matrix
        batch_cm = confusion_matrix(
            targets_np, preds_np, labels=range(self.num_classes)
        )
        self.confusion_mat += batch_cm

    def compute(self, prefix: str = "") -> Dict[str, float]:
        """
        Compute all metrics.

        Args:
            prefix: Optional prefix for metric names

        Returns:
            Dictionary of metrics
        """
        if not self.all_preds:
            logger.warning("No predictions to compute metrics from.")
            return {}

        # Concatenate all predictions and targets
        all_preds = np.concatenate(self.all_preds)
        all_targets = np.concatenate(self.all_targets)

        # Calculate global metrics
        accuracy = accuracy_score(all_targets, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average="weighted", zero_division=0
        )

        # Store in metrics dict with optional prefix
        metrics = {
            f"{prefix}accuracy": float(accuracy),
            f"{prefix}precision": float(precision),
            f"{prefix}recall": float(recall),
            f"{prefix}f1": float(f1),
        }

        # Calculate per-class metrics
        class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
            all_targets,
            all_preds,
            average=None,
            labels=range(self.num_classes),
            zero_division=0,
        )

        # Class accuracy
        class_accuracy = np.divide(
            self.class_correct,
            self.class_total,
            out=np.zeros_like(self.class_correct),
            where=self.class_total > 0,
        )

        # Store per-class metrics
        for i in range(self.num_classes):
            class_name = self.class_names[i].replace(" ", "_")
            metrics[f"{prefix}class_{class_name}_accuracy"] = float(class_accuracy[i])
            metrics[f"{prefix}class_{class_name}_precision"] = float(class_precision[i])
            metrics[f"{prefix}class_{class_name}_recall"] = float(class_recall[i])
            metrics[f"{prefix}class_{class_name}_f1"] = float(class_f1[i])

        # Store all metrics
        self.metrics = metrics
        return metrics

    def get_confusion_matrix(self) -> np.ndarray:
        """Get the confusion matrix."""
        return self.confusion_mat

    def get_classification_report(self) -> Dict[str, Dict[str, float]]:
        """Get a detailed classification report."""
        if not self.all_preds:
            logger.warning("No predictions to generate classification report.")
            return {}

        all_preds = np.concatenate(self.all_preds)
        all_targets = np.concatenate(self.all_targets)

        report = classification_report(
            all_targets,
            all_preds,
            labels=range(self.num_classes),
            target_names=self.class_names,
            output_dict=True,
            zero_division=0,
        )

        return report

    def save_metrics(
        self, output_dir: Union[str, Path], filename: str = "metrics.json"
    ):
        """
        Save metrics to JSON file.

        Args:
            output_dir: Directory to save the metrics
            filename: Name of the output file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / filename

        try:
            # Ensure metrics are serializable
            serializable_metrics = {}
            for k, v in self.metrics.items():
                if isinstance(v, (np.float32, np.float64, np.int32, np.int64)):
                    serializable_metrics[k] = float(v)
                else:
                    serializable_metrics[k] = v

            # Add confusion matrix
            if hasattr(self, "confusion_mat") and self.confusion_mat is not None:
                cm_path = output_dir / "evaluation_artifacts" / "confusion_matrix.npy"
                np.save(cm_path, self.confusion_mat)
                logger.info(f"Saved confusion matrix to {cm_path}")

                # Also save a simplified version in the metrics
                serializable_metrics["confusion_matrix"] = self.confusion_mat.tolist()

            with open(output_path, "w") as f:
                json.dump(serializable_metrics, f, indent=2)

            logger.info(f"Saved metrics to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")

    def save_classification_report(
        self, output_dir: Union[str, Path], filename: str = "classification_report.json"
    ):
        """
        Save classification report to JSON file.

        Args:
            output_dir: Directory to save the report
            filename: Name of the output file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / filename

        try:
            report = self.get_classification_report()

            # Convert numpy values to Python types
            def convert_numpy(obj):
                if isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj

            serializable_report = convert_numpy(report)

            with open(output_path, "w") as f:
                json.dump(serializable_report, f, indent=2)

            logger.info(f"Saved classification report to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save classification report: {e}")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 8),
    normalize: bool = True,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
) -> plt.Figure:
    """
    Plot confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix
        class_names: List of class names
        output_path: Path to save the plot
        figsize: Figure size (width, height)
        normalize: Whether to normalize the confusion matrix
        title: Plot title
        cmap: Colormap

    Returns:
        Matplotlib figure
    """
    if normalize:
        # Normalize by row (true labels)
        cm_norm = cm.astype("float") / (
            cm.sum(axis=1)[:, np.newaxis] + 1e-6
        )  # avoid div by zero
        cm_plot = cm_norm
        fmt = ".2f"
    else:
        cm_plot = cm
        fmt = "d"

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    sns.heatmap(
        cm_plot,
        annot=False,
        fmt=fmt,
        cmap=cmap,
        xticklabels=[name[:15] for name in class_names],  # Shorten long names
        yticklabels=[name[:15] for name in class_names],
        ax=ax,
    )

    # Remove ticks but keep labels
    ax.set_xticks(np.arange(len(class_names)) + 0.5)
    ax.set_yticks(np.arange(len(class_names)) + 0.5)

    # Set labels and title
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)

    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_label("Normalized Confusion" if normalize else "Count")

    plt.tight_layout()

    # Save figure if output path provided
    if output_path:
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=400, bbox_inches="tight")
            logger.info(f"Saved confusion matrix plot to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save confusion matrix plot: {e}")

    return fig


def plot_metrics_history(
    metrics_history: Dict[str, List[float]],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 8),
    metrics_to_plot: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Plot training/validation metrics history.

    Args:
        metrics_history: Dictionary of metric lists
        output_path: Path to save the plot
        figsize: Figure size (width, height)
        metrics_to_plot: List of metrics to plot (default: loss, accuracy)

    Returns:
        Matplotlib figure
    """
    # Default metrics to plot
    metrics_to_plot = metrics_to_plot or ["loss", "accuracy"]

    # Filter available metrics
    available_metrics = set()
    for metric in metrics_history:
        # Strip prefix like 'train_' or 'val_'
        base_metric = metric.split("_", 1)[-1] if "_" in metric else metric
        available_metrics.add(base_metric)

    # Only plot metrics that are available
    metrics_to_plot = [m for m in metrics_to_plot if m in available_metrics]

    if not metrics_to_plot:
        logger.warning("No metrics available to plot")
        return None

    # Create figure
    fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=figsize, sharex=True)

    # If only one metric, axes is not a list
    if len(metrics_to_plot) == 1:
        axes = [axes]

    # Plot each metric
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]

        # Find train and val metrics
        train_key = f"train_{metric}" if f"train_{metric}" in metrics_history else None
        val_key = f"val_{metric}" if f"val_{metric}" in metrics_history else None

        # Plot train metric if available
        if train_key and metrics_history[train_key]:
            ax.plot(
                metrics_history[train_key],
                label=f"Train {metric}",
                marker="o",
                markersize=3,
            )

        # Plot val metric if available
        if val_key and metrics_history[val_key]:
            ax.plot(
                metrics_history[val_key],
                label=f"Validation {metric}",
                marker="s",
                markersize=3,
            )

        # Set labels and legend
        ax.set_ylabel(metric.capitalize())
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    # Set x-axis label for the bottom subplot
    axes[-1].set_xlabel("Epoch")

    # Set title
    fig.suptitle("Training Metrics History", fontsize=14)

    plt.tight_layout()

    # Save figure if output path provided
    if output_path:
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=400, bbox_inches="tight")
            logger.info(f"Saved metrics history plot to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save metrics history plot: {e}")

    return fig
