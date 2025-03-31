# PATH: plantdl/core/evaluation/metrics.py

"""
Functions for computing various classification metrics.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


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
