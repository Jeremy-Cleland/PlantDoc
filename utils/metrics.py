# metrics.py stub
"""
Metrics calculation and tracking utilities for model evaluation.
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
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
            all_targets, all_preds, average=None, labels=range(self.num_classes), zero_division=0
        )
        
        # Class accuracy
        class_accuracy = np.divide(
            self.class_correct, self.class_total, 
            out=np.zeros_like(self.class_correct), 
            where=self.class_total > 0
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
            all_targets, all_preds, 
            labels=range(self.num_classes), 
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        return report
        
    def save_metrics(self, output_dir: Union[str, Path], filename: str = "metrics.json"):
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
                    
            with open(output_path, "w") as f:
                json.dump(serializable_metrics, f, indent=2)
                
            logger.info(f"Saved metrics to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
            
    def save_classification_report(self, output_dir: Union[str, Path], filename: str = "classification_report.json"):
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
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)  # avoid div by zero
        cm_plot = cm_norm
        fmt = '.2f'
    else:
        cm_plot = cm
        fmt = 'd'
        
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        cm_plot, annot=False, fmt=fmt, cmap=cmap,
        xticklabels=[name[:15] for name in class_names],  # Shorten long names
        yticklabels=[name[:15] for name in class_names],
        ax=ax
    )
    
    # Remove ticks but keep labels
    ax.set_xticks(np.arange(len(class_names)) + 0.5)
    ax.set_yticks(np.arange(len(class_names)) + 0.5)
    
    # Set labels and title
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_label('Normalized Confusion' if normalize else 'Count')
    
    plt.tight_layout()
    
    # Save figure if output path provided
    if output_path:
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
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
    metrics_to_plot = metrics_to_plot or ['loss', 'accuracy']
    
    # Filter available metrics
    available_metrics = set()
    for metric in metrics_history:
        # Strip prefix like 'train_' or 'val_'
        base_metric = metric.split('_', 1)[-1] if '_' in metric else metric
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
        train_key = f'train_{metric}' if f'train_{metric}' in metrics_history else None
        val_key = f'val_{metric}' if f'val_{metric}' in metrics_history else None
        
        # Plot train metric if available
        if train_key and metrics_history[train_key]:
            ax.plot(metrics_history[train_key], label=f'Train {metric}', marker='o', markersize=3)
            
        # Plot val metric if available
        if val_key and metrics_history[val_key]:
            ax.plot(metrics_history[val_key], label=f'Validation {metric}', marker='s', markersize=3)
            
        # Set labels and legend
        ax.set_ylabel(metric.capitalize())
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
    # Set x-axis label for the bottom subplot
    axes[-1].set_xlabel('Epoch')
    
    # Set title
    fig.suptitle('Training Metrics History', fontsize=14)
    
    plt.tight_layout()
    
    # Save figure if output path provided
    if output_path:
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved metrics history plot to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save metrics history plot: {e}")
    
    return fig