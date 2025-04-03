"""
Enhanced visualization utilities for model analysis and reporting.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

# Import from base_visualization instead of utils.visualization to avoid circular import
from utils.logger import get_logger
from utils.paths import ensure_dir

from .base_visualization import DEFAULT_THEME, apply_dark_theme

logger = get_logger(__name__)


def plot_enhanced_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 12),
    normalize: bool = True,
    cmap: str = "YlOrRd",  # Changed from viridis to YlOrRd
    title: str = "Confusion Matrix",
    theme: Optional[Dict] = None,
) -> Figure:
    """
    Plot enhanced confusion matrix as a heatmap, styled like the example image.

    Args:
        cm: Confusion matrix
        class_names: List of class names
        output_path: Path to save the figure
        figsize: Figure size
        normalize: Whether to normalize by row
        cmap: Colormap (defaults to YlOrRd)
        title: Plot title
        theme: Theme settings to apply (uses DEFAULT_THEME if None)

    Returns:
        Matplotlib figure
    """
    # Apply theme
    theme = theme or DEFAULT_THEME
    apply_dark_theme(theme)

    # Create a normalized version of the confusion matrix
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.zeros_like(cm, dtype=float)
        np.divide(cm, row_sums, out=cm_norm, where=row_sums != 0)
    else:
        cm_norm = cm

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor="black")
    ax.set_facecolor("black")

    # Create custom colormap with black at value 0
    from matplotlib.colors import LinearSegmentedColormap

    # Get the colormap and modify it to have black at the bottom
    base_cmap = plt.cm.get_cmap(cmap)
    colors = base_cmap(np.linspace(0, 1, 256))

    # Set first color to black for values = 0
    colors[0] = [0, 0, 0, 1]  # black

    # Create new colormap
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

    # Plot heatmap with custom colormap
    im = ax.imshow(
        cm_norm,
        cmap=custom_cmap,
        aspect="auto",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )

    # Add colorbar to the right with appropriate styles
    cbar = ax.figure.colorbar(im, ax=ax, pad=0.01)
    cbar.ax.set_ylabel(
        "Proportion" if normalize else "Count", rotation=-90, va="bottom", color="white"
    )
    cbar.ax.tick_params(colors="white")

    # Set title and labels
    plt.title(title, fontsize=14, color="white")
    ax.set_xlabel("Predicted label", fontsize=12, color="white")
    ax.set_ylabel("True label", fontsize=12, color="white")

    # Configure ticks
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=8, color="white")
    ax.set_yticklabels(class_names, fontsize=8, color="white")

    # Customize appearance
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Add grid lines to better delineate cells
    ax.set_xticks(np.arange(-0.5, len(class_names)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(class_names)), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=1)

    # Save figure if output path provided
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        plt.savefig(
            output_path,
            dpi=400,
            bbox_inches="tight",
            facecolor="black",
            transparent=False,
        )
        logger.info(f"Saved enhanced confusion matrix plot to {output_path}")

    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    class_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 10),  # Larger figure
    theme: Optional[Dict] = None,
) -> Figure:
    """
    Plot precision-recall curves for multi-class classification.

    Args:
        y_true: One-hot encoded true labels or class indices
        y_scores: Predicted scores/probabilities
        class_names: List of class names
        output_path: Path to save the figure
        figsize: Figure size
        theme: Theme settings to apply (uses DEFAULT_THEME if None)

    Returns:
        Matplotlib figure
    """
    # Apply theme
    theme = theme or DEFAULT_THEME
    apply_dark_theme(theme)

    # Create figure with two subplots: main PR and zoomed PR
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=figsize,
        facecolor=theme["background_color"],
        gridspec_kw={"width_ratios": [2, 1]},
    )

    # Set background color for both axes
    ax1.set_facecolor(theme["background_color"])
    ax2.set_facecolor(theme["background_color"])

    # Convert to class indices if needed
    if y_true.ndim > 1 and y_true.shape[1] > 1:  # One-hot encoded
        y_true_indices = np.argmax(y_true, axis=1)
    else:
        y_true_indices = y_true.flatten()

    # Get unique classes
    unique_classes = np.unique(y_true_indices)

    # Colors for lines
    color_map = plt.cm.get_cmap("tab20", len(unique_classes))

    # Store average precision scores for display
    avg_precisions = {}

    # Determine how many classes to show individually (to avoid clutter)
    max_visible_classes = min(10, len(unique_classes))

    # Find classes with highest average precision for display
    all_ap_scores = []

    # Calculate PR curve for each class
    for _i, class_idx in enumerate(unique_classes):
        # Convert to binary classification problem (one-vs-rest)
        binary_y_true = (y_true_indices == class_idx).astype(int)

        # Get scores for this class
        if y_scores.ndim > 1 and y_scores.shape[1] > 1:
            class_scores = y_scores[:, class_idx]
        else:
            class_scores = y_scores

        # Calculate precision and recall
        precision, recall, _ = precision_recall_curve(binary_y_true, class_scores)

        # Calculate average precision
        ap = average_precision_score(binary_y_true, class_scores)

        # Store for sorting
        all_ap_scores.append((class_idx, ap))
        avg_precisions[class_idx] = ap

    # Sort classes by AP score and get top classes
    all_ap_scores.sort(key=lambda x: x[1], reverse=True)
    top_classes = [idx for idx, _ in all_ap_scores[:max_visible_classes]]

    # Plot PR curve for each of the top classes
    for i, class_idx in enumerate(top_classes):
        # Convert to binary classification problem (one-vs-rest)
        binary_y_true = (y_true_indices == class_idx).astype(int)

        # Get scores for this class
        if y_scores.ndim > 1 and y_scores.shape[1] > 1:
            class_scores = y_scores[:, class_idx]
        else:
            class_scores = y_scores

        # Calculate precision and recall
        precision, recall, _ = precision_recall_curve(binary_y_true, class_scores)

        # Get AP score from stored values
        ap = avg_precisions[class_idx]

        # Get class name (handling index errors)
        if class_idx < len(class_names):
            class_name = class_names[class_idx]
        else:
            class_name = f"Class {class_idx}"

        # Plot precision-recall curve on main plot
        ax1.plot(
            recall,
            precision,
            lw=2,
            alpha=0.8,
            color=color_map(i),
            label=f"{class_name} (AP: {ap:.2f})",
        )

        # Also plot on zoomed plot
        ax2.plot(
            recall,
            precision,
            lw=2,
            alpha=0.8,
            color=color_map(i),
        )

    # Calculate micro-average precision-recall curve
    if y_scores.ndim > 1 and y_scores.shape[1] > 1:
        # For multi-class, convert to one-hot
        y_true_bin = np.zeros_like(y_scores)
        for i in range(len(y_true_indices)):
            if y_true_indices[i] < y_true_bin.shape[1]:
                y_true_bin[i, y_true_indices[i]] = 1

        # Calculate micro-average
        precision_micro, recall_micro, _ = precision_recall_curve(
            y_true_bin.ravel(), y_scores.ravel()
        )
        ap_micro = average_precision_score(y_true_bin, y_scores, average="micro")

        # Plot micro-average curve with thicker line on both plots
        ax1.plot(
            recall_micro,
            precision_micro,
            color="gold",
            lw=3,
            alpha=1.0,
            label=f"Micro-average (AP: {ap_micro:.2f})",
        )

        ax2.plot(
            recall_micro,
            precision_micro,
            color="gold",
            lw=3,
            alpha=1.0,
        )

    # Set axis range for main plot
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])

    # Set zoomed region on second plot (0.8-1.0 Recall, 0.8-1.0 Precision)
    ax2.set_xlim([0.8, 1.0])
    ax2.set_ylim([0.8, 1.02])

    # Set title and labels
    ax1.set_title("Precision-Recall Curves", fontsize=16, color=theme["text_color"])
    ax1.set_xlabel("Recall", fontsize=14, color=theme["text_color"])
    ax1.set_ylabel("Precision", fontsize=14, color=theme["text_color"])

    ax2.set_title(
        "Zoomed PR (High Performance)", fontsize=16, color=theme["text_color"]
    )
    ax2.set_xlabel("Recall", fontsize=14, color=theme["text_color"])
    ax2.set_ylabel("Precision", fontsize=14, color=theme["text_color"])

    # Add grid to both plots
    ax1.grid(True, linestyle="--", alpha=0.3, color=theme["grid_color"])
    ax2.grid(True, linestyle="--", alpha=0.3, color=theme["grid_color"])

    # Customize tick colors
    ax1.tick_params(colors=theme["text_color"], labelsize=12)
    ax2.tick_params(colors=theme["text_color"], labelsize=12)

    # Create legend box outside the plot
    legend = ax1.legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=True,
        facecolor=theme["background_color"],
        edgecolor=theme["grid_color"],
        fontsize=10,
    )
    for text in legend.get_texts():
        text.set_color(theme["text_color"])

    plt.tight_layout()

    # Save figure if output path provided
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        plt.savefig(
            output_path,
            dpi=400,
            bbox_inches="tight",
            facecolor=theme["background_color"],
        )
        logger.info(f"Saved precision-recall curve plot to {output_path}")

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    class_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 10),  # Larger figure
    theme: Optional[Dict] = None,
) -> Figure:
    """
    Generate ROC curves for multi-class classification.

    Args:
        y_true: One-hot encoded true labels or class indices
        y_scores: Predicted scores/probabilities
        class_names: List of class names
        output_path: Path to save the figure
        figsize: Figure size
        theme: Theme settings to apply (uses DEFAULT_THEME if None)

    Returns:
        Matplotlib figure
    """
    # Apply theme
    theme = theme or DEFAULT_THEME
    apply_dark_theme(theme)

    # Create figure with two subplots: main ROC and zoomed ROC
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=figsize,
        facecolor=theme["background_color"],
        gridspec_kw={"width_ratios": [2, 1]},
    )

    # Set background color for both axes
    ax1.set_facecolor(theme["background_color"])
    ax2.set_facecolor(theme["background_color"])

    # Convert to class indices if needed
    if y_true.ndim > 1 and y_true.shape[1] > 1:  # One-hot encoded
        y_true_indices = np.argmax(y_true, axis=1)
    else:
        y_true_indices = y_true.flatten()

    # Get unique classes
    unique_classes = np.unique(y_true_indices)

    # Colors for lines
    color_map = plt.cm.get_cmap("tab20", len(unique_classes))

    # Store AUC scores for display
    auc_scores = {}

    # Determine how many classes to show individually (to avoid clutter)
    max_visible_classes = min(10, len(unique_classes))

    # Find classes with highest AUC for display
    all_auc_scores = []

    # Calculate ROC curve for each class
    for _i, class_idx in enumerate(unique_classes):
        # Convert to binary classification problem (one-vs-rest)
        binary_y_true = (y_true_indices == class_idx).astype(int)

        # Get scores for this class
        if y_scores.ndim > 1 and y_scores.shape[1] > 1:
            class_scores = y_scores[:, class_idx]
        else:
            class_scores = y_scores

        # Calculate false positive rate and true positive rate
        fpr, tpr, _ = roc_curve(binary_y_true, class_scores)

        # Calculate AUC
        roc_auc = auc(fpr, tpr)

        # Store for sorting
        all_auc_scores.append((class_idx, roc_auc))
        auc_scores[class_idx] = roc_auc

    # Sort classes by AUC score and get top classes
    all_auc_scores.sort(key=lambda x: x[1], reverse=True)
    top_classes = [idx for idx, _ in all_auc_scores[:max_visible_classes]]

    # Plot ROC curve for each of the top classes
    for i, class_idx in enumerate(top_classes):
        # Convert to binary classification problem (one-vs-rest)
        binary_y_true = (y_true_indices == class_idx).astype(int)

        # Get scores for this class
        if y_scores.ndim > 1 and y_scores.shape[1] > 1:
            class_scores = y_scores[:, class_idx]
        else:
            class_scores = y_scores

        # Calculate FPR and TPR
        fpr, tpr, _ = roc_curve(binary_y_true, class_scores)

        # Get AUC score from stored values
        roc_auc = auc_scores[class_idx]

        # Get class name (handling index errors)
        if class_idx < len(class_names):
            class_name = class_names[class_idx]
        else:
            class_name = f"Class {class_idx}"

        # Plot ROC curve on main plot
        ax1.plot(
            fpr,
            tpr,
            lw=2,
            alpha=0.8,
            color=color_map(i),
            label=f"{class_name} (AUC: {roc_auc:.2f})",
        )

        # Also plot on zoomed plot
        ax2.plot(
            fpr,
            tpr,
            lw=2,
            alpha=0.8,
            color=color_map(i),
        )

    # Calculate micro-average ROC curve
    if y_scores.ndim > 1 and y_scores.shape[1] > 1:
        # For multi-class, convert to one-hot
        y_true_bin = np.zeros_like(y_scores)
        for i in range(len(y_true_indices)):
            if y_true_indices[i] < y_true_bin.shape[1]:
                y_true_bin[i, y_true_indices[i]] = 1

        # Calculate micro-average
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
        auc_micro = auc(fpr_micro, tpr_micro)

        # Plot micro-average curve with thicker line on both plots
        ax1.plot(
            fpr_micro,
            tpr_micro,
            color="gold",
            lw=3,
            alpha=1.0,
            label=f"Micro-average (AUC: {auc_micro:.2f})",
        )

        ax2.plot(
            fpr_micro,
            tpr_micro,
            color="gold",
            lw=3,
            alpha=1.0,
        )

    # Plot diagonal reference line on both plots
    ax1.plot(
        [0, 1],
        [0, 1],
        "#888888",
        linestyle="--",
        lw=2,
        alpha=0.5,
        label="Random classifier",
    )

    ax2.plot(
        [0, 1],
        [0, 1],
        "#888888",
        linestyle="--",
        lw=2,
        alpha=0.5,
    )

    # Set axis range for main plot
    ax1.set_xlim([-0.01, 1.01])
    ax1.set_ylim([-0.01, 1.01])

    # Set zoomed region on second plot (0-0.2 FPR, 0.8-1.0 TPR)
    ax2.set_xlim([-0.01, 0.2])
    ax2.set_ylim([0.8, 1.01])

    # Set title and labels
    ax1.set_title("ROC Curves", fontsize=16, color=theme["text_color"])
    ax1.set_xlabel("False Positive Rate", fontsize=14, color=theme["text_color"])
    ax1.set_ylabel("True Positive Rate", fontsize=14, color=theme["text_color"])

    ax2.set_title(
        "Zoomed ROC (High Performance)", fontsize=16, color=theme["text_color"]
    )
    ax2.set_xlabel("False Positive Rate", fontsize=14, color=theme["text_color"])
    ax2.set_ylabel("True Positive Rate", fontsize=14, color=theme["text_color"])

    # Add grid to both plots
    ax1.grid(True, linestyle="--", alpha=0.3, color=theme["grid_color"])
    ax2.grid(True, linestyle="--", alpha=0.3, color=theme["grid_color"])

    # Customize tick colors for both plots
    ax1.tick_params(colors=theme["text_color"], labelsize=12)
    ax2.tick_params(colors=theme["text_color"], labelsize=12)

    # Show the zoomed region on the main plot
    from matplotlib.patches import Rectangle

    ax1.add_patch(
        Rectangle(
            (0, 0.8),
            0.2,
            0.2,
            fill=False,
            edgecolor="white",
            linestyle=":",
            linewidth=1.5,
        )
    )

    # Create legend at the bottom
    legend = ax1.legend(
        loc="center left",
        bbox_to_anchor=(0.5, -0.15),
        frameon=True,
        facecolor=theme["background_color"],
        edgecolor=theme["grid_color"],
        fontsize=10,
        ncol=2,
    )
    for text in legend.get_texts():
        text.set_color(theme["text_color"])

    plt.tight_layout()

    # Save figure if output path provided
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        plt.savefig(
            output_path,
            dpi=400,
            bbox_inches="tight",
            facecolor=theme["background_color"],
        )
        logger.info(f"Saved ROC curve plot to {output_path}")

    return fig


def plot_feature_space(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 10),
    reduction_method: str = "tsne",
    theme: Optional[Dict] = None,
) -> Figure:
    """
    Visualize feature space using dimensionality reduction.

    Args:
        features: Feature vectors (n_samples, n_features)
        labels: Class labels (n_samples,)
        class_names: List of class names
        output_path: Path to save the figure
        figsize: Figure size
        reduction_method: Dimensionality reduction method ('pca', 'tsne', or 'umap')
        theme: Theme settings to apply (uses DEFAULT_THEME if None)

    Returns:
        Matplotlib figure
    """
    # Apply theme
    theme = theme or DEFAULT_THEME
    apply_dark_theme(theme)

    # Apply dimensionality reduction
    if reduction_method == "pca":
        reducer = PCA(n_components=2, random_state=42)
        reduced_features = reducer.fit_transform(features)
        title = "PCA Feature Space Visualization"
    elif reduction_method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        reduced_features = reducer.fit_transform(features)
        title = "t-SNE Feature Space Visualization"
    elif reduction_method == "umap":
        reducer = umap.UMAP(
            n_components=2, random_state=42, n_neighbors=15, min_dist=0.1
        )
        reduced_features = reducer.fit_transform(features)
        title = "UMAP Feature Space Visualization"
    else:
        logger.error(f"Unknown reduction method: {reduction_method}")
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor=theme["background_color"])
    ax.set_facecolor(theme["background_color"])

    # Create color palette for classes
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)

    # Create scatter plot
    scatter = ax.scatter(
        reduced_features[:, 0],
        reduced_features[:, 1],
        c=labels,
        cmap=theme["cmap"] if num_classes > 10 else None,
        s=30,
        alpha=0.7,
        edgecolors="none",
    )

    # Set title and labels
    ax.set_title(title, color=theme["text_color"])
    ax.set_xlabel(f"{reduction_method.upper()} Component 1", color=theme["text_color"])
    ax.set_ylabel(f"{reduction_method.upper()} Component 2", color=theme["text_color"])

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.3, color=theme["grid_color"])

    # Customize tick colors
    ax.tick_params(colors=theme["text_color"])

    # Add legend
    if num_classes <= 20:  # Add legend only if not too many classes
        legend = ax.legend(
            handles=scatter.legend_elements()[0],
            labels=[class_names[i] for i in unique_labels],
            title="Classes",
            loc="best",
            frameon=True,
            facecolor=theme["background_color"],
            edgecolor=theme["grid_color"],
        )
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
            dpi=400,
            bbox_inches="tight",
            facecolor=theme["background_color"],
        )
        logger.info(f"Saved feature space visualization to {output_path}")

    return fig


def plot_tsne_visualization(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 10),
    theme: Optional[Dict] = None,
) -> Figure:
    """
    Visualize features using t-SNE.

    Args:
        features: Feature vectors (n_samples, n_features)
        labels: Class labels (n_samples,)
        class_names: List of class names
        output_path: Path to save the figure
        figsize: Figure size
        theme: Theme settings to apply (uses DEFAULT_THEME if None)

    Returns:
        Matplotlib figure
    """
    return plot_feature_space(
        features=features,
        labels=labels,
        class_names=class_names,
        output_path=output_path,
        figsize=figsize,
        reduction_method="tsne",
        theme=theme,
    )


def plot_umap_visualization(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 10),
    theme: Optional[Dict] = None,
) -> Figure:
    """
    Visualize features using UMAP.

    Args:
        features: Feature vectors (n_samples, n_features)
        labels: Class labels (n_samples,)
        class_names: List of class names
        output_path: Path to save the figure
        figsize: Figure size
        theme: Theme settings to apply (uses DEFAULT_THEME if None)

    Returns:
        Matplotlib figure
    """
    return plot_feature_space(
        features=features,
        labels=labels,
        class_names=class_names,
        output_path=output_path,
        figsize=figsize,
        reduction_method="umap",
        theme=theme,
    )


def plot_histogram(
    data: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 6),
    bins: int = 30,
    kde: bool = True,
    title: str = "Distribution Histogram",
    xlabel: str = "Value",
    ylabel: str = "Frequency",
    theme: Optional[Dict] = None,
) -> Figure:
    """
    Create a histogram with optional KDE.

    Args:
        data: Data to visualize
        output_path: Path to save the figure
        figsize: Figure size
        bins: Number of bins
        kde: Whether to show KDE curve
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        theme: Theme settings to apply (uses DEFAULT_THEME if None)

    Returns:
        Matplotlib figure
    """
    # Apply theme
    theme = theme or DEFAULT_THEME
    apply_dark_theme(theme)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor=theme["background_color"])
    ax.set_facecolor(theme["background_color"])

    # Plot histogram with seaborn
    sns.histplot(
        data,
        bins=bins,
        kde=kde,
        color=theme["main_color"],
        ax=ax,
        edgecolor=theme["background_color"],
        alpha=0.7,
        line_kws={"linewidth": 2, "color": theme["bar_colors"][1]},
    )

    # Set title and labels
    ax.set_title(title, color=theme["text_color"], fontsize=14)
    ax.set_xlabel(xlabel, color=theme["text_color"], fontsize=12)
    ax.set_ylabel(ylabel, color=theme["text_color"], fontsize=12)

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.3, color=theme["grid_color"])

    # Customize tick colors
    ax.tick_params(colors=theme["text_color"])

    # Add mean line with more visible text
    mean_val = np.mean(data)
    ax.axvline(
        mean_val,
        color=theme["bar_colors"][2],
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_val:.2f}",
    )

    # Add median line with more visible text
    median_val = np.median(data)
    ax.axvline(
        median_val,
        color=theme["bar_colors"][1],
        linestyle="-.",
        linewidth=2,
        label=f"Median: {median_val:.2f}",
    )

    # Customize legend to make text more visible
    legend = ax.legend(
        loc="upper right",
        frameon=True,
        facecolor=theme["background_color"],
        edgecolor=theme["grid_color"],
    )

    # Increase legend text size and brightness for better visibility
    for text in legend.get_texts():
        text.set_color("#FFFFFF")  # Bright white for legend text
        text.set_fontweight("bold")
        text.set_fontsize(12)

    plt.tight_layout()

    # Save figure if output path provided
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        plt.savefig(
            output_path,
            dpi=400,
            bbox_inches="tight",
            facecolor=theme["background_color"],
        )
        logger.info(f"Saved histogram plot to {output_path}")

    return fig


def plot_distribution(
    data: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 6),
    rug: bool = True,
    title: str = "Distribution Plot",
    xlabel: str = "Value",
    ylabel: str = "Density",
    theme: Optional[Dict] = None,
) -> Figure:
    """
    Create a KDE distribution plot with optional rug plot.

    Args:
        data: Data to visualize
        output_path: Path to save the figure
        figsize: Figure size
        rug: Whether to show rug plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        theme: Theme settings to apply (uses DEFAULT_THEME if None)

    Returns:
        Matplotlib figure
    """
    # Apply theme
    theme = theme or DEFAULT_THEME
    apply_dark_theme(theme)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor=theme["background_color"])
    ax.set_facecolor(theme["background_color"])

    # Plot distribution with seaborn
    sns.kdeplot(
        data,
        color=theme["main_color"],
        ax=ax,
        fill=True,
        alpha=0.3,
        linewidth=2,
        label="Density",
    )

    # Add rug plot if requested
    if rug:
        sns.rugplot(
            data,
            color=theme["bar_colors"][1],
            ax=ax,
            alpha=0.5,
            height=0.05,
            label="Data points",
        )

    # Set title and labels
    ax.set_title(title, color=theme["text_color"])
    ax.set_xlabel(xlabel, color=theme["text_color"])
    ax.set_ylabel(ylabel, color=theme["text_color"])

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.3, color=theme["grid_color"])

    # Customize tick colors
    ax.tick_params(colors=theme["text_color"])

    # Add mean line
    mean_val = np.mean(data)
    ax.axvline(
        mean_val,
        color=theme["bar_colors"][2],
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_val:.2f}",
    )

    # Add legend
    ax.legend(
        frameon=True,
        facecolor=theme["background_color"],
        edgecolor=theme["grid_color"],
        loc="best",
    )

    plt.tight_layout()

    # Save figure if output path provided
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        plt.savefig(
            output_path,
            dpi=400,
            bbox_inches="tight",
            facecolor=theme["background_color"],
        )
        logger.info(f"Saved distribution plot to {output_path}")

    return fig


def plot_class_performance(
    metrics: Dict[str, Dict[str, float]],
    class_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 12),  # Increased height
    metrics_to_plot: List[str] = ["precision", "recall", "f1"],
    sort_by: str = "f1",
    theme: Optional[Dict] = None,
) -> Figure:
    """
    Plot class performance as horizontal bar chart.

    Args:
        metrics: Dictionary of class metrics (class_name -> metric_name -> value)
        class_names: List of class names
        output_path: Path to save the figure
        figsize: Figure size
        metrics_to_plot: List of metrics to plot
        sort_by: Which metric to sort by
        theme: Theme settings to apply (uses DEFAULT_THEME if None)

    Returns:
        Matplotlib figure
    """
    # Apply theme
    theme = theme or DEFAULT_THEME
    apply_dark_theme(theme)

    # Convert metrics to DataFrame
    data = []
    for class_name in class_names:
        if class_name in metrics:
            row = {"Class": class_name}
            for metric in metrics_to_plot:
                if metric in metrics[class_name]:
                    row[metric.capitalize()] = metrics[class_name][metric]
            data.append(row)

    df = pd.DataFrame(data)

    # If no data, return empty figure
    if len(df) == 0:
        logger.warning("No data available for class performance plot")
        fig, ax = plt.subplots(figsize=figsize, facecolor=theme["background_color"])
        ax.set_facecolor(theme["background_color"])
        ax.text(
            0.5,
            0.5,
            "No performance data available",
            ha="center",
            va="center",
            color=theme["text_color"],
            fontsize=14,
        )
        ax.axis("off")
        return fig

    # Sort by the requested metric
    sort_col = sort_by.capitalize()
    if sort_col in df.columns:
        df = df.sort_values(by=sort_col)

    # Set class as index
    df = df.set_index("Class")

    # Create figure with more height to accommodate classes
    fig, ax = plt.subplots(figsize=figsize, facecolor=theme["background_color"])
    ax.set_facecolor(theme["background_color"])

    # Plot horizontal bars with smaller height to fit more classes
    df.plot(
        kind="barh",
        ax=ax,
        color=[
            theme["bar_colors"][i % len(theme["bar_colors"])]
            for i in range(len(metrics_to_plot))
        ],
        width=0.7,  # Reduced width to create more space between bars
        alpha=0.8,
        edgecolor=theme["background_color"],
    )

    # Set title and labels
    ax.set_title("Performance by Class", color=theme["text_color"])
    ax.set_xlabel("Score", color=theme["text_color"])
    ax.set_ylabel("Class", color=theme["text_color"])

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.3, color=theme["grid_color"])

    # Customize tick colors
    ax.tick_params(colors=theme["text_color"])

    # Reduce font size of the y-axis labels to make them more compact
    ax.tick_params(axis="y", labelsize=8)

    # Adjust bottom margin to ensure visibility of bottom classes
    plt.subplots_adjust(bottom=0.15, top=0.95)

    # Customize legend
    legend = ax.legend(
        frameon=True,
        facecolor=theme["background_color"],
        edgecolor=theme["grid_color"],
        loc="lower right",
    )
    for text in legend.get_texts():
        text.set_color(theme["text_color"])

    # Set x-axis range
    ax.set_xlim([0, 1.02])

    plt.tight_layout()

    # Save figure if output path provided
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        plt.savefig(
            output_path,
            dpi=400,
            bbox_inches="tight",
            facecolor=theme["background_color"],
        )
        logger.info(f"Saved class performance plot to {output_path}")

    return fig


def plot_confidence_distribution(
    confidences: np.ndarray,
    correctness: Optional[np.ndarray] = None,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 6),
    bins: int = 20,
    title: str = "Prediction Confidence Distribution",
    theme: Optional[Dict] = None,
) -> Figure:
    """
    Plot distribution of prediction confidences.

    Args:
        confidences: Prediction confidence scores
        correctness: Boolean array indicating if predictions were correct
        output_path: Path to save the figure
        figsize: Figure size
        bins: Number of bins
        title: Plot title
        theme: Theme settings to apply (uses DEFAULT_THEME if None)

    Returns:
        Matplotlib figure
    """
    # Apply theme
    theme = theme or DEFAULT_THEME
    apply_dark_theme(theme)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor=theme["background_color"])
    ax.set_facecolor(theme["background_color"])

    # If correctness is provided, show separate distributions
    if correctness is not None:
        correct_conf = confidences[correctness]
        incorrect_conf = confidences[~correctness]

        # Plot histograms
        sns.histplot(
            correct_conf,
            bins=bins,
            color=theme["bar_colors"][2],  # Green for correct
            alpha=0.7,
            label=f"Correct predictions ({len(correct_conf)})",
            ax=ax,
            kde=True,
            stat="density",
        )

        sns.histplot(
            incorrect_conf,
            bins=bins,
            color=theme["bar_colors"][3],  # Red for incorrect
            alpha=0.7,
            label=f"Incorrect predictions ({len(incorrect_conf)})",
            ax=ax,
            kde=True,
            stat="density",
        )

        # Add means
        if len(correct_conf) > 0:
            mean_correct = np.mean(correct_conf)
            ax.axvline(
                mean_correct,
                color=theme["bar_colors"][2],
                linestyle="--",
                linewidth=2,
                label=f"Mean correct: {mean_correct:.2f}",
            )

        if len(incorrect_conf) > 0:
            mean_incorrect = np.mean(incorrect_conf)
            ax.axvline(
                mean_incorrect,
                color=theme["bar_colors"][3],
                linestyle="--",
                linewidth=2,
                label=f"Mean incorrect: {mean_incorrect:.2f}",
            )
    else:
        # Plot single histogram
        sns.histplot(
            confidences,
            bins=bins,
            color=theme["main_color"],
            alpha=0.7,
            kde=True,
            ax=ax,
            stat="density",
        )

        # Add mean line
        mean_conf = np.mean(confidences)
        ax.axvline(
            mean_conf,
            color=theme["bar_colors"][1],
            linestyle="--",
            linewidth=2,
            label=f"Mean confidence: {mean_conf:.2f}",
        )

    # Set title and labels
    ax.set_title(title, color=theme["text_color"])
    ax.set_xlabel("Confidence", color=theme["text_color"])
    ax.set_ylabel("Density", color=theme["text_color"])

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.3, color=theme["grid_color"])

    # Customize tick colors
    ax.tick_params(colors=theme["text_color"])

    # Customize legend
    legend = ax.legend(
        frameon=True,
        facecolor=theme["background_color"],
        edgecolor=theme["grid_color"],
        loc="best",
    )
    for text in legend.get_texts():
        text.set_color(theme["text_color"])

    plt.tight_layout()

    # Save figure if output path provided
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        plt.savefig(
            output_path,
            dpi=400,
            bbox_inches="tight",
            facecolor=theme["background_color"],
        )
        logger.info(f"Saved confidence distribution plot to {output_path}")

    return fig


def create_classification_examples_grid(
    images: np.ndarray,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    class_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (15, 12),
    max_examples: int = 20,
    separate_correct_incorrect: bool = True,
    theme: Optional[Dict] = None,
) -> Figure:
    """
    Create a grid visualization of image examples with their predictions.

    Args:
        images: Array of images (n_samples, height, width, channels) or (n_samples, channels, height, width)
        true_labels: Array of true labels (indices)
        pred_labels: Array of predicted labels (indices)
        class_names: List of class names
        output_path: Path to save the figure
        figsize: Figure size
        max_examples: Maximum number of examples to show
        separate_correct_incorrect: Whether to separate correct and incorrect predictions
        theme: Theme settings to apply (uses DEFAULT_THEME if None)

    Returns:
        Matplotlib figure
    """
    # Apply theme
    theme = theme or DEFAULT_THEME
    apply_dark_theme(theme)

    # Convert predicted and true labels to class indices if needed
    if true_labels.ndim > 1 and true_labels.shape[1] > 1:  # One-hot encoded
        true_indices = np.argmax(true_labels, axis=1)
    else:
        true_indices = true_labels.flatten()

    if pred_labels.ndim > 1 and pred_labels.shape[1] > 1:  # One-hot encoded
        pred_indices = np.argmax(pred_labels, axis=1)
    else:
        pred_indices = pred_labels.flatten()

    # Check if images are in CHW format (PyTorch) and convert to HWC if needed
    if images.ndim == 4 and images.shape[1] in [1, 3]:
        images = np.transpose(images, (0, 2, 3, 1))

    # Convert grayscale to RGB for display
    if images.shape[-1] == 1:
        images = np.repeat(images, 3, axis=-1)

    # Check correctness of predictions
    correctness = pred_indices == true_indices

    # Determine samples to display
    if separate_correct_incorrect:
        correct_samples = np.where(correctness)[0]
        incorrect_samples = np.where(~correctness)[0]

        # Limit to max_examples/2 for each category
        max_per_category = max_examples // 2

        if len(correct_samples) > max_per_category:
            correct_samples = np.random.choice(
                correct_samples, max_per_category, replace=False
            )

        if len(incorrect_samples) > max_per_category:
            incorrect_samples = np.random.choice(
                incorrect_samples, max_per_category, replace=False
            )

        # Combine samples
        selected_samples = np.concatenate([correct_samples, incorrect_samples])
    else:
        # Random selection up to max_examples
        if len(images) > max_examples:
            selected_samples = np.random.choice(
                len(images), max_examples, replace=False
            )
        else:
            selected_samples = np.arange(len(images))

    # Sort selected samples for better visualization
    if separate_correct_incorrect:
        # Sort by correctness first
        selected_samples = sorted(
            selected_samples, key=lambda i: (not correctness[i], true_indices[i])
        )

    # Calculate grid dimensions
    n_samples = len(selected_samples)
    grid_size = int(np.ceil(np.sqrt(n_samples)))
    rows = grid_size
    cols = grid_size

    # Create figure
    fig, axes = plt.subplots(
        rows, cols, figsize=figsize, facecolor=theme["background_color"]
    )
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    # Flatten axes for easier indexing
    axes = axes.flatten()

    # Plot each image
    for i, ax_idx in enumerate(range(rows * cols)):
        ax = axes[ax_idx]

        # Set background color
        ax.set_facecolor(theme["background_color"])

        if i < n_samples:
            sample_idx = selected_samples[i]
            image = images[sample_idx]
            true_idx = true_indices[sample_idx]
            pred_idx = pred_indices[sample_idx]

            # Normalize image if needed
            if image.max() > 1.0:
                image = image / 255.0

            # Display image
            ax.imshow(image, interpolation="nearest")

            # Add colored border based on correctness
            is_correct = true_idx == pred_idx
            border_color = (
                theme["bar_colors"][2] if is_correct else theme["bar_colors"][3]
            )

            # Add title with true and predicted labels
            true_name = class_names[true_idx]
            pred_name = class_names[pred_idx]

            if len(true_name) > 15:
                true_name = true_name[:12] + "..."
            if len(pred_name) > 15:
                pred_name = pred_name[:12] + "..."

            title_color = (
                theme["bar_colors"][2] if is_correct else theme["bar_colors"][3]
            )
            ax.set_title(
                f"True: {true_name}\nPred: {pred_name}", color=title_color, fontsize=8
            )

            # Add colored border
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(2)
        else:
            # Hide unused subplots
            ax.axis("off")

        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    # Add legend for correctness
    legend_elements = [
        plt.Line2D(
            [0], [0], color=theme["bar_colors"][2], lw=4, label="Correct Prediction"
        ),
        plt.Line2D(
            [0], [0], color=theme["bar_colors"][3], lw=4, label="Incorrect Prediction"
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        frameon=True,
        facecolor=theme["background_color"],
        edgecolor=theme["grid_color"],
        ncol=2,
    )

    # Add title
    plt.suptitle(
        "Classification Examples", fontsize=16, color=theme["text_color"], y=0.98
    )

    # Save figure if output path provided
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        plt.savefig(
            output_path,
            dpi=400,
            bbox_inches="tight",
            facecolor=theme["background_color"],
        )
        logger.info(f"Saved classification examples grid to {output_path}")

    return fig


def visualize_augmentations(
    image: np.ndarray,
    augmentation_fn: callable,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (15, 10),
    n_augmentations: int = 8,
    seed: int = 42,
    title: str = "Data Augmentation Visualization",
    theme: Optional[Dict] = None,
) -> Figure:
    """
    Visualize augmentations of an image.

    Args:
        image: Original image (height, width, channels) or (channels, height, width)
        augmentation_fn: Function that takes an image and returns an augmented version
        output_path: Path to save the figure
        figsize: Figure size
        n_augmentations: Number of augmented examples to generate
        seed: Random seed for reproducibility
        title: Figure title
        theme: Theme settings to apply (uses DEFAULT_THEME if None)

    Returns:
        Matplotlib figure
    """
    # Apply theme
    theme = theme or DEFAULT_THEME
    apply_dark_theme(theme)

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Check if image is in CHW format (PyTorch) and convert to HWC if needed
    if image.ndim == 3 and image.shape[0] in [1, 3]:
        image = np.transpose(image, (1, 2, 0))

    # Convert grayscale to RGB for display
    if image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)

    # Generate augmented examples
    augmented_images = []
    for _ in range(n_augmentations):
        # Apply augmentation
        aug_image = augmentation_fn(image.copy())

        # Convert to HWC format if necessary
        if aug_image.ndim == 3 and aug_image.shape[0] in [1, 3]:
            aug_image = np.transpose(aug_image, (1, 2, 0))

        # Handle grayscale images
        if aug_image.ndim == 3 and aug_image.shape[-1] == 1:
            aug_image = np.repeat(aug_image, 3, axis=-1)
        elif aug_image.ndim == 2:
            aug_image = np.stack([aug_image, aug_image, aug_image], axis=-1)

        augmented_images.append(aug_image)

    # Calculate grid dimensions
    rows = 3
    cols = 3  # Original + n_augmentations (up to 8)

    # Create figure
    fig, axes = plt.subplots(
        rows, cols, figsize=figsize, facecolor=theme["background_color"]
    )
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    # Plot original image in the center of the first row
    center_col = cols // 2
    ax_original = axes[0, center_col]
    ax_original.imshow(image, interpolation="nearest")
    ax_original.set_title("Original Image", color=theme["text_color"])
    ax_original.set_xticks([])
    ax_original.set_yticks([])

    # Hide unused axes in the first row
    for c in range(cols):
        if c != center_col:
            axes[0, c].axis("off")

    # Plot augmented images in the remaining rows
    for i, aug_image in enumerate(augmented_images):
        row = (i // (cols)) + 1
        col = i % cols

        if row < rows:  # Ensure we don't exceed the grid
            # Normalize image if needed
            if aug_image.max() > 1.0:
                aug_image = aug_image / 255.0

            # Display image
            axes[row, col].imshow(aug_image, interpolation="nearest")
            axes[row, col].set_title(
                f"Augmentation {i + 1}", color=theme["text_color"], fontsize=10
            )
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

            # Add border to highlight the difference
            for spine in axes[row, col].spines.values():
                spine.set_edgecolor(theme["bar_colors"][i % len(theme["bar_colors"])])
                spine.set_linewidth(2)

    # Hide any unused subplots
    for i in range(n_augmentations, (rows - 1) * cols):
        row = (i // cols) + 1
        col = i % cols
        if row < rows:
            axes[row, col].axis("off")

    # Set background color for all subplots
    for ax in axes.flatten():
        ax.set_facecolor(theme["background_color"])

    # Add overall title
    plt.suptitle(title, fontsize=16, color=theme["text_color"], y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for title

    # Save figure if output path provided
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        plt.savefig(
            output_path,
            dpi=400,
            bbox_inches="tight",
            facecolor=theme["background_color"],
        )
        logger.info(f"Saved augmentation visualization to {output_path}")

    return fig


def plot_scatter(
    x: np.ndarray,
    y: np.ndarray,
    labels: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Scatter Plot",
    xlabel: str = "X",
    ylabel: str = "Y",
    alpha: float = 0.7,
    theme: Optional[Dict] = None,
) -> Figure:
    """
    Create a scatter plot for two variables with optional categorical coloring.

    Args:
        x: X-axis data
        y: Y-axis data
        labels: Optional categorical labels for coloring points
        class_names: Names of classes (if labels are provided)
        output_path: Path to save the figure
        figsize: Figure size
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        alpha: Alpha transparency for points
        theme: Theme settings to apply (uses DEFAULT_THEME if None)

    Returns:
        Matplotlib figure
    """
    # Apply theme
    theme = theme or DEFAULT_THEME
    apply_dark_theme(theme)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor=theme["background_color"])
    ax.set_facecolor(theme["background_color"])

    # Plot scatter with or without categorical coloring
    if labels is not None:
        unique_labels = np.unique(labels)

        # Create scatter plot with categorical coloring
        scatter = ax.scatter(
            x, y, c=labels, cmap=theme["cmap"], alpha=alpha, s=50, edgecolors="none"
        )

        # Add legend if class names are provided
        if class_names is not None:
            legend_elements = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=scatter.cmap(scatter.norm(label)),
                    markersize=10,
                    label=(
                        class_names[label]
                        if label < len(class_names)
                        else f"Class {label}"
                    ),
                )
                for label in unique_labels
            ]

            ax.legend(
                handles=legend_elements,
                loc="best",
                frameon=True,
                facecolor=theme["background_color"],
                edgecolor=theme["grid_color"],
            )
    else:
        # Simple scatter plot without categorical coloring
        ax.scatter(
            x, y, color=theme["main_color"], alpha=alpha, s=50, edgecolors="none"
        )

    # Set title and labels
    ax.set_title(title, color=theme["text_color"])
    ax.set_xlabel(xlabel, color=theme["text_color"])
    ax.set_ylabel(ylabel, color=theme["text_color"])

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.3, color=theme["grid_color"])

    # Customize tick colors
    ax.tick_params(colors=theme["text_color"])

    plt.tight_layout()

    # Save figure if output path provided
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        plt.savefig(
            output_path,
            dpi=400,
            bbox_inches="tight",
            facecolor=theme["background_color"],
        )
        logger.info(f"Saved scatter plot to {output_path}")

    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    output_path: Optional[Union[str, Path]] = None,
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8),
    theme: Optional[Dict] = None,
) -> Figure:
    """
    Plot training metrics over epochs as line graphs.

    Args:
        history: Dictionary with metrics history (keys like 'loss', 'val_loss', etc.)
        output_path: Path to save the figure
        metrics: List of metrics to plot ('loss', 'accuracy', etc.)
        figsize: Figure size
        theme: Theme settings to apply (uses DEFAULT_THEME if None)

    Returns:
        Matplotlib figure
    """
    # Apply theme
    theme = theme or DEFAULT_THEME
    apply_dark_theme(theme)

    # Determine metrics to plot
    if metrics is None:
        # Try to detect metrics by looking for common patterns
        available_metrics = set()
        for key in history:
            # Check for train/val prefix
            parts = key.split("_")
            if parts[0] in ["train", "val"]:
                metric = "_".join(parts[1:])
                available_metrics.add(metric)
            else:
                available_metrics.add(key)

        metrics = list(available_metrics)

    # Calculate grid size
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division

    # Create figure
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=figsize, facecolor=theme["background_color"]
    )

    # Convert to array for easy indexing
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    elif n_rows == 1 or n_cols == 1:
        axes = np.array(axes).reshape(-1)

    # Plot each metric
    for i, metric in enumerate(metrics):
        if i < len(axes):
            ax = axes.flat[i]

            # Identify train and validation keys for this metric
            train_key = f"train_{metric}" if f"train_{metric}" in history else metric
            val_key = f"val_{metric}" if f"val_{metric}" in history else None

            # Plot training metric
            if train_key in history:
                epochs = range(1, len(history[train_key]) + 1)
                ax.plot(
                    epochs,
                    history[train_key],
                    "o-",
                    color=theme["main_color"],
                    label=f"Training {metric.capitalize()}",
                    linewidth=2,
                )

            # Plot validation metric if available
            if val_key in history:
                epochs = range(1, len(history[val_key]) + 1)
                ax.plot(
                    epochs,
                    history[val_key],
                    "o-",
                    color=theme["bar_colors"][0],
                    label=f"Validation {metric.capitalize()}",
                    linewidth=2,
                )

            # Add best value as a star marker
            if train_key in history:
                if "loss" in train_key.lower():
                    best_epoch = np.argmin(history[train_key])
                    best_value = np.min(history[train_key])
                else:
                    best_epoch = np.argmax(history[train_key])
                    best_value = np.max(history[train_key])

                ax.plot(
                    best_epoch + 1,
                    best_value,
                    "*",
                    color="gold",
                    markersize=15,
                    label=f"Best: {best_value:.4f}",
                )

            # Set title and labels
            ax.set_title(f"{metric.capitalize()}", color=theme["text_color"])
            ax.set_xlabel("Epoch", color=theme["text_color"])
            ax.set_ylabel(metric.capitalize(), color=theme["text_color"])

            # Add grid
            ax.grid(True, linestyle="--", alpha=0.3, color=theme["grid_color"])

            # Customize tick colors
            ax.tick_params(colors=theme["text_color"])

            # Add legend
            ax.legend(
                loc="best",
                frameon=True,
                facecolor=theme["background_color"],
                edgecolor=theme["grid_color"],
            )

            # Set background color
            ax.set_facecolor(theme["background_color"])

    # Hide unused subplots
    for i in range(len(metrics), len(axes)):
        axes.flat[i].axis("off")

    plt.tight_layout()

    # Save figure if output path provided
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        plt.savefig(
            output_path,
            dpi=400,
            bbox_inches="tight",
            facecolor=theme["background_color"],
        )
        logger.info(f"Saved training curves plot to {output_path}")

    return fig


def plot_training_time_analysis(
    epoch_times: List[float],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 6),
    theme: Optional[Dict] = None,
) -> Figure:
    """
    Plot training time per epoch with additional statistics.

    Args:
        epoch_times: List of training times per epoch (in seconds)
        output_path: Path to save the figure
        figsize: Figure size
        theme: Theme settings to apply (uses DEFAULT_THEME if None)

    Returns:
        Matplotlib figure
    """
    # Apply theme
    theme = theme or DEFAULT_THEME
    apply_dark_theme(theme)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor=theme["background_color"])
    ax.set_facecolor(theme["background_color"])

    # Create epochs array
    epochs = np.arange(1, len(epoch_times) + 1)

    # Plot training time per epoch
    ax.plot(
        epochs,
        epoch_times,
        "o-",
        color=theme["main_color"],
        label="Training time per epoch",
        linewidth=2,
    )

    # Calculate statistics
    mean_time = np.mean(epoch_times)
    median_time = np.median(epoch_times)
    min_time = np.min(epoch_times)
    max_time = np.max(epoch_times)

    # Add horizontal lines for mean and median
    ax.axhline(
        mean_time,
        color=theme["bar_colors"][0],
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_time:.2f}s",
    )
    ax.axhline(
        median_time,
        color=theme["bar_colors"][1],
        linestyle=":",
        linewidth=2,
        label=f"Median: {median_time:.2f}s",
    )

    # Add annotations for min and max values
    min_epoch = np.argmin(epoch_times) + 1
    max_epoch = np.argmax(epoch_times) + 1

    ax.annotate(
        f"Min: {min_time:.2f}s",
        xy=(min_epoch, min_time),
        xytext=(min_epoch + 1, min_time - 0.1 * (max_time - min_time)),
        color=theme["text_color"],
        arrowprops=dict(color=theme["bar_colors"][2], arrowstyle="->"),
    )

    ax.annotate(
        f"Max: {max_time:.2f}s",
        xy=(max_epoch, max_time),
        xytext=(max_epoch + 1, max_time - 0.1 * (max_time - min_time)),
        color=theme["text_color"],
        arrowprops=dict(color=theme["bar_colors"][3], arrowstyle="->"),
    )

    # Set title and labels
    ax.set_title("Training Time Analysis", color=theme["text_color"])
    ax.set_xlabel("Epoch", color=theme["text_color"])
    ax.set_ylabel("Time (seconds)", color=theme["text_color"])

    # Set integer ticks for epochs
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.3, color=theme["grid_color"])

    # Customize tick colors
    ax.tick_params(colors=theme["text_color"])

    # Add legend
    ax.legend(
        loc="best",
        frameon=True,
        facecolor=theme["background_color"],
        edgecolor=theme["grid_color"],
    )

    plt.tight_layout()

    # Save figure if output path provided
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        plt.savefig(
            output_path,
            dpi=400,
            bbox_inches="tight",
            facecolor=theme["background_color"],
        )
        logger.info(f"Saved training time analysis plot to {output_path}")

    return fig


def create_image_grid(
    images: np.ndarray,
    labels: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 10),
    n_images: int = 25,
    title: str = "Image Grid",
    theme: Optional[Dict] = None,
) -> Figure:
    """
    Create a grid visualization displaying a set of images.

    Args:
        images: Array of images (n_images, height, width, channels) or (n_images, channels, height, width)
        labels: Optional array of labels for the images
        class_names: Optional list of class names
        output_path: Path to save the figure
        figsize: Figure size
        n_images: Number of images to display (max)
        title: Grid title
        theme: Theme settings to apply (uses DEFAULT_THEME if None)

    Returns:
        Matplotlib figure
    """
    # Apply theme
    theme = theme or DEFAULT_THEME
    apply_dark_theme(theme)

    # Limit number of images
    n_images = min(n_images, len(images))
    images = images[:n_images]

    if labels is not None:
        labels = labels[:n_images]

    # Convert PyTorch tensor format (CHW) to HWC if needed
    if images.ndim == 4 and images.shape[1] in [1, 3]:
        images = np.transpose(images, (0, 2, 3, 1))

    # Convert grayscale to RGB for display
    if images.ndim == 4 and images.shape[-1] == 1:
        images = np.repeat(images, 3, axis=-1)

    # Calculate grid dimensions
    n_cols = int(np.ceil(np.sqrt(n_images)))
    n_rows = int(np.ceil(n_images / n_cols))

    # Create figure
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=figsize, facecolor=theme["background_color"]
    )

    # If there's only one image, axes is not a numpy array
    if n_images == 1:
        axes = np.array([axes])

    # Flatten axes for easy iteration
    axes = axes.flatten()

    # Plot each image
    for i in range(n_images):
        ax = axes[i]
        img = images[i]

        # Normalize image if needed
        if img.max() > 1.0:
            img = img / 255.0

        # Display image
        ax.imshow(img, interpolation="nearest")

        # Add label if provided
        if labels is not None:
            label = labels[i]
            if class_names is not None and label < len(class_names):
                label_text = class_names[label]
            else:
                label_text = f"Class {label}"

            ax.set_title(label_text, color=theme["text_color"], fontsize=10)

        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Set background color
        ax.set_facecolor(theme["background_color"])

    # Hide unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis("off")

    # Set super title
    plt.suptitle(title, fontsize=16, color=theme["text_color"], y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout for title

    # Save figure if output path provided
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        plt.savefig(
            output_path,
            dpi=400,
            bbox_inches="tight",
            facecolor=theme["background_color"],
        )
        logger.info(f"Saved image grid to {output_path}")

    return fig


def create_analysis_dashboard(
    dataset_info: Dict,
    class_dist: np.ndarray,
    class_names: List[str],
    sample_images: Optional[np.ndarray] = None,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (15, 12),
    theme: Optional[Dict] = None,
) -> Figure:
    """
    Create a comprehensive dashboard for dataset analysis.

    Args:
        dataset_info: Dictionary with dataset information (num_samples, img_size, etc.)
        class_dist: Array with number of samples per class
        class_names: List of class names
        sample_images: Optional array of sample images to display
        output_path: Path to save the figure
        figsize: Figure size
        theme: Theme settings to apply (uses DEFAULT_THEME if None)

    Returns:
        Matplotlib figure
    """
    # Apply theme
    theme = theme or DEFAULT_THEME
    apply_dark_theme(theme)

    # Create figure with subfigures
    fig = plt.figure(figsize=figsize, facecolor=theme["background_color"])
    gs = plt.GridSpec(3, 4, figure=fig)

    # Class distribution bar chart
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor(theme["background_color"])

    bars = ax1.barh(
        np.arange(len(class_names)), class_dist, color=theme["main_color"], alpha=0.7
    )

    # Add value labels
    for _i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(
            width + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width)}",
            ha="left",
            va="center",
            color=theme["text_color"],
        )

    ax1.set_title("Class Distribution", color=theme["text_color"])
    ax1.set_xlabel("Number of Samples", color=theme["text_color"])
    ax1.set_yticks(np.arange(len(class_names)))
    ax1.set_yticklabels(class_names, color=theme["text_color"])
    ax1.tick_params(colors=theme["text_color"])

    # Class distribution pie chart
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.set_facecolor(theme["background_color"])

    wedges, texts, autotexts = ax2.pie(
        class_dist,
        labels=None,
        autopct="%1.1f%%",
        startangle=90,
        colors=[theme["main_color"]] + theme["bar_colors"],
        wedgeprops=dict(width=0.5, edgecolor=theme["background_color"]),
    )

    for text in autotexts:
        text.set_color(theme["text_color"])

    ax2.set_title("Class Distribution (Pie Chart)", color=theme["text_color"])

    # Add legend
    ax2.legend(
        wedges,
        class_names,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=True,
        facecolor=theme["background_color"],
        edgecolor=theme["grid_color"],
    )

    # Dataset info text
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.set_facecolor(theme["background_color"])
    ax3.axis("off")

    info_text = "Dataset Information:\n\n"

    for key, value in dataset_info.items():
        info_text += f"• {key}: {value}\n"

    ax3.text(
        0.05,
        0.95,
        info_text,
        transform=ax3.transAxes,
        fontsize=12,
        verticalalignment="top",
        color=theme["text_color"],
    )

    # Class imbalance scatter plot
    ax4 = fig.add_subplot(gs[1, 2:])
    ax4.set_facecolor(theme["background_color"])

    # Calculate percentage of total for each class
    total_samples = np.sum(class_dist)
    class_percentages = class_dist / total_samples * 100

    # Create scatter plot
    scatter = ax4.scatter(
        np.arange(len(class_names)),
        class_percentages,
        c=class_percentages,
        cmap="viridis",
        s=100,
        alpha=0.8,
    )

    # Add horizontal line for balanced dataset
    balanced_percentage = 100 / len(class_names)
    ax4.axhline(
        balanced_percentage,
        color="gray",
        linestyle="--",
        label=f"Balanced ({balanced_percentage:.1f}%)",
    )

    ax4.set_title("Class Distribution (%)", color=theme["text_color"])
    ax4.set_xlabel("Class Index", color=theme["text_color"])
    ax4.set_ylabel("Percentage of Dataset", color=theme["text_color"])
    ax4.tick_params(colors=theme["text_color"])
    ax4.grid(True, linestyle="--", alpha=0.3, color=theme["grid_color"])
    ax4.legend(
        frameon=True,
        facecolor=theme["background_color"],
        edgecolor=theme["grid_color"],
    )

    # Sample images (if provided)
    if sample_images is not None:
        ax5 = fig.add_subplot(gs[2, :])
        ax5.set_facecolor(theme["background_color"])

        # Determine number of sample images to show
        n_samples = min(8, len(sample_images))
        sample_indices = np.linspace(0, len(sample_images) - 1, n_samples, dtype=int)
        samples = sample_images[sample_indices]

        # Create mini image grid
        n_cols = n_samples
        n_rows = 1

        for i in range(n_samples):
            # Create new axis for each image
            ax_img = fig.add_subplot(gs[2, i // (n_cols // 4)])

            # Get the image
            img = samples[i]

            # Convert PyTorch tensor format (CHW) to HWC if needed
            if img.ndim == 3 and img.shape[0] in [1, 3]:
                img = np.transpose(img, (1, 2, 0))

            # Convert grayscale to RGB for display
            if img.ndim == 3 and img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)
            elif img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)

            # Normalize image if needed
            if img.max() > 1.0:
                img = img / 255.0

            # Display image
            ax_img.imshow(img, interpolation="nearest")
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            ax_img.set_facecolor(theme["background_color"])

    # Adjust layout and add title
    plt.suptitle(
        "Dataset Analysis Dashboard", fontsize=16, color=theme["text_color"], y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout for title

    # Save figure if output path provided
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        plt.savefig(
            output_path,
            dpi=400,
            bbox_inches="tight",
            facecolor=theme["background_color"],
        )
        logger.info(f"Saved analysis dashboard to {output_path}")

    return fig


def plot_hierarchical_clustering(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 8),
    method: str = "ward",
    max_samples: int = 100,
    theme: Optional[Dict] = None,
) -> Figure:
    """
    Generate a hierarchical clustering dendrogram based on image features.

    Args:
        features: Feature vectors (n_samples, n_features)
        labels: Class labels (n_samples,)
        class_names: List of class names
        output_path: Path to save the figure
        figsize: Figure size
        method: Linkage method ('ward', 'complete', 'average', 'single')
        max_samples: Maximum number of samples to include
        theme: Theme settings to apply (uses DEFAULT_THEME if None)

    Returns:
        Matplotlib figure
    """
    # Apply theme
    theme = theme or DEFAULT_THEME
    apply_dark_theme(theme)

    from scipy.cluster import hierarchy
    from scipy.spatial.distance import pdist

    # If too many samples, use class means instead
    if len(features) > max_samples:
        logger.info(
            f"Too many samples ({len(features)}), using class means for clustering"
        )

        # Calculate mean feature vector for each class
        unique_labels = np.unique(labels)
        mean_features = []
        mean_labels = []

        for label in unique_labels:
            class_features = features[labels == label]
            if len(class_features) > 0:
                mean_features.append(np.mean(class_features, axis=0))
                mean_labels.append(label)

        features = np.array(mean_features)
        labels = np.array(mean_labels)
    else:
        # Subsample to max_samples
        if len(features) > max_samples:
            indices = np.random.choice(len(features), max_samples, replace=False)
            features = features[indices]
            labels = labels[indices]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor=theme["background_color"])
    ax.set_facecolor(theme["background_color"])

    # Calculate distance matrix
    distance_matrix = pdist(features)

    # Perform hierarchical clustering
    linkage_matrix = hierarchy.linkage(distance_matrix, method=method)

    # Create dendrogram
    class_colors = [theme["main_color"]] + theme["bar_colors"]

    # Create a color map for class colors
    color_map = {}
    for i, label in enumerate(np.unique(labels)):
        color_map[label] = class_colors[i % len(class_colors)]

    # Create leaf colors
    leaf_colors = [color_map[label] for label in labels]

    # Create leaf labels
    if class_names is not None:
        leaf_labels = [
            class_names[label] if label < len(class_names) else f"Class {label}"
            for label in labels
        ]
    else:
        leaf_labels = [f"Class {label}" for label in labels]

    # Plot dendrogram
    dendrogram = hierarchy.dendrogram(
        linkage_matrix,
        ax=ax,
        labels=leaf_labels,
        leaf_rotation=90,
        leaf_font_size=8,
        link_color_func=lambda k: "gray",
    )

    # Adjust leaf colors
    for _i, leaf_color in enumerate(leaf_colors):
        ax.tick_params(axis="x", colors=leaf_color, which="major")

    # Set title and labels
    ax.set_title("Hierarchical Clustering Dendrogram", color=theme["text_color"])
    ax.set_xlabel("Samples/Classes", color=theme["text_color"])
    ax.set_ylabel("Distance", color=theme["text_color"])

    # Customize tick colors
    ax.tick_params(axis="y", colors=theme["text_color"])

    # Add legend for classes
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map[label],
            markersize=10,
            label=class_names[label] if label < len(class_names) else f"Class {label}",
        )
        for label in np.unique(labels)
    ]

    ax.legend(
        handles=legend_elements,
        loc="best",
        frameon=True,
        facecolor=theme["background_color"],
        edgecolor=theme["grid_color"],
    )

    plt.tight_layout()

    # Save figure if output path provided
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        plt.savefig(
            output_path,
            dpi=400,
            bbox_inches="tight",
            facecolor=theme["background_color"],
        )
        logger.info(f"Saved hierarchical clustering dendrogram to {output_path}")

    return fig


def plot_similarity_matrix(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 10),
    metric: str = "cosine",
    display_values: bool = True,
    theme: Optional[Dict] = None,
) -> Figure:
    """
    Create a heatmap showing similarity between classes based on feature vectors.

    Args:
        features: Feature vectors (n_samples, n_features)
        labels: Class labels (n_samples,)
        class_names: List of class names
        output_path: Path to save the figure
        figsize: Figure size
        metric: Similarity metric ('cosine', 'euclidean', 'correlation')
        display_values: Whether to display numerical similarity values
        theme: Theme settings to apply (uses DEFAULT_THEME if None)

    Returns:
        Matplotlib figure
    """
    # Apply theme
    theme = theme or DEFAULT_THEME
    apply_dark_theme(theme)

    from scipy.spatial.distance import pdist, squareform

    # Calculate mean feature vector for each class
    unique_labels = np.unique(labels)
    class_mean_features = {}

    for label in unique_labels:
        class_features = features[labels == label]
        if len(class_features) > 0:
            class_mean_features[label] = np.mean(class_features, axis=0)

    # Create matrix of mean features
    mean_features = np.array([class_mean_features[label] for label in unique_labels])

    # Calculate similarity matrix
    if metric == "cosine":
        # Cosine similarity ranges from -1 to 1, where 1 is most similar
        from sklearn.metrics.pairwise import cosine_similarity

        similarity_matrix = cosine_similarity(mean_features)
    elif metric == "correlation":
        # Correlation ranges from -1 to 1, where 1 is most similar
        from scipy.stats import pearsonr

        n_classes = len(unique_labels)
        similarity_matrix = np.zeros((n_classes, n_classes))

        for i in range(n_classes):
            for j in range(n_classes):
                similarity_matrix[i, j], _ = pearsonr(
                    mean_features[i], mean_features[j]
                )
    else:
        # Euclidean distance - convert to similarity
        distances = pdist(mean_features, metric="euclidean")
        distance_matrix = squareform(distances)

        # Convert distances to similarities (1 / (1 + distance))
        similarity_matrix = 1 / (1 + distance_matrix)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor=theme["background_color"])
    ax.set_facecolor(theme["background_color"])

    # Use class names for labels
    display_class_names = [
        class_names[label] if label < len(class_names) else f"Class {label}"
        for label in unique_labels
    ]

    # Create heatmap
    im = ax.imshow(
        similarity_matrix, cmap=theme["cmap"], aspect="auto", interpolation="nearest"
    )

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, pad=0.01)
    cbar.ax.set_ylabel(
        f"{metric.capitalize()} Similarity",
        rotation=-90,
        va="bottom",
        color=theme["text_color"],
    )
    cbar.ax.tick_params(colors=theme["text_color"])

    # Display values inside cells
    if display_values:
        for i in range(len(unique_labels)):
            for j in range(len(unique_labels)):
                value = similarity_matrix[i, j]
                # Use white text for dark cells, black text for light cells
                text_color = "white" if value < 0.7 else "black"
                ax.text(
                    j, i, f"{value:.2f}", ha="center", va="center", color=text_color
                )

    # Set title and labels
    ax.set_title(
        f"Class Similarity Matrix ({metric.capitalize()})", color=theme["text_color"]
    )

    # Configure ticks
    ax.set_xticks(np.arange(len(display_class_names)))
    ax.set_yticks(np.arange(len(display_class_names)))
    ax.set_xticklabels(
        display_class_names, rotation=90, fontsize=8, color=theme["text_color"]
    )
    ax.set_yticklabels(display_class_names, fontsize=8, color=theme["text_color"])

    # Customize appearance
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.tight_layout()

    # Save figure if output path provided
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        plt.savefig(
            output_path,
            dpi=400,
            bbox_inches="tight",
            facecolor=theme["background_color"],
        )
        logger.info(f"Saved similarity matrix plot to {output_path}")

    return fig


def plot_categorical(
    data: np.ndarray,
    categories: List[str],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (16, 12),  # Increased size
    title: str = "Category Counts",
    xlabel: str = "Category",
    ylabel: str = "Count",
    kind: str = "bar",
    theme: Optional[Dict] = None,
) -> Figure:
    """
    Create a plot for categorical data showing counts per category.

    Args:
        data: Array of counts per category
        categories: List of category names
        output_path: Path to save the figure
        figsize: Figure size
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        kind: Plot type ('bar' or 'pie')
        theme: Theme settings to apply (uses DEFAULT_THEME if None)

    Returns:
        Matplotlib figure
    """
    # Apply theme
    theme = theme or DEFAULT_THEME
    apply_dark_theme(theme)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor=theme["background_color"])
    ax.set_facecolor(theme["background_color"])

    # Create DataFrame for easier plotting
    df = pd.DataFrame(
        {"Category": categories[: len(data)], "Count": data[: len(categories)]}
    )

    # Sort by count in descending order
    df = df.sort_values("Count", ascending=False)

    # Plot based on kind
    if kind.lower() == "pie":
        # Create pie chart
        ax.pie(
            df["Count"],
            labels=df["Category"],
            autopct="%1.1f%%",
            colors=[
                theme["bar_colors"][i % len(theme["bar_colors"])]
                for i in range(len(df))
            ],
            textprops={"color": theme["text_color"]},
            wedgeprops={"linewidth": 1, "edgecolor": theme["background_color"]},
        )
        ax.set_title(title, color=theme["text_color"])

    else:  # Default to bar
        # Create bar chart
        bars = ax.bar(
            df["Category"],
            df["Count"],
            color=[
                theme["bar_colors"][i % len(theme["bar_colors"])]
                for i in range(len(df))
            ],
            width=0.7,
            alpha=0.8,
            edgecolor=theme["grid_color"],
        )

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                color=theme["text_color"],
            )

        # Set title and labels
        ax.set_title(title, color=theme["text_color"])
        ax.set_xlabel(xlabel, color=theme["text_color"])
        ax.set_ylabel(ylabel, color=theme["text_color"])

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.3, color=theme["grid_color"], axis="y")

        # Customize tick colors
        ax.tick_params(colors=theme["text_color"])

        # Rotate x-axis labels if we have many categories
        if len(categories) > 5:
            plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    # Save figure if output path provided
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        plt.savefig(
            output_path,
            dpi=400,
            bbox_inches="tight",
            facecolor=theme["background_color"],
        )
        logger.info(f"Saved categorical plot to {output_path}")

    return fig


def create_augmentation_grid(
    original_image: np.ndarray,
    augmented_images: List[np.ndarray],
    augmentation_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (16, 10),
    title: str = "Data Augmentation Examples",
    theme: Optional[Dict] = None,
) -> Figure:
    """
    Create a grid visualization of augmentation examples.

    Args:
        original_image: Original input image
        augmented_images: List of augmented versions of the image
        augmentation_names: List of names for each augmentation
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

    # Ensure we have same number of images and names
    n_augmentations = min(len(augmented_images), len(augmentation_names))

    # Calculate grid dimensions - original image in center of top row, then a grid of augmentations
    rows = int(np.ceil((n_augmentations + 1) / 3)) + 1
    cols = 3

    # Create figure
    fig = plt.figure(figsize=figsize, facecolor=theme["background_color"])

    # Create top row for original image
    ax_orig = fig.add_subplot(rows, cols, 2)  # Center of top row
    ax_orig.imshow(original_image)
    ax_orig.set_title("Original Image", color=theme["text_color"], fontsize=12)
    ax_orig.axis("off")

    # Hide other axes in the top row
    for i in [1, 3]:
        ax = fig.add_subplot(rows, cols, i)
        ax.axis("off")
        ax.set_facecolor(theme["background_color"])

    # Plot each augmented image
    for i in range(n_augmentations):
        ax = fig.add_subplot(rows, cols, i + cols + 1)  # Start from second row

        # Display image
        ax.imshow(augmented_images[i])

        # Add title
        ax.set_title(augmentation_names[i], color=theme["text_color"], fontsize=10)

        # Remove axis ticks
        ax.axis("off")

    # Add main title
    plt.suptitle(title, fontsize=16, color=theme["text_color"], y=0.98)

    # Adjust spacing
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    # Save figure if output path provided
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        plt.savefig(
            output_path,
            dpi=400,
            bbox_inches="tight",
            facecolor=theme["background_color"],
        )
        logger.info(f"Saved augmentation grid to {output_path}")

    return fig


# def create_analysis_dashboard(
#     dataset_info: Dict,
#     class_dist: np.ndarray,
#     class_names: List[str],
#     sample_images: Optional[np.ndarray] = None,
#     output_path: Optional[Union[str, Path]] = None,
#     figsize: Tuple[int, int] = (18, 16),  # Increased size
#     theme: Optional[Dict] = None,
# ) -> Figure:
#     """
#     Create a comprehensive dashboard for dataset analysis.

#     Args:
#         dataset_info: Dictionary with dataset information (num_samples, img_size, etc.)
#         class_dist: Array with number of samples per class
#         class_names: List of class names
#         sample_images: Optional array of sample images to display (N, H, W, C) or (N, C, H, W)
#         output_path: Path to save the figure
#         figsize: Figure size
#         theme: Theme settings to apply (uses DEFAULT_THEME if None)

#     Returns:
#         Matplotlib figure
#     """
#     # Apply theme
#     theme = theme or DEFAULT_THEME
#     apply_dark_theme(theme)

#     # Create figure with subfigures
#     fig = plt.figure(figsize=figsize, facecolor=theme["background_color"])
#     # Adjusted GridSpec: 3 rows, 4 columns
#     gs = plt.GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)

#     # --- Row 1: Class Distribution ---
#     # Bar chart (takes 2 columns)
#     ax1 = fig.add_subplot(gs[0, :2])
#     ax1.set_facecolor(theme["background_color"])
#     indices = np.arange(len(class_names))
#     bars = ax1.barh(
#         indices, class_dist, color=theme["main_color"], alpha=0.8, height=0.7
#     )
#     # Add value labels
#     for i, bar in enumerate(bars):
#         width = bar.get_width()
#         ax1.text(
#             width + 0.01 * max(class_dist),  # Adjust offset based on max value
#             bar.get_y() + bar.get_height() / 2,
#             f"{int(width)}",
#             ha="left",
#             va="center",
#             color=theme["text_color"],
#             fontsize=8,
#         )
#     ax1.set_title("Class Distribution (Counts)", color=theme["text_color"])
#     ax1.set_xlabel("Number of Samples", color=theme["text_color"])
#     ax1.set_yticks(indices)
#     ax1.set_yticklabels(class_names, color=theme["text_color"], fontsize=8)
#     ax1.tick_params(colors=theme["text_color"])
#     ax1.invert_yaxis()  # Display classes top-to-bottom
#     ax1.grid(True, axis="x", linestyle="--", alpha=0.3, color=theme["grid_color"])

#     # Pie chart (takes 2 columns)
#     ax2 = fig.add_subplot(gs[0, 2:])
#     ax2.set_facecolor(theme["background_color"])
#     # Filter out classes with zero samples for pie chart
#     non_zero_indices = class_dist > 0
#     filtered_dist = class_dist[non_zero_indices]
#     filtered_names = [class_names[i] for i, flag in enumerate(non_zero_indices) if flag]
#     pie_colors = [
#         theme["bar_colors"][i % len(theme["bar_colors"])]
#         for i in range(len(filtered_names))
#     ]
#     wedges, texts, autotexts = ax2.pie(
#         filtered_dist,
#         labels=None,  # Labels handled by legend
#         autopct="%1.1f%%",
#         startangle=90,
#         colors=pie_colors,
#         pctdistance=0.85,  # Move percentage text inside wedges
#         wedgeprops=dict(width=0.4, edgecolor=theme["background_color"]),
#     )
#     for text in autotexts:
#         text.set_color("white")  # White text for percentages
#         text.set_fontsize(9)
#     ax2.set_title("Class Distribution (%)", color=theme["text_color"])
#     # Add legend outside the pie chart
#     ax2.legend(
#         wedges,
#         filtered_names,
#         title="Classes",
#         loc="center left",
#         bbox_to_anchor=(1, 0, 0.5, 1),
#         frameon=False,
#         fontsize=8,
#         title_fontsize=10,
#         labelcolor=theme["text_color"],
#     )

#     # --- Row 2: Dataset Info & Sample Images ---
#     # Dataset info text (takes 1 column)
#     ax3 = fig.add_subplot(gs[1, 0])
#     ax3.set_facecolor(theme["background_color"])
#     ax3.axis("off")
#     info_text = "Dataset Information:\n\n"
#     for key, value in dataset_info.items():
#         # Format keys nicely
#         key_formatted = key.replace("_", " ").title()
#         info_text += f"• {key_formatted}: {value}\n"
#     ax3.text(
#         0.01,
#         0.95,
#         info_text,
#         transform=ax3.transAxes,
#         fontsize=10,
#         verticalalignment="top",
#         color=theme["text_color"],
#         bbox=dict(
#             boxstyle="round,pad=0.3", fc=theme["grid_color"], alpha=0.2, ec="none"
#         ),
#     )

#     # Sample images (takes 3 columns)
#     if sample_images is not None and len(sample_images) > 0:
#         n_samples_to_show = min(9, len(sample_images))  # Show up to 9 samples
#         sample_indices = np.random.choice(
#             len(sample_images), n_samples_to_show, replace=False
#         )
#         samples = sample_images[sample_indices]

#         # Create a 3x3 grid within the allocated space (gs[1, 1:])
#         gs_samples = gs[1, 1:].subgridspec(3, 3, wspace=0.1, hspace=0.1)

#         for i in range(n_samples_to_show):
#             ax_img = fig.add_subplot(gs_samples[i])
#             img = samples[i]
#             # Convert CHW to HWC if needed
#             if img.ndim == 3 and img.shape[0] in [1, 3]:
#                 img = np.transpose(img, (1, 2, 0))
#             # Handle grayscale
#             if img.ndim == 3 and img.shape[-1] == 1:
#                 img = np.repeat(img, 3, axis=-1)
#             elif img.ndim == 2:
#                 img = np.stack([img] * 3, axis=-1)
#             # Normalize/Scale image for display
#             img = img.astype(np.float32)
#             if img.max() > 1.0:
#                 img = img / 255.0
#             img = np.clip(img, 0, 1)

#             ax_img.imshow(img, interpolation="nearest")
#             ax_img.set_xticks([])
#             ax_img.set_yticks([])
#             ax_img.set_facecolor(theme["background_color"])
#             # Add border
#             for spine in ax_img.spines.values():
#                 spine.set_edgecolor(theme["grid_color"])
#                 spine.set_linewidth(1)

#         # Add title for sample images section
#         ax_sample_title = fig.add_subplot(gs[1, 1:])
#         ax_sample_title.set_facecolor(theme["background_color"])
#         ax_sample_title.axis("off")
#         ax_sample_title.set_title("Sample Images", color=theme["text_color"], y=1.02)
#     else:
#         # Placeholder if no images
#         ax_no_img = fig.add_subplot(gs[1, 1:])
#         ax_no_img.set_facecolor(theme["background_color"])
#         ax_no_img.axis("off")
#         ax_no_img.text(
#             0.5,
#             0.5,
#             "No sample images provided",
#             ha="center",
#             va="center",
#             color=theme["text_color"],
#         )

#     # --- Row 3: Placeholder for future plots ---
#     # Example: Could add feature space plot or other analysis here
#     ax_placeholder1 = fig.add_subplot(gs[2, :2])
#     ax_placeholder1.set_facecolor(theme["background_color"])
#     ax_placeholder1.axis("off")
#     ax_placeholder1.text(
#         0.5,
#         0.5,
#         "Future Plot Area 1",
#         ha="center",
#         va="center",
#         color=theme["text_color"],
#     )

#     ax_placeholder2 = fig.add_subplot(gs[2, 2:])
#     ax_placeholder2.set_facecolor(theme["background_color"])
#     ax_placeholder2.axis("off")
#     ax_placeholder2.text(
#         0.5,
#         0.5,
#         "Future Plot Area 2",
#         ha="center",
#         va="center",
#         color=theme["text_color"],
#     )

#     # Adjust layout and add overall title
#     plt.suptitle(
#         "Dataset Analysis Dashboard", fontsize=18, color=theme["text_color"], y=0.99
#     )
#     # Use tight_layout first, then adjust subplots if needed
#     fig.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust rect to prevent title overlap

#     # Save figure if output path provided
#     if output_path:
#         output_path = Path(output_path)
#         ensure_dir(output_path.parent)
#         plt.savefig(
#             output_path,
#             dpi=400,
#             bbox_inches="tight",
#             facecolor=theme["background_color"],
#         )
#         logger.info(f"Saved analysis dashboard to {output_path}")

#     return fig
