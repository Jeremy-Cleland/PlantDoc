"""
Confusion flow visualization module for visualizing misclassification patterns.

This module provides tools to create flow diagrams that show how misclassifications
occur between classes, making it easier to understand model weaknesses.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

from utils.logger import get_logger
from utils.paths import ensure_dir

logger = get_logger(__name__)


class ConfusionFlow:
    """
    Class for generating confusion flow diagrams that visualize misclassification patterns.

    This visualization shows how predictions flow between classes, highlighting
    which classes are commonly confused with each other.

    Args:
        confusion_matrix: The confusion matrix to visualize
        class_names: List of class names
        min_flow_threshold: Minimum number of misclassifications to show a flow
        max_flows: Maximum number of flows to display
    """

    def __init__(
        self,
        confusion_matrix: np.ndarray,
        class_names: List[str],
        min_flow_threshold: int = 3,
        max_flows: int = 50,
    ):
        self.cm = confusion_matrix
        self.class_names = class_names
        self.min_flow_threshold = min_flow_threshold
        self.max_flows = max_flows

        # Calculate derived properties
        self.num_classes = len(class_names)

        # Validate confusion matrix
        if self.cm.shape != (self.num_classes, self.num_classes):
            raise ValueError(
                f"Confusion matrix shape {self.cm.shape} doesn't match "
                f"number of classes {self.num_classes}"
            )

        # Calculate total samples and class counts
        self.total_samples = np.sum(self.cm)
        self.class_counts = np.sum(self.cm, axis=1)

        # Calculate error rates
        self.error_rates = 1.0 - np.diag(self.cm) / self.class_counts

        # Calculate the misclassification flows
        self.flows = self._calculate_flows()

    def _calculate_flows(self) -> List[Dict[str, Union[int, float, str]]]:
        """
        Calculate the misclassification flows from the confusion matrix.

        Returns:
            List of flow dictionaries with source, target, and value
        """
        flows = []

        # Loop through confusion matrix to find misclassifications
        for true_idx in range(self.num_classes):
            for pred_idx in range(self.num_classes):
                # Skip correct classifications (diagonal)
                if true_idx == pred_idx:
                    continue

                # Get the number of misclassifications
                flow_value = self.cm[true_idx, pred_idx]

                # Skip flows below threshold
                if flow_value < self.min_flow_threshold:
                    continue

                # Calculate the percentage of the class that was misclassified
                flow_pct = (
                    flow_value / self.class_counts[true_idx]
                    if self.class_counts[true_idx] > 0
                    else 0
                )

                # Add to flows list
                flows.append(
                    {
                        "source": true_idx,
                        "target": pred_idx,
                        "value": int(flow_value),
                        "percentage": float(flow_pct),
                        "source_name": self.class_names[true_idx],
                        "target_name": self.class_names[pred_idx],
                    }
                )

        # Sort flows by value (descending)
        flows.sort(key=lambda x: x["value"], reverse=True)

        # Limit number of flows
        if len(flows) > self.max_flows:
            flows = flows[: self.max_flows]

        return flows

    def plot(
        self,
        output_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (16, 12),
        cmap: str = "YlOrRd",
        node_color: str = "#4CAF50",
        arrow_cmap: str = "YlOrRd",
        title: str = "Misclassification Flow Diagram",
        layout: str = "circular",
        label_fontsize: int = 10,
        title_fontsize: int = 18,
        use_dark_theme: bool = True,
        show_values: bool = True,
        arrow_scale: float = 1.0,
        node_scale: float = 1.0,
        fig_background_color: str = "#121212",
        text_color: str = "#FFFFFF",
    ) -> plt.Figure:
        """
        Generate the confusion flow diagram.

        Args:
            output_path: Path to save the diagram
            figsize: Figure size (width, height)
            cmap: Colormap for error rates
            node_color: Color for nodes
            arrow_cmap: Colormap for flow arrows
            title: Plot title
            layout: Layout type ('circular' or 'force')
            label_fontsize: Font size for labels
            title_fontsize: Font size for title
            use_dark_theme: Whether to use dark theme
            show_values: Whether to show flow values
            arrow_scale: Scaling factor for arrow width
            node_scale: Scaling factor for node size
            fig_background_color: Background color of the figure
            text_color: Text color

        Returns:
            Matplotlib figure object
        """
        # Set dark theme if requested
        if use_dark_theme:
            plt.style.use("dark_background")
            plt.rcParams.update(
                {
                    "figure.facecolor": fig_background_color,
                    "axes.facecolor": fig_background_color,
                    "text.color": text_color,
                }
            )

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Skip if no significant flows
        if not self.flows:
            ax.text(
                0.5,
                0.5,
                "No significant misclassification flows to display",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=14,
                color=text_color,
            )
            plt.title(title, fontsize=title_fontsize)

            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                logger.info(f"Saved empty confusion flow diagram to {output_path}")

            return fig

        # Calculate node positions based on layout
        node_positions = self._get_node_positions(layout)

        # Setup colormaps
        error_norm = plt.Normalize(0, min(1.0, max(self.error_rates)))
        flow_norm = plt.Normalize(0, max(flow["value"] for flow in self.flows))

        # Draw nodes (classes)
        node_colors = plt.cm.get_cmap(cmap)(error_norm(self.error_rates))

        # Adjust node size based on class counts
        max_count = max(self.class_counts)
        min_size = 500 * node_scale
        max_size = 2000 * node_scale

        for i, (pos, err, count, name) in enumerate(
            zip(node_positions, self.error_rates, self.class_counts, self.class_names)
        ):
            # Calculate node size based on count
            size = min_size + (max_size - min_size) * (count / max_count)

            # Draw node
            ax.scatter(
                pos[0],
                pos[1],
                s=size,
                color=node_colors[i],
                alpha=0.8,
                edgecolors="white",
                linewidth=1,
            )

            # Truncate long class names
            if len(name) > 15:
                display_name = name[:12] + "..."
            else:
                display_name = name

            # Add label with error rate
            error_pct = err * 100
            label = f"{display_name}\n({error_pct:.1f}% error)"

            # Draw label with outline for better visibility
            ax.text(
                pos[0],
                pos[1],
                label,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=label_fontsize,
                fontweight="bold",
                color=text_color,
                path_effects=[
                    plt.matplotlib.patheffects.withStroke(
                        linewidth=2, foreground="black"
                    )
                ],
            )

        # Draw arrows (flows)
        arrow_cmap_fn = plt.cm.get_cmap(arrow_cmap)

        # Sort flows by value for better visualization
        sorted_flows = sorted(self.flows, key=lambda f: f["value"])

        for flow in sorted_flows:
            source_idx = flow["source"]
            target_idx = flow["target"]
            value = flow["value"]
            pct = flow["percentage"] * 100

            # Calculate arrow parameters
            source_pos = node_positions[source_idx]
            target_pos = node_positions[target_idx]

            # Skip if positions are too close
            if np.linalg.norm(np.array(source_pos) - np.array(target_pos)) < 0.1:
                continue

            # Normalize positions
            dx = target_pos[0] - source_pos[0]
            dy = target_pos[1] - source_pos[1]
            dist = np.sqrt(dx**2 + dy**2)
            nx = dx / dist
            ny = dy / dist

            # Calculate arrow width based on flow value
            width = 1 + 10 * flow_norm(value) * arrow_scale

            # Draw curved arrow (more visually appealing)
            arrow = FancyArrowPatch(
                source_pos,
                target_pos,
                connectionstyle="arc3,rad=0.2",
                arrowstyle=f"simple,head_width={2 * width},head_length={2 * width}",
                linewidth=width,
                color=arrow_cmap_fn(flow_norm(value)),
                alpha=0.8,
                zorder=1,
            )
            ax.add_patch(arrow)

            # Add flow value if requested
            if show_values:
                # Calculate midpoint of the arrow
                mid_x = (source_pos[0] + target_pos[0]) / 2
                mid_y = (source_pos[1] + target_pos[1]) / 2

                # Offset it slightly to avoid overlapping with the arrow
                offset_factor = 0.15
                mid_x += offset_factor * (source_pos[1] - target_pos[1]) / dist
                mid_y += offset_factor * (target_pos[0] - source_pos[0]) / dist

                # Show both absolute value and percentage
                flow_text = f"{value} ({pct:.1f}%)"

                # Add the text with outline
                ax.text(
                    mid_x,
                    mid_y,
                    flow_text,
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=max(8, min(12, label_fontsize - 2)),
                    fontweight="bold",
                    color="white",
                    bbox=dict(facecolor="black", alpha=0.5, boxstyle="round"),
                    path_effects=[
                        plt.matplotlib.patheffects.withStroke(
                            linewidth=2, foreground="black"
                        )
                    ],
                )

        # Add title and remove axes
        plt.title(title, fontsize=title_fontsize)
        ax.set_aspect("equal")
        ax.axis("off")

        # Add legend for error rates
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=error_norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
        cbar.set_label("Class Error Rate", color=text_color)

        # Add legend for arrow widths
        arrow_sm = plt.cm.ScalarMappable(cmap=arrow_cmap, norm=flow_norm)
        arrow_sm.set_array([])
        arrow_cbar = plt.colorbar(arrow_sm, ax=ax, shrink=0.6, location="right")
        arrow_cbar.set_label("Flow Size", color=text_color)

        # Adjust layout
        plt.tight_layout()

        # Save if output path is provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved confusion flow diagram to {output_path}")

        return fig

    def _get_node_positions(
        self, layout: str = "circular"
    ) -> List[Tuple[float, float]]:
        """
        Calculate node positions based on layout type.

        Args:
            layout: Layout type ('circular' or 'force')

        Returns:
            List of (x, y) positions for each node
        """
        if layout == "circular":
            # Place nodes in a circle
            positions = []
            n = self.num_classes

            for i in range(n):
                angle = 2 * np.pi * i / n
                x = np.cos(angle)
                y = np.sin(angle)
                positions.append((x, y))

            return positions
        else:
            # Default to circular layout
            return self._get_node_positions("circular")


def generate_confusion_flow(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
    min_flow_threshold: int = 3,
    max_flows: int = 50,
    figsize: Tuple[int, int] = (16, 12),
    cmap: str = "YlOrRd",
    arrow_cmap: str = "YlOrRd",
    title: str = "Misclassification Flow Diagram",
    layout: str = "circular",
    use_dark_theme: bool = True,
    show_values: bool = True,
    arrow_scale: float = 1.0,
    node_scale: float = 1.0,
) -> plt.Figure:
    """
    Generate a confusion flow diagram from a confusion matrix.

    Args:
        confusion_matrix: The confusion matrix to visualize
        class_names: List of class names
        output_path: Path to save the diagram
        min_flow_threshold: Minimum number of misclassifications to show a flow
        max_flows: Maximum number of flows to display
        figsize: Figure size (width, height)
        cmap: Colormap for error rates
        arrow_cmap: Colormap for flow arrows
        title: Plot title
        layout: Layout type ('circular' or 'force')
        use_dark_theme: Whether to use dark theme
        show_values: Whether to show flow values
        arrow_scale: Scaling factor for arrow width
        node_scale: Scaling factor for node size

    Returns:
        Matplotlib figure object
    """
    flow = ConfusionFlow(
        confusion_matrix=confusion_matrix,
        class_names=class_names,
        min_flow_threshold=min_flow_threshold,
        max_flows=max_flows,
    )

    return flow.plot(
        output_path=output_path,
        figsize=figsize,
        cmap=cmap,
        arrow_cmap=arrow_cmap,
        title=title,
        layout=layout,
        use_dark_theme=use_dark_theme,
        show_values=show_values,
        arrow_scale=arrow_scale,
        node_scale=node_scale,
    )


def copy_dataset_analysis_to_model(
    source_dir: Union[str, Path],
    dest_dir: Union[str, Path],
    output_subdir: str = "dataset_analysis",
) -> bool:
    """
    Copy dataset analysis files to the model output directory.

    Args:
        source_dir: Source directory with dataset analysis files
        dest_dir: Destination directory for model
        output_subdir: Subdirectory to create in the model directory

    Returns:
        True if successful, False otherwise
    """
    import shutil

    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)

    # Verify source directory exists
    if not source_dir.exists():
        logger.error(f"Source directory does not exist: {source_dir}")
        return False

    # Create destination directory
    dest_subdir = dest_dir / "reports" / "plots" / output_subdir
    ensure_dir(dest_subdir)

    # Check common analysis directories
    analysis_paths = [
        source_dir / "analysis",
        source_dir / "analysis" / "figures",
        source_dir / "analysis" / "plots",
    ]

    # Find available analysis images
    analysis_images = []
    for path in analysis_paths:
        if path.exists() and path.is_dir():
            for ext in ["*.png", "*.jpg"]:
                analysis_images.extend(list(path.glob(ext)))

    if not analysis_images:
        logger.warning(f"No analysis images found in {source_dir}")
        return False

    # Copy files
    copied_count = 0
    for image_path in analysis_images:
        try:
            shutil.copy2(image_path, dest_subdir / image_path.name)
            copied_count += 1
        except Exception as e:
            logger.error(f"Failed to copy {image_path}: {e}")

    logger.info(f"Copied {copied_count} dataset analysis images to {dest_subdir}")
    return copied_count > 0
