#!/usr/bin/env python
"""
Test script for enhanced visualizations.

This script checks if the necessary files are available in the experiment directory
and generates enhanced visualizations.
"""

import argparse
from pathlib import Path

import numpy as np

from core.visualization.visualization import (
    create_analysis_dashboard,
    create_image_grid,
    plot_hierarchical_clustering,
    plot_scatter,
    plot_similarity_matrix,
    plot_training_curves,
    plot_training_time_analysis,
)
from reports.generate_plots import generate_plots_for_report
from utils.logger import get_logger
from utils.paths import ensure_dir

logger = get_logger(__name__)


def check_visualization_files(experiment_dir: Path) -> bool:
    """
    Check if the necessary files for enhanced visualizations are available.

    Args:
        experiment_dir: Path to the experiment directory

    Returns:
        True if all necessary files are available, False otherwise
    """
    # List of required files
    required_files = [
        "metrics.json",
        "confusion_matrix.npy",
        "history.json",
        "class_names.txt",
    ]

    # List of optional files for enhanced visualizations
    enhanced_visualization_files = [
        "predictions.npy",
        "features.npy",
        "true_labels.npy",
        "scores.npy",
        "test_images.npy",
    ]

    # Check if required files are available
    for file in required_files:
        if not (experiment_dir / file).exists():
            logger.warning(f"Required file {file} not available")
            return False

    # Check if enhanced visualization files are available
    available_enhanced_files = []
    for file in enhanced_visualization_files:
        if (experiment_dir / file).exists():
            available_enhanced_files.append(file)

    # Check if augmentation examples directory exists
    if (experiment_dir / "augmentation_examples").exists():
        available_enhanced_files.append("augmentation_examples/")

    if available_enhanced_files:
        logger.info(
            f"Available enhanced visualization files: {', '.join(available_enhanced_files)}"
        )
    else:
        logger.warning("No enhanced visualization files available")

    # Return True only if some enhanced visualization files are available
    return len(available_enhanced_files) > 0


def generate_additional_visualizations(experiment_dir: Path, output_dir: Path) -> None:
    """
    Generate additional visualizations beyond those in generate_plots_for_report.

    Args:
        experiment_dir: Path to the experiment directory
        output_dir: Directory to save the visualizations
    """
    # Ensure output directory exists
    ensure_dir(output_dir)

    # Load class names
    class_names_path = experiment_dir / "class_names.txt"
    if class_names_path.exists():
        with open(class_names_path) as f:
            class_names = [line.strip() for line in f if line.strip()]
    else:
        class_names = [
            f"Class {i}" for i in range(39)
        ]  # Default to 39 plant disease classes

    # Load history
    history_path = experiment_dir / "history.json"
    if history_path.exists():
        import json

        with open(history_path) as f:
            history = json.load(f)

        # Generate training curves visualization
        plot_training_curves(
            history=history,
            output_path=output_dir / "training_curves.png",
            metrics=None,  # Auto-detect metrics
        )
        logger.info("Generated training curves visualization")

        # Generate training time analysis if time data is available
        time_key = "time" if "time" in history else "train_time"
        if time_key in history:
            plot_training_time_analysis(
                epoch_times=history[time_key],
                output_path=output_dir / "training_time_analysis.png",
            )
            logger.info("Generated training time analysis visualization")

    # Load features and labels if available
    features_path = experiment_dir / "features.npy"
    true_labels_path = experiment_dir / "true_labels.npy"

    if features_path.exists() and true_labels_path.exists():
        features = np.load(features_path)
        true_labels = np.load(true_labels_path)

        # Convert one-hot encoded labels if needed
        if true_labels.ndim > 1:
            true_labels = np.argmax(true_labels, axis=1)

        # Generate scatter plot of first two PCA components
        if len(features) > 0:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(features)

            plot_scatter(
                x=pca_result[:, 0],
                y=pca_result[:, 1],
                labels=true_labels,
                class_names=class_names,
                output_path=output_dir / "pca_scatter.png",
                title="PCA Feature Space",
                xlabel="PC1",
                ylabel="PC2",
            )
            logger.info("Generated PCA scatter plot")

            # Generate hierarchical clustering visualization
            plot_hierarchical_clustering(
                features=features,
                labels=true_labels,
                class_names=class_names,
                output_path=output_dir / "hierarchical_clustering.png",
                max_samples=100,  # Limit samples for better visualization
            )
            logger.info("Generated hierarchical clustering visualization")

            # Generate similarity matrix
            plot_similarity_matrix(
                features=features,
                labels=true_labels,
                class_names=class_names,
                output_path=output_dir / "similarity_matrix.png",
                metric="cosine",
            )
            logger.info("Generated similarity matrix visualization")

    # Load test images if available
    test_images_path = experiment_dir / "test_images.npy"
    if test_images_path.exists() and true_labels_path.exists():
        test_images = np.load(test_images_path)
        true_labels = np.load(true_labels_path)

        # Convert one-hot encoded labels if needed
        if true_labels.ndim > 1:
            true_labels = np.argmax(true_labels, axis=1)

        # Generate image grid
        create_image_grid(
            images=test_images[:25],  # Limit to 25 images
            labels=true_labels[:25] if len(true_labels) >= 25 else true_labels,
            class_names=class_names,
            output_path=output_dir / "image_grid.png",
            title="Test Images Sample",
            n_images=25,
        )
        logger.info("Generated image grid visualization")

        # Generate analysis dashboard if we have class distribution info
        # Create dummy dataset info and distribution for demonstration
        unique_labels, counts = np.unique(true_labels, return_counts=True)
        class_dist = np.zeros(len(class_names))
        for label, count in zip(unique_labels, counts):
            if label < len(class_dist):
                class_dist[label] = count

        dataset_info = {
            "Total samples": len(true_labels),
            "Number of classes": len(class_names),
            "Image dimensions": f"{test_images.shape[1]}x{test_images.shape[2]}",
            "Color channels": test_images.shape[3] if test_images.ndim > 3 else 1,
        }

        create_analysis_dashboard(
            dataset_info=dataset_info,
            class_dist=class_dist,
            class_names=class_names,
            sample_images=test_images[:8],  # Sample of 8 images
            output_path=output_dir / "analysis_dashboard.png",
        )
        logger.info("Generated analysis dashboard visualization")


def main():
    """
    Main entry point for testing enhanced visualizations.
    """
    parser = argparse.ArgumentParser(description="Test enhanced visualizations")
    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        default="cbam_only_resnet18_v3",
        help="Name of the experiment or path to experiment directory",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force regeneration of plots even if enhanced visualization files are not available",
    )
    parser.add_argument(
        "--additional",
        "-a",
        action="store_true",
        help="Generate additional visualizations beyond the standard set",
    )

    args = parser.parse_args()

    # Resolve experiment directory
    experiment_dir = Path(args.experiment)
    if not experiment_dir.is_absolute():
        # Try to find in outputs directory
        outputs_dir = Path.cwd() / "outputs"
        if not outputs_dir.exists():
            # Try parent directory
            outputs_dir = Path.cwd().parent / "outputs"
        experiment_dir = outputs_dir / experiment_dir

    # Check if experiment directory exists
    if not experiment_dir.exists():
        logger.error(f"Experiment directory {experiment_dir} does not exist")
        return

    # Check if necessary files are available
    if check_visualization_files(experiment_dir) or args.force:
        # Generate plots
        output_dir = experiment_dir / "reports" / "plots"
        ensure_dir(output_dir)
        generate_plots_for_report(experiment_dir, output_dir)
        logger.info(f"Generated enhanced visualizations in {output_dir}")

        # Generate additional visualizations if requested
        if args.additional:
            additional_output_dir = output_dir / "additional"
            ensure_dir(additional_output_dir)
            generate_additional_visualizations(experiment_dir, additional_output_dir)
            logger.info(
                f"Generated additional visualizations in {additional_output_dir}"
            )
    else:
        logger.error(
            "Enhanced visualization files not available, cannot generate enhanced visualizations"
        )


if __name__ == "__main__":
    main()
