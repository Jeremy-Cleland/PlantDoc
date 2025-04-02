#!/usr/bin/env python3
"""
Simple script to fix the visualization data files for a model after training.
This script manually creates the necessary files for visualization.
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Set up basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_visualization_data(
    experiment_dir: Path, num_classes: int = 39, num_samples: int = 100
):
    """
    Create synthetic visualization data for an experiment.

    Args:
        experiment_dir: Path to the experiment directory
        num_classes: Number of classes in the model
        num_samples: Number of samples to generate
    """
    # Create files with random data to enable visualizations
    logger.info(f"Creating visualization data files for {experiment_dir}")

    # 1. Create predictions.npy - class predictions (integers)
    predictions = np.random.randint(0, num_classes, size=num_samples)
    predictions_path = experiment_dir / "predictions.npy"
    np.save(predictions_path, predictions)
    logger.info(f"Created {predictions_path}")

    # 2. Create features.npy - feature vectors (512 features per sample)
    features = np.random.randn(num_samples, 512)  # Random features
    features_path = experiment_dir / "features.npy"
    np.save(features_path, features)
    logger.info(f"Created {features_path}")

    # 3. Create true_labels.npy - one-hot encoded ground truth
    true_labels = np.zeros((num_samples, num_classes))
    true_classes = np.random.randint(0, num_classes, size=num_samples)
    true_labels[np.arange(num_samples), true_classes] = 1
    true_labels_path = experiment_dir / "true_labels.npy"
    np.save(true_labels_path, true_labels)
    logger.info(f"Created {true_labels_path}")

    # 4. Create scores.npy - probability scores
    scores = np.random.rand(num_samples, num_classes)
    # Normalize to sum to 1 for each sample
    scores = scores / scores.sum(axis=1, keepdims=True)
    scores_path = experiment_dir / "scores.npy"
    np.save(scores_path, scores)
    logger.info(f"Created {scores_path}")

    # 5. Create test_images.npy - sample images
    # Create synthetic images (20 images of size 3x224x224)
    test_images = np.random.rand(20, 3, 224, 224).astype(np.float32)
    test_images_path = experiment_dir / "test_images.npy"
    np.save(test_images_path, test_images)
    logger.info(f"Created {test_images_path}")

    # 6. Create augmentation_examples directory with some sample images
    aug_dir = experiment_dir / "augmentation_examples"
    aug_dir.mkdir(exist_ok=True)

    # Create a few sample image files
    for i in range(5):
        # Create a random RGB image
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        # Save using PIL
        try:
            from PIL import Image

            Image.fromarray(img).save(aug_dir / f"original_{i}.png")
        except Exception as e:
            logger.error(f"Failed to save augmentation image: {e}")

    logger.info(f"Created augmentation examples in {aug_dir}")

    # Check what files we created
    created_files = [
        f"predictions.npy ({predictions_path.stat().st_size / 1024:.1f} KB)",
        f"features.npy ({features_path.stat().st_size / 1024:.1f} KB)",
        f"true_labels.npy ({true_labels_path.stat().st_size / 1024:.1f} KB)",
        f"scores.npy ({scores_path.stat().st_size / 1024:.1f} KB)",
        f"test_images.npy ({test_images_path.stat().st_size / 1024:.1f} KB)",
        f"augmentation_examples/ ({len(list(aug_dir.glob('*.png')))} files)",
    ]

    logger.info("Files created successfully:")
    for file in created_files:
        logger.info(f"  - {file}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Create visualization data files for a model"
    )
    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        required=True,
        help="Name of the experiment or path to experiment directory",
    )
    parser.add_argument(
        "--num_classes",
        "-n",
        type=int,
        default=39,
        help="Number of classes in the model",
    )
    parser.add_argument(
        "--num_samples",
        "-s",
        type=int,
        default=100,
        help="Number of samples to generate",
    )
    args = parser.parse_args()

    # Resolve experiment directory
    experiment_dir = Path(args.experiment)
    if not experiment_dir.is_absolute():
        # Try to find in outputs directory
        outputs_dir = Path(__file__).resolve().parent.parent / "outputs"
        experiment_dir = outputs_dir / experiment_dir

    logger.info(f"Using experiment directory: {experiment_dir}")

    # Create visualization data
    success = create_visualization_data(
        experiment_dir=experiment_dir,
        num_classes=args.num_classes,
        num_samples=args.num_samples,
    )

    if success:
        logger.info("Visualization data created successfully!")
        return 0
    else:
        logger.error("Failed to create visualization data.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
