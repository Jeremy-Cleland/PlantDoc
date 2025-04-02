#!/usr/bin/env python
"""
Generate augmentation examples for visualization.
This script creates visualizations of various data augmentation techniques
used in training and saves them to the augmentation_examples directory.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from utils.logger import get_logger
from utils.paths import ensure_dir

logger = get_logger(__name__)


def create_augmentation_grid(
    image_path: str,
    output_path: Path,
    figsize=(15, 15),
):
    """
    Generate and save a grid of augmentation examples for a single image.

    Args:
        image_path: Path to the original image
        output_path: Path to save the augmentation grid
        figsize: Figure size for the grid
    """
    try:
        # Load original image
        image = Image.open(image_path).convert("RGB")

        # Create augmentations
        augmentations = [
            ("Original", image),
            ("Horizontal Flip", image.transpose(Image.FLIP_LEFT_RIGHT)),
            ("Vertical Flip", image.transpose(Image.FLIP_TOP_BOTTOM)),
            ("Rotation 30°", image.rotate(30, resample=Image.BICUBIC, expand=True)),
            ("Brightness +30%", ImageEnhance.Brightness(image).enhance(1.3)),
            ("Contrast +30%", ImageEnhance.Contrast(image).enhance(1.3)),
            ("Color +50%", ImageEnhance.Color(image).enhance(1.5)),
            ("Gaussian Blur", image.filter(ImageFilter.GaussianBlur(radius=2))),
            (
                "Crop Center",
                image.crop(
                    (
                        image.width // 4,
                        image.height // 4,
                        3 * image.width // 4,
                        3 * image.height // 4,
                    )
                ).resize((image.width, image.height), Image.BICUBIC),
            ),
        ]

        # Create figure with consistent square grid
        rows = 3
        cols = 3
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        # Set a PlantDoc theme with green styling
        plt.style.use('default')
        fig.patch.set_facecolor('#f8f9fa')  # Light background
        fig.suptitle("Data Augmentation Examples", fontsize=22, fontweight='bold', color='#2e7d32')
        
        # Add a subtitle
        plt.figtext(0.5, 0.92, "Training techniques to improve model generalization", 
                   ha='center', fontsize=14, color='#1b5e20')

        # Add each augmentation to the grid
        for i, (name, aug_img) in enumerate(augmentations):
            if i < rows * cols:
                row = i // cols
                col = i % cols
                
                # Create subplot with consistent aspect ratio
                ax = axes[row, col]
                ax.imshow(aug_img)
                ax.set_title(name, fontsize=14, pad=10, fontweight='medium', color='#2e7d32')
                ax.axis("off")
                
                # Add a subtle border around each image
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color('#dddddd')
                    spine.set_linewidth(1)

        # Ensure layout is tight but with consistent spacing
        plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=3.0)
        plt.subplots_adjust(top=0.88)  # Adjust to make room for title
        
        # Save figure with high resolution
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()

        logger.info(f"Created augmentation grid: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error creating augmentation grid: {e}")
        return False


def generate_augmentation_examples(
    experiment_dir: Path,
    num_images: int = 5,
):
    """
    Generate augmentation examples for training data.

    Args:
        experiment_dir: Path to experiment directory
        num_images: Number of images to process
    """
    # Create output directory
    output_dir = experiment_dir / "augmentation_examples"
    ensure_dir(output_dir)

    # Try to find image files
    # First check in common data directories
    potential_data_dirs = [
        Path("data"),
        Path("data/train"),
        Path("data/val"),
        Path("data/test"),
        Path("data/plant_disease"),
        Path("data/images"),
    ]

    # Find first existing directory
    data_dir = None
    for dir_path in potential_data_dirs:
        if dir_path.exists() and dir_path.is_dir():
            file_count = sum(1 for _ in dir_path.glob("**/*.jpg")) + sum(
                1 for _ in dir_path.glob("**/*.png")
            )
            if file_count > 0:
                data_dir = dir_path
                logger.info(f"Found {file_count} images in {data_dir}")
                break

    if data_dir is None:
        logger.warning("No suitable data directory found. Creating random images.")
        # Create random images if no real data found
        create_random_images(output_dir, num_images)
        return

    # Collect image files
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_files.extend(list(data_dir.glob(f"**/{ext}")))

    if not image_files:
        logger.warning("No image files found. Creating random images.")
        create_random_images(output_dir, num_images)
        return

    # Select random subset
    if len(image_files) > num_images:
        image_files = list(np.random.choice(image_files, num_images, replace=False))

    # Create augmentation grid for each image
    success_count = 0
    for i, img_path in enumerate(image_files):
        output_path = output_dir / f"augmentation_example_{i + 1}.png"
        if create_augmentation_grid(img_path, output_path):
            success_count += 1

    logger.info(f"Generated {success_count} augmentation example grids in {output_dir}")


def create_random_images(output_dir: Path, num_images: int = 5):
    """
    Create random images when no real data is available.

    Args:
        output_dir: Directory to save images
        num_images: Number of images to create
    """
    for i in range(num_images):
        # Create a random noise image
        random_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(random_img)

        # Save augmentation grid
        output_path = output_dir / f"random_augmentation_{i + 1}.png"
        create_augmentation_grid(img, output_path)

    logger.info(f"Generated {num_images} augmentation examples from random data")


def main():
    parser = argparse.ArgumentParser(description="Generate augmentation examples")
    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        required=True,
        help="Path to experiment directory",
    )
    parser.add_argument(
        "--num-images",
        "-n",
        type=int,
        default=5,
        help="Number of images to process",
    )

    args = parser.parse_args()

    # Resolve experiment directory
    experiment_dir = Path(args.experiment)
    if not experiment_dir.is_absolute():
        experiment_dir = Path.cwd() / experiment_dir

    if not experiment_dir.exists():
        logger.error(f"Experiment directory {experiment_dir} does not exist")
        return

    generate_augmentation_examples(
        experiment_dir=experiment_dir,
        num_images=args.num_images,
    )


if __name__ == "__main__":
    main()
