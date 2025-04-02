#!/usr/bin/env python
"""
Generate augmentation examples for visualization.
This script creates visualizations of various data augmentation techniques
used in training and saves them to the augmentation_examples directory.
"""

import argparse
import os
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from utils.logger import get_logger

logger = get_logger(__name__)


def create_augmentation_grid(
    image_path_or_img: Union[str, Image.Image],
    output_path: Path,
    figsize=(15, 15),
    theme="dark",
):
    """
    Generate and save a grid of augmentation examples for a single image.

    Args:
        image_path_or_img: Path to the original image or PIL Image object
        output_path: Path to save the augmentation grid
        figsize: Figure size for the grid
        theme: Theme to use for visualization ('dark' or 'light')
    """
    try:
        # Load original image or use provided PIL Image
        if isinstance(image_path_or_img, str):
            image = Image.open(image_path_or_img).convert("RGB")
        else:
            # Ensure we're working with an RGB image
            image = image_path_or_img.convert("RGB")

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

        # Apply dark theme for visualizations
        from core.visualization.base_visualization import (
            DEFAULT_THEME,
            apply_dark_theme,
        )

        theme_settings = DEFAULT_THEME.copy()
        apply_dark_theme(theme_settings)

        # Create figure with consistent square grid
        rows = 3
        cols = 3
        fig, axes = plt.subplots(
            rows, cols, figsize=figsize, facecolor=theme_settings["background_color"]
        )

        # Set a dark theme with green styling
        fig.patch.set_facecolor(theme_settings["background_color"])
        fig.suptitle(
            "Data Augmentation Examples",
            fontsize=22,
            fontweight="bold",
            color=theme_settings["text_color"],
        )

        # Add a subtitle
        plt.figtext(
            0.5,
            0.92,
            "Training techniques to improve model generalization",
            ha="center",
            fontsize=14,
            color=theme_settings["main_color"],
        )

        # Add each augmentation to the grid
        for i, (name, aug_img) in enumerate(augmentations):
            if i < rows * cols:
                row = i // cols
                col = i % cols

                # Create subplot with consistent aspect ratio
                ax = axes[row, col]
                ax.set_facecolor(theme_settings["background_color"])
                ax.imshow(aug_img)
                ax.set_title(
                    name,
                    fontsize=14,
                    pad=10,
                    fontweight="medium",
                    color=theme_settings["text_color"],
                )
                ax.axis("off")

                # Add a subtle border around each image
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color(theme_settings["grid_color"])
                    spine.set_linewidth(1)

        # Ensure layout is tight but with consistent spacing
        plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=3.0)
        plt.subplots_adjust(top=0.88)  # Adjust to make room for title

        # Save figure with high resolution
        plt.savefig(
            output_path,
            dpi=400,
            bbox_inches="tight",
            facecolor=theme_settings["background_color"],
        )
        plt.close()

        logger.info(f"Created augmentation grid: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error creating augmentation grid: {e}")
        return False


def generate_augmentation_examples(
    experiment_dir: Path,
    num_images: int = 5,
    theme: str = "dark",
):
    """
    Generate examples of data augmentation techniques for a specified experiment.

    Args:
        experiment_dir: Path to experiment directory
        num_images: Number of images to generate examples for
        theme: Theme to use for visualization ('dark' or 'light')
    """
    try:
        # Load config
        config_path = experiment_dir / "config.yaml"
        if not config_path.exists():
            logger.error(f"No config found at {config_path}")
            return

        import yaml

        with open(config_path) as f:
            # Load experiment config
            try:
                config = yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                return

        # Get data directory
        data_dir = config.get("data", {}).get("data_dir", "data/raw")
        if not os.path.exists(data_dir):
            logger.error(f"Data directory {data_dir} not found")
            return

        # Create output directory
        output_dir = experiment_dir / "reports" / "plots" / "augmentation"
        os.makedirs(output_dir, exist_ok=True)

        # Sample one image from each class to show augmentations
        import random

        # Get class folders
        class_folders = [
            f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))
        ]

        # Filter valid image files
        valid_extensions = (".jpg", ".jpeg", ".png")
        image_files = []

        # Randomly sample images
        for class_name in class_folders:
            class_path = os.path.join(data_dir, class_name)
            files = [
                os.path.join(class_path, f)
                for f in os.listdir(class_path)
                if f.lower().endswith(valid_extensions)
            ]
            if files:
                image_files.append(random.choice(files))
                # Only get up to num_images
                if len(image_files) >= num_images:
                    break

        if not image_files:
            logger.error("No image files found in data directory")
            return

        # Create augmentation grid for each image
        success_count = 0
        for i, img_path in enumerate(image_files):
            output_path = output_dir / f"augmentation_example_{i + 1}.png"
            if create_augmentation_grid(img_path, output_path, theme=theme):
                success_count += 1

        logger.info(f"Generated {success_count} augmentation examples")

    except Exception as e:
        logger.error(f"Error generating augmentation examples: {e}")
        import traceback

        logger.error(traceback.format_exc())


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
