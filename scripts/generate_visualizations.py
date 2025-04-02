#!/usr/bin/env python
"""
Generate all visualizations for a trained model.

This script generates:
1. Augmentation examples
2. Attention visualizations (for models with CBAM)
3. SHAP explanations

Usage:
    python scripts/generate_visualizations.py --experiment outputs/experiment_name
"""

import argparse
import os
import sys
import traceback
from pathlib import Path

import torch

from core.models import get_model
from utils.logger import get_logger
from utils.paths import ensure_dir

logger = get_logger(__name__)


def generate_all_visualizations(
    experiment_dir: Path,
    model_name: str = None,
    device: str = "cpu",
    num_images: int = 5,
    skip_augmentation: bool = False,
    skip_attention: bool = False,
    skip_shap: bool = False,
    force: bool = False,
    theme: str = "plantdoc",  # Use a plantdoc theme for all visualizations
):
    """
    Generate all visualizations for a trained model.

    Args:
        experiment_dir: Path to experiment directory
        model_name: Name of the model architecture
        device: Device to run model on ('cpu', 'cuda', 'mps')
        num_images: Number of images to process
        skip_augmentation: Whether to skip augmentation examples
        skip_attention: Whether to skip attention visualizations
        skip_shap: Whether to skip SHAP explanations
        force: Whether to regenerate existing visualizations
    """
    logger.info(f"Generating visualizations for {experiment_dir}")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Device: {device}")

    # Directories for visualizations
    aug_dir = experiment_dir / "augmentation_examples"
    attention_dir = experiment_dir / "attention_visualizations"
    shap_dir = experiment_dir / "shap_visualizations"

    # Check for existing visualizations
    has_aug = aug_dir.exists() and any(aug_dir.glob("*.png"))
    has_attention = attention_dir.exists() and any(attention_dir.glob("*.png"))
    has_shap = shap_dir.exists() and any(shap_dir.glob("*.png"))

    if not force:
        if has_aug:
            logger.info(
                f"Augmentation examples already exist in {aug_dir}. Use --force to regenerate."
            )
            skip_augmentation = True

        if has_attention:
            logger.info(
                f"Attention visualizations already exist in {attention_dir}. Use --force to regenerate."
            )
            skip_attention = True

        if has_shap:
            logger.info(
                f"SHAP visualizations already exist in {shap_dir}. Use --force to regenerate."
            )
            skip_shap = True

    # 1. Generate augmentation examples
    if not skip_augmentation:
        logger.info("Generating augmentation examples...")
        try:
            from scripts.generate_augmentations import generate_augmentation_examples

            generate_augmentation_examples(
                experiment_dir=experiment_dir,
                num_images=num_images,
            )
            logger.info("✅ Augmentation examples generated successfully")
        except Exception as e:
            logger.error(f"❌ Error generating augmentation examples: {e}")
            logger.error(traceback.format_exc())

    # 2. Generate attention visualizations
    if not skip_attention:
        if not model_name:
            model_name = detect_model_name(experiment_dir)

        # Check if model likely has attention modules
        has_attention_mechanisms = (
            "cbam" in model_name.lower()
            or "attention" in model_name.lower()
            or "se" in model_name.lower()  # Squeeze and Excitation networks
        )

        if has_attention_mechanisms:
            logger.info(f"Generating attention visualizations for {model_name}...")
            try:
                model_path = find_model_checkpoint(experiment_dir)
                if not model_path:
                    logger.error("❌ No model checkpoint found")
                else:
                    # Load model
                    model = load_model(model_path, model_name, device)

                    # Generate attention visualizations
                    from scripts.generate_attention import (
                        generate_attention_visualizations,
                    )

                    # Pass the model directly to avoid loading it again
                    generate_attention_visualizations(
                        model=model,
                        experiment_dir=experiment_dir,
                        num_images=num_images,
                        device=device,
                    )
                    logger.info("✅ Attention visualizations generated successfully")
            except Exception as e:
                logger.error(f"❌ Error generating attention visualizations: {e}")
                logger.error(traceback.format_exc())
        else:
            logger.info(
                f"Skipping attention visualizations: {model_name} likely doesn't have attention mechanisms"
            )

    # 3. Generate SHAP explanations
    if not skip_shap:
        logger.info("Generating SHAP explanations...")
        try:
            if not model_name:
                model_name = detect_model_name(experiment_dir)

            model_path = find_model_checkpoint(experiment_dir)
            if not model_path:
                logger.error("❌ No model checkpoint found")
            else:
                # Get class names
                class_names = load_class_names(experiment_dir)
                if not class_names:
                    logger.warning("⚠️ No class names found. Using placeholder names.")
                    class_names = [
                        f"Class {i}" for i in range(10)
                    ]  # Default to 10 classes

                # Create dataset
                dataset = create_test_dataset(experiment_dir)
                if dataset is None:
                    logger.error("❌ Could not create dataset for SHAP analysis")
                    return

                # Load model
                model = load_model(model_path, model_name, device)

                # Generate SHAP visualizations
                from core.evaluation.shap_interpreter import (
                    generate_shap_visualizations,
                )

                generate_shap_visualizations(
                    model=model,
                    dataset=dataset,
                    class_names=class_names,
                    experiment_dir=experiment_dir,
                    device=device,
                    num_samples=min(100, len(dataset)),
                    batch_size=32,
                )
                logger.info("✅ SHAP explanations generated successfully")
        except Exception as e:
            logger.error(f"❌ Error generating SHAP explanations: {e}")
            logger.error(traceback.format_exc())

    logger.info("Visualization generation complete!")


def detect_model_name(experiment_dir: Path) -> str:
    """Detect model name from experiment directory or config file."""
    # Try to detect from config
    model_name = None
    config_path = experiment_dir / "config.yaml"

    if config_path.exists():
        try:
            import yaml

            with open(config_path) as f:
                config = yaml.safe_load(f)

            if "model" in config and "name" in config["model"]:
                model_name = config["model"]["name"]
                logger.info(f"Detected model name from config: {model_name}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")

    # If not found, try to infer from experiment name
    if model_name is None:
        exp_name = experiment_dir.name.lower()

        # Check for common model architectures in the name
        if "resnet" in exp_name:
            model_type = "resnet18"
            if "cbam" in exp_name:
                model_type = "cbam_resnet18"
            if "50" in exp_name:
                model_type = "resnet50" if "cbam" not in exp_name else "cbam_resnet50"
            model_name = model_type
        elif "densenet" in exp_name:
            model_name = "densenet121"
        elif "efficientnet" in exp_name:
            model_name = "efficientnet_b0"
        elif "mobilenet" in exp_name:
            model_name = "mobilenet_v2"
        else:
            model_name = "resnet18"  # Default fallback

        logger.info(f"Inferred model name from experiment name: {model_name}")

    return model_name


def find_model_checkpoint(experiment_dir: Path) -> Path:
    """Find the best model checkpoint in the experiment directory."""
    checkpoint_dir = experiment_dir / "checkpoints"

    if not checkpoint_dir.exists():
        logger.error(f"Checkpoint directory {checkpoint_dir} does not exist")
        return None

    # Try to find best model first
    best_model_path = checkpoint_dir / "best_model.pth"
    if best_model_path.exists():
        logger.info(f"Found best model checkpoint: {best_model_path}")
        return best_model_path

    # If best model not found, look for any checkpoint
    checkpoints = list(checkpoint_dir.glob("*.pth"))
    if checkpoints:
        logger.info(f"Found checkpoint: {checkpoints[0]}")
        return checkpoints[0]

    logger.error("No model checkpoint found")
    return None


def load_model(
    model_path: Path, model_name: str, device: str = "cpu"
) -> torch.nn.Module:
    """Load model from checkpoint."""
    logger.info(f"Loading model {model_name} from {model_path}")

    try:
        # Use the new get_model function
        model = get_model(
            model_name=model_name, checkpoint_path=model_path, device=device
        )

        # Set model to evaluation mode
        model.eval()

        logger.info(
            f"Successfully loaded model with {sum(p.numel() for p in model.parameters())} parameters"
        )
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error(traceback.format_exc())
        raise


def load_class_names(experiment_dir: Path) -> list:
    """Load class names from experiment directory."""
    class_names_path = experiment_dir / "class_names.txt"

    if class_names_path.exists():
        try:
            with open(class_names_path, "r") as f:
                class_names = [line.strip() for line in f]
            logger.info(f"Loaded {len(class_names)} class names")
            return class_names
        except Exception as e:
            logger.error(f"Error loading class names: {e}")

    return None


def create_test_dataset(experiment_dir: Path):
    """Create a test dataset for SHAP analysis."""
    # Try to find dataset path
    dataset_path = None
    config_path = experiment_dir / "config.yaml"

    if config_path.exists():
        try:
            import yaml

            with open(config_path) as f:
                config = yaml.safe_load(f)

            if "data" in config and "dataset_path" in config["data"]:
                dataset_path = Path(config["data"]["dataset_path"])
                logger.info(f"Found dataset path in config: {dataset_path}")
        except Exception as e:
            logger.error(f"Error loading dataset path from config: {e}")

    # If not found in config, check common locations
    if dataset_path is None or not dataset_path.exists():
        candidate_dirs = [
            Path("data/raw"),
            Path("data/processed"),
            Path("data"),
        ]

        for candidate in candidate_dirs:
            if candidate.exists() and any(candidate.glob("**/*.jpg")):
                dataset_path = candidate
                logger.info(f"Using dataset from {dataset_path}")
                break

    if dataset_path is None or not dataset_path.exists():
        logger.error("Could not find dataset")
        return None

    # Load dataset with appropriate transform
    try:
        from torchvision import transforms

        # Use standard torchvision transforms instead of Albumentations to avoid errors
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        logger.info("Using torchvision transforms for SHAP analysis")

        # Try different approaches to create the dataset
        dataset = None

        try:
            # Avoid using PlantDiseaseDataset directly due to transform errors
            # Use ImageFolder instead which works with the torchvision transforms
            from torchvision.datasets import ImageFolder

            dataset = ImageFolder(str(dataset_path), transform=transform)
            logger.info(
                f"Created dataset with {len(dataset)} samples using ImageFolder"
            )

            # Optionally create a smaller subset for visualization
            if len(dataset) > 100:
                import torch
                from torch.utils.data import Subset

                indices = torch.randperm(len(dataset))[:100].tolist()
                dataset = Subset(dataset, indices)
                logger.info(
                    f"Created subset with {len(dataset)} samples for visualization"
                )

        except Exception as e:
            logger.error(f"Error creating ImageFolder dataset: {e}")
            # If ImageFolder fails, try one more approach with a specific subfolder
            try:
                # Try with a specific subfolder that might contain proper class structure
                class_dirs = [
                    d
                    for d in dataset_path.iterdir()
                    if d.is_dir() and not d.name.startswith(".")
                ]
                if class_dirs:
                    logger.info(
                        f"Trying with first few class directories out of {len(class_dirs)} total"
                    )

                    # Create a custom dataset to handle a subset of the data
                    from torch.utils.data import Dataset

                    class SimpleImageDataset(Dataset):
                        def __init__(self, samples, transform=None):
                            self.samples = samples  # List of (path, class_idx) tuples
                            self.transform = transform
                            self.classes = sorted(
                                list(set(class_idx for _, class_idx in samples))
                            )

                        def __len__(self):
                            return len(self.samples)

                        def __getitem__(self, idx):
                            path, class_idx = self.samples[idx]
                            from PIL import Image

                            img = Image.open(path).convert("RGB")
                            if self.transform:
                                img = self.transform(img)
                            return img, class_idx

                    # Gather samples from first few classes
                    import random

                    samples = []
                    max_classes = min(10, len(class_dirs))  # Use at most 10 classes
                    max_samples_per_class = 10  # Use at most 10 samples per class

                    for class_idx, class_dir in enumerate(class_dirs[:max_classes]):
                        # Get all image files in this class
                        image_files = (
                            list(class_dir.glob("*.jpg"))
                            + list(class_dir.glob("*.jpeg"))
                            + list(class_dir.glob("*.png"))
                        )
                        if image_files:
                            # Take a subset of images
                            selected_files = random.sample(
                                image_files,
                                min(max_samples_per_class, len(image_files)),
                            )
                            for img_path in selected_files:
                                samples.append((str(img_path), class_idx))

                    if samples:
                        dataset = SimpleImageDataset(samples, transform)
                        logger.info(
                            f"Created custom dataset with {len(dataset)} samples"
                        )
            except Exception as nested_e:
                logger.error(f"Error creating custom dataset: {nested_e}")
                return None

        return dataset

    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        logger.error(traceback.format_exc())
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate model visualizations")
    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        required=True,
        help="Path to experiment directory",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run models on",
    )
    parser.add_argument(
        "--num-images",
        "-n",
        type=int,
        default=5,
        help="Number of images to process for each visualization type",
    )
    parser.add_argument(
        "--model-name",
        "-m",
        type=str,
        default=None,
        help="Model architecture name (will try to detect if not provided)",
    )
    parser.add_argument(
        "--skip-augmentation",
        action="store_true",
        help="Skip generating augmentation examples",
    )
    parser.add_argument(
        "--skip-attention",
        action="store_true",
        help="Skip generating attention visualizations",
    )
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip generating SHAP explanations",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force regeneration of visualizations even if they already exist",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        import logging

        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # Resolve experiment directory
    experiment_dir = Path(args.experiment)
    if not experiment_dir.is_absolute():
        experiment_dir = Path.cwd() / experiment_dir

    if not experiment_dir.exists():
        logger.error(f"Experiment directory {experiment_dir} does not exist")
        return

    # Generate visualizations
    generate_all_visualizations(
        experiment_dir=experiment_dir,
        model_name=args.model_name,
        device=args.device,
        num_images=args.num_images,
        skip_augmentation=args.skip_augmentation,
        skip_attention=args.skip_attention,
        skip_shap=args.skip_shap,
        force=args.force,
    )


if __name__ == "__main__":
    main()
