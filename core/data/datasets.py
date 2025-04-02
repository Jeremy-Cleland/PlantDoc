# Path: core/data/datasets.py
# Description: Dataset class for plant disease classification

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from utils.logger import get_logger

logger = get_logger(__name__)


class PlantDiseaseDataset(Dataset):
    """Dataset class for plant disease classification from directory-based image data."""

    def __init__(
        self,
        data_dir: str,
        transform: Optional[callable] = None,
        split: str = "train",
        classes: Optional[List[str]] = None,
        image_extensions: Tuple[str, ...] = (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
        ),
    ):
        """
        Initialize the PlantDiseaseDataset.

        Args:
            data_dir: Path to the dataset directory containing class subfolders
                     or specific split folder (e.g., 'data/raw/train')
            transform: Transformations to apply to the images
            split: Dataset split identifier ('train', 'val', 'test', 'all')
            classes: Optional list of class names to ensure consistent ordering
            image_extensions: Valid image file extensions (case-insensitive)
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        self.image_extensions = image_extensions
        self.samples: List[Tuple[str, int]] = []
        self.class_to_idx: Dict[str, int] = {}
        self.classes: List[str] = []

        if not self.data_dir.is_dir():
            logger.error(f"Dataset directory not found: {self.data_dir}")
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")

        self._find_classes(classes)
        self._load_samples()

        if not self.samples:
            logger.warning(
                f"No valid image samples found in {self.data_dir} for split '{self.split}'."
            )
        else:
            logger.info(
                f"Initialized dataset for split '{self.split}' with {len(self.samples)} samples from {len(self.classes)} classes."
            )

    def _find_classes(self, provided_classes: Optional[List[str]]):
        """Find classes from subdirectories or use provided list."""
        if provided_classes is not None:
            logger.info("Using provided class list.")
            self.classes = sorted(provided_classes)
        else:
            logger.info(f"Inferring classes from subdirectories in {self.data_dir}.")
            self.classes = sorted(
                [
                    d.name
                    for d in self.data_dir.iterdir()
                    if d.is_dir() and not d.name.startswith(".")
                ]
            )

        if not self.classes:
            raise ValueError(
                f"Could not find any class subdirectories in {self.data_dir}."
            )

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        logger.debug(f"Class to index mapping: {self.class_to_idx}")

    def _load_samples(self):
        """Load image paths and their corresponding class labels."""
        logger.info(f"Loading samples for split '{self.split}' from {self.data_dir}...")
        num_skipped = 0

        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if not class_dir.is_dir():
                logger.warning(f"Class directory '{class_dir}' not found. Skipping.")
                continue

            class_idx = self.class_to_idx[class_name]
            for item in class_dir.iterdir():
                if item.is_file() and item.suffix.lower() in self.image_extensions:
                    self.samples.append((str(item), class_idx))
                elif item.is_file():
                    num_skipped += 1
                    logger.debug(f"Skipping file with unsupported extension: {item}")

        if num_skipped > 0:
            logger.warning(f"Skipped {num_skipped} files with unsupported extensions.")
        logger.info(f"Found {len(self.samples)} potential image samples.")

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int, str]]:
        """
        Get a sample from the dataset by index.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Dictionary with 'image' tensor, 'label' index, and file 'path'
        """
        img_path, label = self.samples[idx]

        try:
            # Load and process the image
            image = Image.open(img_path).convert("RGB")

            if self.transform:
                image_tensor = self.transform(image)
            else:
                # Default conversion to tensor if no transform provided
                image_tensor = (
                    torch.from_numpy(np.array(image).transpose(2, 0, 1)).float() / 255.0
                )

            return {"image": image_tensor, "label": label, "path": img_path}

        except UnidentifiedImageError:
            logger.error(
                f"Cannot identify image file: {img_path}. Skipping sample {idx}."
            )
            # Return dummy sample to allow training to continue
            dummy_image = torch.zeros((3, 224, 224), dtype=torch.float32)
            dummy_label = label if isinstance(label, int) else 0
            return {
                "image": dummy_image,
                "label": dummy_label,
                "path": f"INVALID:{img_path}",
            }

        except Exception as e:
            logger.error(f"Error loading image {img_path} (sample {idx}): {e}")
            dummy_image = torch.zeros((3, 224, 224), dtype=torch.float32)
            dummy_label = label if isinstance(label, int) else 0
            return {
                "image": dummy_image,
                "label": dummy_label,
                "path": f"ERROR:{img_path}",
            }

    def get_class_weights(self, mode="sqrt_inv") -> Optional[torch.Tensor]:
        """
        Calculate class weights for handling imbalanced datasets.

        Args:
            mode: Weighting method ('inv', 'sqrt_inv', 'effective')

        Returns:
            Tensor of class weights or None if samples aren't loaded
        """
        if not self.samples or not self.classes:
            logger.warning(
                "Cannot calculate class weights: samples or classes not available."
            )
            return None

        num_classes = len(self.classes)
        labels = [label for _, label in self.samples]
        class_counts = np.bincount(labels, minlength=num_classes)

        if np.any(class_counts == 0):
            logger.warning(
                f"Some classes have zero samples in '{self.split}' split. Weights for these classes will be 0."
            )

        if mode == "inv":
            weights = np.where(class_counts > 0, 1.0 / class_counts, 0)
        elif mode == "sqrt_inv":
            weights = np.where(class_counts > 0, 1.0 / np.sqrt(class_counts), 0)
        elif mode == "effective":
            beta = 0.999
            effective_num = 1.0 - np.power(beta, class_counts)
            weights = np.where(effective_num > 0, (1.0 - beta) / effective_num, 0)
        else:
            logger.error(f"Unknown class weight mode: {mode}. Returning None.")
            return None

        # Normalize weights
        total_weight = np.sum(weights)
        if total_weight > 0:
            weights = weights / total_weight * num_classes
        else:
            logger.warning("Total weight is zero. Returning equal weights.")
            weights = np.ones_like(weights)

        logger.debug(
            f"Calculated class weights for split '{self.split}' (mode={mode}): {weights}"
        )
        return torch.tensor(weights, dtype=torch.float32)
