# datamodule.py stub
# Path: core/data/datamodule.py
# Description: Data module for plant disease classification using PyTorch Lightning

import os
from typing import List, Optional

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

# Import from the new project structure
from core.data.datasets import (
    PlantDiseaseDataset,
)
from core.data.transforms import (  # Assuming transforms.py exists based on the second example
    AlbumentationsWrapper,
    get_transforms,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class PlantDiseaseDataModule:
    """
    PyTorch Lightning DataModule for plant disease classification.

    Handles dataset splitting, transformations, and data loading.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize the data module.

        Args:
            cfg: Configuration object (Hydra DictConfig). Expected keys:
                data:
                    data_dir: Root directory of the raw dataset.
                    train_val_test_split: List/tuple of ratios for train, val, test splits.
                    random_seed: Seed for reproducibility of splits.
                loader:
                    batch_size: Batch size for data loaders.
                    num_workers: Number of workers for data loading.
                    pin_memory: Whether to use pin_memory.
                    drop_last: Whether to drop the last batch if smaller.
                    prefetch_factor: Prefetch factor for DataLoader.
                preprocessing: Configuration for image transforms (used by get_transforms).
                augmentation: Configuration for image augmentations (used by get_transforms).
        """
        super().__init__()
        self.cfg = cfg
        # Use OmegaConf interpolation or direct access for paths
        self.data_dir = cfg.paths.raw_dir  # Use resolved path from config
        self.batch_size = cfg.loader.batch_size
        self.num_workers = cfg.loader.num_workers
        self.pin_memory = cfg.loader.pin_memory
        self.drop_last = cfg.loader.drop_last
        self.prefetch_factor = cfg.loader.prefetch_factor
        self.train_val_test_split = cfg.data.train_val_test_split
        self.random_seed = cfg.data.random_seed

        # Placeholders for datasets and class info
        self.train_dataset: Optional[PlantDiseaseDataset] = None
        self.val_dataset: Optional[PlantDiseaseDataset] = None
        self.test_dataset: Optional[PlantDiseaseDataset] = None
        self.class_names: Optional[List[str]] = None
        self.num_classes: Optional[int] = None

        logger.info("PlantDiseaseDataModule initialized.")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Batch size: {self.batch_size}, Num workers: {self.num_workers}")

    def prepare_data(self):
        """
        Download or prepare data. Only runs on the main process.
        Placeholder for potential download logic.
        """
        # Example: Check if data exists, if not, download/extract
        if not os.path.exists(self.data_dir):
            logger.error(f"Data directory not found: {self.data_dir}")
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        logger.info("Data preparation step completed (checking data existence).")
        # If using HDF5 caching (from plantdl), this is where you might trigger cache creation.

    def setup(self, stage: Optional[str] = None):
        """
        Set up the datasets and splits. Runs on all processes.

        Args:
            stage: 'fit', 'validate', 'test', 'predict', or None.
                   Determines which datasets to set up.
        """
        logger.info(f"Setting up DataModule for stage: {stage}")

        # Get transforms for each split
        # Assuming get_transforms uses cfg.preprocessing and cfg.augmentation
        train_transform = AlbumentationsWrapper(get_transforms(self.cfg, split="train"))
        val_transform = AlbumentationsWrapper(get_transforms(self.cfg, split="val"))
        test_transform = AlbumentationsWrapper(get_transforms(self.cfg, split="test"))

        # --- Determine Class Names ---
        # Create a temporary dataset just to get class names consistently
        try:
            temp_dataset = PlantDiseaseDataset(
                data_dir=self.data_dir,  # Assumes classes are subdirs of data_dir
                transform=None,
                split="train",  # Split doesn't matter here
            )
            self.class_names = temp_dataset.classes
            self.num_classes = len(self.class_names)
            logger.info(f"Found {self.num_classes} classes: {self.class_names}")
        except FileNotFoundError:
            logger.error(
                f"Data directory or class subdirectories not found in {self.data_dir}"
            )
            raise
        except Exception as e:
            logger.error(f"Error initializing temporary dataset to find classes: {e}")
            raise

        # --- Create Datasets (Handle pre-split or split on the fly) ---
        train_data_path = os.path.join(self.data_dir, "train")
        val_data_path = os.path.join(self.data_dir, "val")
        test_data_path = os.path.join(self.data_dir, "test")

        if (
            os.path.exists(train_data_path)
            and os.path.exists(val_data_path)
            and os.path.exists(test_data_path)
        ):
            logger.info("Found pre-split train/val/test directories.")
            if stage == "fit" or stage is None:
                self.train_dataset = PlantDiseaseDataset(
                    data_dir=train_data_path,
                    transform=train_transform,
                    split="train",
                    classes=self.class_names,
                )
                self.val_dataset = PlantDiseaseDataset(
                    data_dir=val_data_path,
                    transform=val_transform,
                    split="val",
                    classes=self.class_names,
                )
            if stage == "test" or stage is None:
                self.test_dataset = PlantDiseaseDataset(
                    data_dir=test_data_path,
                    transform=test_transform,
                    split="test",
                    classes=self.class_names,
                )
            # Handle validate stage explicitly if needed (usually covered by 'fit')
            if stage == "validate" and self.val_dataset is None:
                self.val_dataset = PlantDiseaseDataset(
                    data_dir=val_data_path,
                    transform=val_transform,
                    split="val",
                    classes=self.class_names,
                )

        else:
            logger.info(
                "Pre-split directories not found. Splitting dataset on the fly."
            )
            logger.info(
                f"Using split ratio: {self.train_val_test_split} and seed: {self.random_seed}"
            )

            full_dataset = PlantDiseaseDataset(
                data_dir=self.data_dir,
                transform=None,  # Apply transforms after splitting
                split="all",  # Indicate it's the full dataset
                classes=self.class_names,
            )

            total_len = len(full_dataset)
            train_len = int(self.train_val_test_split[0] * total_len)
            val_len = int(self.train_val_test_split[1] * total_len)
            test_len = total_len - train_len - val_len

            if train_len + val_len + test_len != total_len:
                logger.warning(
                    f"Split lengths ({train_len}, {val_len}, {test_len}) do not sum to total ({total_len}). Adjusting test_len."
                )
                test_len = total_len - train_len - val_len  # Recalculate to be safe

            logger.info(
                f"Splitting into Train: {train_len}, Val: {val_len}, Test: {test_len}"
            )

            # Split the dataset using torch.utils.data.random_split
            generator = torch.Generator().manual_seed(self.random_seed)
            train_subset, val_subset, test_subset = random_split(
                full_dataset, [train_len, val_len, test_len], generator=generator
            )

            # Create new datasets from subsets with the correct transforms
            if stage == "fit" or stage is None:
                self.train_dataset = self._create_dataset_from_subset(
                    train_subset, train_transform, "train"
                )
                self.val_dataset = self._create_dataset_from_subset(
                    val_subset, val_transform, "val"
                )
            if stage == "test" or stage is None:
                self.test_dataset = self._create_dataset_from_subset(
                    test_subset, test_transform, "test"
                )
            # Handle validate stage explicitly if needed
            if stage == "validate" and self.val_dataset is None:
                self.val_dataset = self._create_dataset_from_subset(
                    val_subset, val_transform, "val"
                )

        logger.info("DataModule setup complete.")
        if self.train_dataset:
            logger.info(f"Train dataset size: {len(self.train_dataset)}")
        if self.val_dataset:
            logger.info(f"Validation dataset size: {len(self.val_dataset)}")
        if self.test_dataset:
            logger.info(f"Test dataset size: {len(self.test_dataset)}")

    def _create_dataset_from_subset(
        self, subset, transform, split_name: str
    ) -> PlantDiseaseDataset:
        """
        Helper to create a PlantDiseaseDataset instance from a Subset,
        applying the correct transform.

        Args:
            subset: A torch.utils.data.Subset instance.
            transform: The transformations to apply.
            split_name: Name of the split ('train', 'val', 'test').

        Returns:
            A new PlantDiseaseDataset instance containing only the subset samples.
        """
        # Create a new dataset instance but populate it with subset samples
        dataset = PlantDiseaseDataset(
            data_dir=self.data_dir,  # Keep original data_dir reference if needed
            transform=transform,
            split=split_name,
            classes=self.class_names,
        )
        # Overwrite samples with only those from the subset
        dataset.samples = [subset.dataset.samples[i] for i in subset.indices]
        return dataset

    def train_dataloader(self) -> DataLoader:
        """Get the training data loader with weighted sampling for class balance."""
        if not self.train_dataset:
            self.setup(stage="fit")  # Ensure setup has run

        if not self.train_dataset:
            raise RuntimeError("Train dataset not initialized. Call setup() first.")

        # --- Weighted Random Sampling ---
        logger.info("Setting up weighted random sampler for training.")
        targets = [label for _, label in self.train_dataset.samples]

        class_counts = np.bincount(targets, minlength=self.num_classes)
        logger.debug(f"Class counts for sampling: {class_counts}")

        # Handle zero counts to avoid division by zero
        weights = np.where(class_counts > 0, 1.0 / np.sqrt(class_counts), 0)
        weights[class_counts == 0] = (
            0  # Explicitly set weight to 0 for classes with 0 samples
        )

        # Assign weight to each sample based on its class
        sample_weights = torch.tensor([weights[t] for t in targets], dtype=torch.float)

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,  # Sample with replacement
        )
        # --- End Sampling ---

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,  # Use sampler, shuffle=False is implicit
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        """Get the validation data loader."""
        if not self.val_dataset:
            self.setup(stage="fit")  # Ensure setup has run (fit usually includes val)

        if not self.val_dataset:
            raise RuntimeError(
                "Validation dataset not initialized. Call setup() first."
            )

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # No shuffling for validation
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,  # Keep all validation samples
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader:
        """Get the test data loader."""
        if not self.test_dataset:
            self.setup(stage="test")  # Ensure setup has run

        if not self.test_dataset:
            raise RuntimeError("Test dataset not initialized. Call setup() first.")

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # No shuffling for testing
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,  # Keep all test samples
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def get_class_names(self) -> List[str]:
        """Get the list of class names."""
        if not self.class_names:
            self.setup()  # Ensure setup has run
        if not self.class_names:
            raise RuntimeError("Class names not available.")
        return self.class_names

    def get_num_classes(self) -> int:
        """Get the number of classes."""
        if not self.num_classes:
            self.setup()  # Ensure setup has run
        if not self.num_classes:
            raise RuntimeError("Number of classes not available.")
        return self.num_classes

    def get_class_weights(self, mode="sqrt_inv") -> Optional[torch.Tensor]:
        """
        Calculate and return class weights, useful for loss functions.

        Args:
            mode: Method for calculating weights ('inv', 'sqrt_inv', 'effective').

        Returns:
            Tensor of class weights or None if train_dataset is not set.
        """
        if not self.train_dataset or not self.num_classes:
            logger.warning(
                "Cannot calculate class weights: train_dataset or num_classes not set."
            )
            return None

        targets = [label for _, label in self.train_dataset.samples]
        class_counts = np.bincount(targets, minlength=self.num_classes)

        if mode == "inv":
            # Inverse frequency
            weights = np.where(class_counts > 0, 1.0 / class_counts, 0)
        elif mode == "sqrt_inv":
            # Inverse square root frequency (often balances better)
            weights = np.where(class_counts > 0, 1.0 / np.sqrt(class_counts), 0)
        elif mode == "effective":
            # Effective Number of Samples weighting: (1 - beta^N) / (1 - beta)
            # Commonly used beta values are 0.9, 0.99, 0.999, 0.9999
            beta = 0.999
            effective_num = 1.0 - np.power(beta, class_counts)
            weights = np.where(effective_num > 0, (1.0 - beta) / effective_num, 0)
        else:
            logger.error(f"Unknown class weight mode: {mode}")
            return None

        # Normalize weights to sum to num_classes (optional, but common)
        total_weight = np.sum(weights)
        if total_weight > 0:
            weights = weights / total_weight * self.num_classes
        else:
            logger.warning("Total weight is zero, cannot normalize.")
            weights = np.ones_like(weights)  # Fallback to equal weights

        logger.info(f"Calculated class weights (mode={mode}): {weights}")
        return torch.tensor(weights, dtype=torch.float32)
