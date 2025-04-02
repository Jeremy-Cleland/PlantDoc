"""
Batch-level data augmentation for plant disease classification.

This module provides batch-level augmentations that operate on
mini-batches of data rather than individual images.
"""

import random
from typing import Dict, Tuple

import numpy as np
import torch
from omegaconf import DictConfig

from utils.logger import get_logger

logger = get_logger(__name__)


class BatchAugmentation:
    """
    Apply batch-level augmentations like MixUp and CutMix.

    Args:
        cfg: Augmentation configuration
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        # MixUp settings
        self.use_mixup = False
        self.mixup_alpha = 1.0
        self.mixup_p = 0.0  # Probability of applying MixUp

        # CutMix settings
        self.use_cutmix = False
        self.cutmix_alpha = 1.0
        self.cutmix_p = 0.0  # Probability of applying CutMix

        self._parse_config()

        logger.info(
            f"BatchAugmentation initialized: "
            f"MixUp={self.use_mixup}(α={self.mixup_alpha},p={self.mixup_p}), "
            f"CutMix={self.use_cutmix}(α={self.cutmix_alpha},p={self.cutmix_p})"
        )

    def _parse_config(self):
        """Parse configuration to set up augmentation parameters."""
        if not hasattr(self.cfg, "augmentation") or not hasattr(
            self.cfg.augmentation, "train"
        ):
            return

        aug_cfg = self.cfg.augmentation.train

        # Setup MixUp
        if hasattr(aug_cfg, "mixup") and aug_cfg.mixup.get("enabled", False):
            self.use_mixup = True
            self.mixup_alpha = aug_cfg.mixup.get("alpha", 1.0)
            self.mixup_p = aug_cfg.mixup.get("p", 0.5)

        # Setup CutMix
        if hasattr(aug_cfg, "cutmix") and aug_cfg.cutmix.get("enabled", False):
            self.use_cutmix = True
            self.cutmix_alpha = aug_cfg.cutmix.get("alpha", 1.0)
            self.cutmix_p = aug_cfg.cutmix.get("p", 0.5)

    def __call__(
        self, images: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Apply batch augmentations to a batch of images and targets.

        Args:
            images: Batch of images with shape [batch_size, channels, height, width]
            targets: Batch of targets with shape [batch_size] or [batch_size, num_classes]

        Returns:
            Tuple of (augmented images, augmented targets, metadata)
        """
        metadata = {
            "mixup_applied": False,
            "cutmix_applied": False,
            "original_targets": None,
            "lam": None,
        }

        # Skip augmentation with probability based on configured values
        if not self.use_mixup and not self.use_cutmix:
            return images, targets, metadata

        # Decide which augmentation to apply
        should_apply_mixup = self.use_mixup and random.random() < self.mixup_p
        should_apply_cutmix = self.use_cutmix and random.random() < self.cutmix_p

        # If both are selected, randomly pick one
        # This avoids always prioritizing one over the other
        if should_apply_mixup and should_apply_cutmix:
            if random.random() < 0.5:
                should_apply_cutmix = False
            else:
                should_apply_mixup = False

        batch_size = images.size(0)
        device = images.device

        # Apply MixUp
        if should_apply_mixup:
            logger.debug("Applying MixUp to batch")
            metadata["mixup_applied"] = True
            metadata["original_targets"] = targets.clone()

            # Generate mixing parameter from beta distribution
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            metadata["lam"] = lam

            # Get random permutation for mixing
            indices = torch.randperm(batch_size, device=device)

            # Mix images
            mixed_images = lam * images + (1 - lam) * images[indices]

            # Mix targets (handle both one-hot and class indices)
            if len(targets.shape) > 1:  # One-hot encoded
                mixed_targets = lam * targets + (1 - lam) * targets[indices]
            else:  # Class indices
                # For cross-entropy loss, we need to return both targets and mixing info
                mixed_targets = targets.clone()  # Will be handled by loss function
                metadata["mixed_indices"] = indices

            return mixed_images, mixed_targets, metadata

        # Apply CutMix
        elif should_apply_cutmix:
            logger.debug("Applying CutMix to batch")
            metadata["cutmix_applied"] = True
            metadata["original_targets"] = targets.clone()

            # Generate mixing parameter from beta distribution
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            metadata["lam"] = lam

            # Get random permutation for mixing
            indices = torch.randperm(batch_size, device=device)

            # Get image dimensions
            _, _, h, w = images.shape

            # Calculate box dimensions
            cut_ratio = np.sqrt(1.0 - lam)
            cut_w = int(w * cut_ratio)
            cut_h = int(h * cut_ratio)

            # Generate random box coordinates
            cx = np.random.randint(w)
            cy = np.random.randint(h)

            # Get box corners (ensure they're within image bounds)
            bbx1 = np.clip(cx - cut_w // 2, 0, w)
            bby1 = np.clip(cy - cut_h // 2, 0, h)
            bbx2 = np.clip(cx + cut_w // 2, 0, w)
            bby2 = np.clip(cy + cut_h // 2, 0, h)

            # Create mixed images (replace part of images with corresponding part from shuffled images)
            mixed_images = images.clone()
            mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[
                indices, :, bby1:bby2, bbx1:bbx2
            ]

            # Adjust lambda based on actual box size (may differ from intended due to clipping)
            actual_box_area = (bbx2 - bbx1) * (bby2 - bby1)
            image_area = w * h
            lam = 1 - (actual_box_area / image_area)
            metadata["lam"] = lam

            # Mix targets (handle both one-hot and class indices)
            if len(targets.shape) > 1:  # One-hot encoded
                mixed_targets = lam * targets + (1 - lam) * targets[indices]
            else:  # Class indices
                # For cross-entropy loss, we need to return both targets and mixing info
                mixed_targets = targets.clone()  # Will be handled by loss function
                metadata["mixed_indices"] = indices

            return mixed_images, mixed_targets, metadata

        # No augmentation applied
        return images, targets, metadata


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Apply MixUp/CutMix criterion.

    Args:
        criterion: Loss function
        pred: Model predictions
        y_a: First set of targets
        y_b: Second set of targets
        lam: Mixing parameter

    Returns:
        Mixed loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
