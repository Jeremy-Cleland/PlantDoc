# Path: plantdoc/core/data/transforms.py
# Description: Image transformations using Albumentations library


import random

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig
from PIL import Image

from utils.logger import get_logger

logger = get_logger(__name__)


def get_transforms(cfg: DictConfig, split: str = "train") -> A.Compose:
    """
    Get image transformations based on configuration and dataset split.

    Args:
        cfg: Configuration with preprocessing and augmentation parameters
        split: Dataset split ('train', 'val', or 'test')

    Returns:
        An Albumentations composition of transformations
    """
    try:
        # Determine configuration format and extract parameters
        if (
            hasattr(cfg.preprocessing, "resize")
            and hasattr(cfg.preprocessing, "center_crop")
            and hasattr(cfg.preprocessing, "normalize")
        ):
            height, width = cfg.preprocessing.resize
            crop_height, crop_width = cfg.preprocessing.center_crop
            mean = cfg.preprocessing.normalize.mean
            std = cfg.preprocessing.normalize.std
            logger.debug("Using new configuration format for transforms.")
        else:
            logger.debug("Using legacy configuration format for transforms.")
            split_cfg = getattr(cfg.preprocessing, split)
            height = split_cfg.resize_height
            width = split_cfg.resize_width

            if split == "train" and hasattr(split_cfg, "random_crop_height"):
                crop_height = split_cfg.random_crop_height
                crop_width = split_cfg.random_crop_width
            else:
                crop_height = split_cfg.center_crop_height
                crop_width = split_cfg.center_crop_width

            mean = split_cfg.mean
            std = split_cfg.std

        logger.debug(
            f"Transform params - Resize: ({height}, {width}), Crop: ({crop_height}, {crop_width})"
        )
        logger.debug(f"Transform params - Mean: {mean}, Std: {std}")

        transforms_list = []

        if (
            split == "train"
            and hasattr(cfg, "augmentation")
            and hasattr(cfg.augmentation, "train")
        ):
            # Training transformations with augmentations
            aug_cfg = cfg.augmentation.train
            logger.info("Applying training augmentations.")

            # Check for advanced augmentations first
            use_rand_augment = aug_cfg.get("rand_augment", {}).get("enabled", False)
            use_augmix = aug_cfg.get("augmix", {}).get("enabled", False)

            # RandAugment and AugMix are alternative augmentation strategies
            # We'll use one or the other, prioritizing RandAugment if both are enabled
            if use_rand_augment:
                rand_aug_ops = aug_cfg.rand_augment.get("num_ops", 2)
                rand_aug_mag = aug_cfg.rand_augment.get("magnitude", 9)
                rand_aug_p = aug_cfg.rand_augment.get("p", 0.5)

                logger.info(
                    f"Using RandAugment with {rand_aug_ops} ops and magnitude {rand_aug_mag}"
                )

                # With RandAugment, we still need basic resize/crop before augmentation
                transforms_list.append(A.Resize(height=height, width=width))
                if hasattr(aug_cfg, "random_crop"):
                    transforms_list.append(
                        A.RandomCrop(height=crop_height, width=crop_width)
                    )
                else:
                    transforms_list.append(
                        A.CenterCrop(height=crop_height, width=crop_width)
                    )

                # Add RandAugment
                transforms_list.append(
                    RandAugment(
                        num_ops=rand_aug_ops, magnitude=rand_aug_mag, p=rand_aug_p
                    )
                )

            elif use_augmix:
                augmix_severity = aug_cfg.augmix.get("severity", 3)
                augmix_width = aug_cfg.augmix.get("width", 3)
                augmix_depth = aug_cfg.augmix.get("depth", -1)
                augmix_alpha = aug_cfg.augmix.get("alpha", 1.0)
                augmix_p = aug_cfg.augmix.get("p", 0.5)

                logger.info(f"Using AugMix with severity {augmix_severity}")

                # With AugMix, we still need basic resize/crop before augmentation
                transforms_list.append(A.Resize(height=height, width=width))
                if hasattr(aug_cfg, "random_crop"):
                    transforms_list.append(
                        A.RandomCrop(height=crop_height, width=crop_width)
                    )
                else:
                    transforms_list.append(
                        A.CenterCrop(height=crop_height, width=crop_width)
                    )

                # Add AugMix
                transforms_list.append(
                    AugMix(
                        severity=augmix_severity,
                        width=augmix_width,
                        depth=augmix_depth,
                        alpha=augmix_alpha,
                        p=augmix_p,
                    )
                )

            else:
                # If no advanced augmentation strategy is enabled, fall back to traditional augmentations
                if hasattr(aug_cfg, "random_resized_crop"):
                    transforms_list.append(
                        A.RandomResizedCrop(
                            size=(crop_height, crop_width),
                            scale=aug_cfg.random_resized_crop.scale,
                            ratio=aug_cfg.random_resized_crop.ratio,
                            p=1.0,
                        )
                    )
                else:
                    transforms_list.append(A.Resize(height=height, width=width))
                    if hasattr(aug_cfg, "random_crop"):
                        transforms_list.append(
                            A.RandomCrop(height=crop_height, width=crop_width)
                        )

                if hasattr(aug_cfg, "horizontal_flip") and aug_cfg.horizontal_flip:
                    transforms_list.append(A.HorizontalFlip(p=0.5))
                if hasattr(aug_cfg, "vertical_flip") and aug_cfg.vertical_flip:
                    transforms_list.append(A.VerticalFlip(p=0.5))
                if hasattr(aug_cfg, "random_rotate"):
                    transforms_list.append(A.Rotate(limit=aug_cfg.random_rotate, p=0.7))
                if hasattr(aug_cfg, "color_jitter"):
                    transforms_list.append(
                        A.ColorJitter(
                            brightness=aug_cfg.color_jitter.brightness,
                            contrast=aug_cfg.color_jitter.contrast,
                            saturation=aug_cfg.color_jitter.saturation,
                            hue=aug_cfg.color_jitter.hue,
                            p=0.7,
                        )
                    )
                if hasattr(aug_cfg, "random_brightness_contrast"):
                    transforms_list.append(
                        A.RandomBrightnessContrast(
                            brightness_limit=aug_cfg.random_brightness_contrast.brightness_limit,
                            contrast_limit=aug_cfg.random_brightness_contrast.contrast_limit,
                            p=aug_cfg.random_brightness_contrast.p,
                        )
                    )
                if hasattr(aug_cfg, "shift_scale_rotate"):
                    transforms_list.append(
                        A.Affine(
                            scale=(
                                1 - aug_cfg.shift_scale_rotate.scale_limit,
                                1 + aug_cfg.shift_scale_rotate.scale_limit,
                            ),
                            translate_percent=(
                                -aug_cfg.shift_scale_rotate.shift_limit,
                                aug_cfg.shift_scale_rotate.shift_limit,
                            ),
                            rotate=(
                                -aug_cfg.shift_scale_rotate.rotate_limit,
                                aug_cfg.shift_scale_rotate.rotate_limit,
                            ),
                            p=aug_cfg.shift_scale_rotate.p,
                            interpolation=cv2.INTER_LINEAR,
                        )
                    )

                # Add CutOut if configured
                if hasattr(aug_cfg, "cutout") and aug_cfg.cutout.get("enabled", False):
                    transforms_list.append(
                        A.Cutout(
                            num_holes=aug_cfg.cutout.get("num_holes", 4),
                            max_h_size=aug_cfg.cutout.get("max_h_size", 16),
                            max_w_size=aug_cfg.cutout.get("max_w_size", 16),
                            p=aug_cfg.cutout.get("p", 0.5),
                        )
                    )

            # Add CenterCrop if no crop operation has been added yet
            if not any(
                isinstance(t, (A.RandomResizedCrop, A.RandomCrop, A.CenterCrop))
                for t in transforms_list
            ):
                transforms_list.append(
                    A.CenterCrop(height=crop_height, width=crop_width)
                )

        else:
            # Validation and test transformations (deterministic)
            logger.info(f"Applying {split} transformations (Resize -> CenterCrop).")
            transforms_list.extend(
                [
                    A.Resize(height=height, width=width),
                    A.CenterCrop(height=crop_height, width=crop_width),
                ]
            )

        # Normalization and tensor conversion for all splits
        transforms_list.extend(
            [
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )

        return A.Compose(transforms_list)

    except KeyError as e:
        logger.error(f"Missing key in configuration for transforms: {e}")
        raise
    except Exception as e:
        logger.error(f"Error creating transforms for split '{split}': {e}")
        raise


class AlbumentationsWrapper:
    """Wrapper for Albumentations transforms to handle PIL images as input."""

    def __init__(self, transform: A.Compose):
        """
        Args:
            transform: An Albumentations Compose object
        """
        self.transform = transform
        logger.debug("AlbumentationsWrapper initialized.")

    def __call__(self, img: Image.Image) -> torch.Tensor:
        """
        Apply transformations to a PIL image.

        Args:
            img: Input PIL Image

        Returns:
            Transformed image as a PyTorch tensor
        """
        try:
            # Convert PIL image to numpy array
            img_np = np.array(img)
            if img_np.ndim == 2:
                logger.warning(
                    "Input image is grayscale, converting to RGB for transform."
                )
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            elif img_np.shape[2] == 4:
                logger.warning("Input image has 4 channels (RGBA), converting to RGB.")
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)

            # Apply transformations
            transformed = self.transform(image=img_np)
            return transformed["image"]
        except Exception as e:
            logger.error(f"Error applying transform in AlbumentationsWrapper: {e}")
            raise


class RandAugment(A.ImageOnlyTransform):
    """
    RandAugment implementation for albumentations.

    Paper: https://arxiv.org/abs/1909.13719

    Args:
        num_ops: Number of operations to apply
        magnitude: Magnitude of augmentation (0-10)
        p: Probability of applying augmentation
    """

    def __init__(
        self, num_ops: int = 2, magnitude: int = 9, p: float = 0.5, always_apply=False
    ):
        super().__init__(p=p)
        self.num_ops = num_ops
        self.magnitude = magnitude

        # Define operations with their respective ranges
        m = float(magnitude) / 10.0  # Normalize magnitude to [0, 1]
        self.operations = [
            A.Affine(
                scale=(1.0 - 0.2 * m, 1.0 + 0.2 * m),
                translate_percent=(-0.1 * m, 0.1 * m),
                rotate=(-30 * m, 30 * m),
                p=1.0,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2 * m,
                contrast_limit=0.2 * m,
                p=1.0,
            ),
            A.HueSaturationValue(
                hue_shift_limit=int(10 * m),
                sat_shift_limit=int(20 * m),
                val_shift_limit=int(10 * m),
                p=1.0,
            ),
            A.Blur(blur_limit=(3, max(3, int(3 + 2 * m))), p=1.0),
            A.CLAHE(clip_limit=max(1, int(2 * m)), p=1.0),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.3, 0.7), p=1.0),
            A.Equalize(p=1.0),
            A.GaussNoise(var_limit=(10.0, 50.0 * m), p=1.0),
            A.Posterize(num_bits=max(1, int(8 - 4 * m)), p=1.0),
            A.Solarize(p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.3 * m, p=1.0),
        ]
        logger.info(
            f"Initialized RandAugment with num_ops={num_ops}, magnitude={magnitude}"
        )

    def apply(self, img, **params):
        ops = random.sample(self.operations, min(self.num_ops, len(self.operations)))

        for op in ops:
            img = op(image=img)["image"]

        return img

    def get_transform_init_args_names(self):
        return ("num_ops", "magnitude", "p")


class AugMix(A.ImageOnlyTransform):
    """
    AugMix implementation for albumentations.

    Paper: https://arxiv.org/abs/1912.02781

    Args:
        severity: Severity of augmentation operations (1-10)
        width: Number of augmentation chains to mix
        depth: Depth of augmentation chains
        alpha: Parameter for beta distribution
        p: Probability of applying augmentation
    """

    def __init__(
        self,
        severity: int = 3,
        width: int = 3,
        depth: int = -1,
        alpha: float = 1.0,
        p: float = 0.5,
        always_apply=False,
    ):
        super().__init__(p=p)
        self.severity = severity
        self.width = width
        self.depth = depth  # -1 means random depth
        self.alpha = alpha

        # Define operations with proper parameter values
        self.operations = [
            A.HorizontalFlip(p=1.0),
            A.Affine(
                scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), rotate=(-30, 30), p=1.0
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.HueSaturationValue(
                hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0
            ),
            A.Blur(blur_limit=3, p=1.0),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
            A.GaussNoise(var_limit=(10, 50), mean=0, p=1.0),
            A.Equalize(p=1.0),
        ]
        logger.info(
            f"Initialized AugMix with severity={severity}, width={width}, depth={depth}, alpha={alpha}"
        )

    def apply(self, img, **params):
        mix = np.zeros_like(img, dtype=np.float32)
        weights = np.random.dirichlet([self.alpha] * self.width)

        for i in range(self.width):
            chain_depth = random.randint(1, 3) if self.depth == -1 else self.depth
            img_aug = img.copy()

            for _ in range(chain_depth):
                op = random.choice(self.operations)
                img_aug = op(image=img_aug)["image"]

            mix += weights[i] * img_aug.astype(np.float32)

        # Convert back to uint8
        return np.clip(mix, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("severity", "width", "depth", "alpha", "p")


class MixUpTransform:
    """
    MixUp augmentation: linearly mix two images.

    Paper: https://arxiv.org/abs/1710.09412

    Args:
        alpha: Parameter for beta distribution
        p: Probability of applying augmentation
    """

    def __init__(self, alpha: float = 1.0, p: float = 0.5):
        self.alpha = alpha
        self.p = p
        logger.info(f"Initialized MixUp with alpha={alpha}, p={p}")

    def __call__(self, batch):
        """Apply MixUp to a batch of images and targets.

        Args:
            batch: Tuple of (images, targets)

        Returns:
            Tuple of mixed images and targets
        """
        if random.random() > self.p:
            return batch

        images, targets = batch
        batch_size = len(images)

        # Generate lambda from beta distribution
        lam = np.random.beta(self.alpha, self.alpha)

        # Create shuffled indices
        indices = torch.randperm(batch_size)

        # Mix images
        mixed_images = lam * images + (1 - lam) * images[indices]

        # If targets are one-hot encoded
        if len(targets.shape) > 1:
            mixed_targets = lam * targets + (1 - lam) * targets[indices]
        else:
            # If targets are not one-hot (class indices)
            # Return both targets and the mixing parameter
            mixed_targets = (targets, targets[indices], lam)

        return mixed_images, mixed_targets


class CutMixTransform:
    """
    CutMix augmentation: cut and paste regions between images.

    Paper: https://arxiv.org/abs/1905.04899

    Args:
        alpha: Parameter for beta distribution
        p: Probability of applying augmentation
    """

    def __init__(self, alpha: float = 1.0, p: float = 0.5):
        self.alpha = alpha
        self.p = p
        logger.info(f"Initialized CutMix with alpha={alpha}, p={p}")

    def __call__(self, batch):
        """Apply CutMix to a batch of images and targets.

        Args:
            batch: Tuple of (images, targets)

        Returns:
            Tuple of mixed images and targets
        """
        if random.random() > self.p:
            return batch

        images, targets = batch
        batch_size, _, h, w = images.shape

        # Generate lambda from beta distribution
        lam = np.random.beta(self.alpha, self.alpha)

        # Create shuffled indices
        indices = torch.randperm(batch_size)

        # Calculate cut size
        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_ratio)
        cut_h = int(h * cut_ratio)

        # Calculate center position
        cx = np.random.randint(w)
        cy = np.random.randint(h)

        # Calculate box bounds
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        # Create mixed images
        mixed_images = images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[
            indices, :, bby1:bby2, bbx1:bbx2
        ]

        # Adjust lambda to account for actual cut region
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))

        # If targets are one-hot encoded
        if len(targets.shape) > 1:
            mixed_targets = lam * targets + (1 - lam) * targets[indices]
        else:
            # If targets are not one-hot (class indices)
            # Return both targets and the mixing parameter
            mixed_targets = (targets, targets[indices], lam)

        return mixed_images, mixed_targets
