# Path: plantdoc/core/data/transforms.py
# Description: Image transformations using Albumentations library


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

            # Add CenterCrop if no crop operation has been added yet
            if not any(
                isinstance(t, (A.RandomResizedCrop, A.RandomCrop))
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
                logger.warning("Input image is grayscale, converting to RGB for transform.")
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
