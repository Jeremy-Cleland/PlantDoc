"""
Factory function for models.
"""

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

from core.models.registry import get_model_class
from utils.logger import get_logger

logger = get_logger(__name__)


def get_model(
    model_name: str,
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: str = "cpu",
    **kwargs,
) -> nn.Module:
    """
    Create and optionally load weights for a model.

    Args:
        model_name: Name of the model in the registry
        checkpoint_path: Optional path to model checkpoint
        device: Device to load model on ('cpu', 'cuda', 'mps')
        **kwargs: Additional parameters to pass to the model constructor

    Returns:
        Instantiated model
    """
    logger.info(f"Creating model: {model_name}")

    try:
        # Get model class from registry
        model_cls = get_model_class(model_name)

        # Create model instance
        model = model_cls(**kwargs)

        # Load checkpoint if provided
        if checkpoint_path is not None:
            logger.info(f"Loading weights from: {checkpoint_path}")
            checkpoint_path = Path(checkpoint_path)

            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            # Load state dict
            state_dict = torch.load(checkpoint_path, map_location="cpu")

            # Handle different checkpoint formats
            if isinstance(state_dict, dict):
                if "model" in state_dict:
                    state_dict = state_dict["model"]
                elif "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]

            # Remove 'module.' prefix if present (from DataParallel)
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

            # Load weights
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded checkpoint: {checkpoint_path}")

        # Move model to specified device
        model.to(device)

        return model

    except Exception as e:
        logger.error(f"Error creating model {model_name}: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise
