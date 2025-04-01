"""
Attention map visualization command implementation.

This module provides functionality to visualize attention maps for CBAM models
and generate HTML reports to explore model attention patterns.
"""

import os
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torchvision import transforms

from core.models.registry import get_model_class, list_models
from core.visualization.attention_viz import generate_attention_report
from utils.config_utils import load_config
from utils.logger import get_logger

logger = get_logger(__name__)


def visualize_attention(
    model_name: str,
    config_path: str,
    checkpoint_path: Optional[str] = None,
    image_path: str = None,
    output_dir: str = "outputs/attention_viz",
    layers: Optional[str] = None,
):
    """
    Visualize attention maps for a CBAM model.

    Args:
        model_name: Name of the model to visualize
        config_path: Path to configuration file
        checkpoint_path: Path to model checkpoint (optional)
        image_path: Path to input image
        output_dir: Output directory for visualization
        layers: Specific layers to visualize (comma-separated string)

    Returns:
        Path to the generated report
    """
    # Load configuration
    cfg = load_config(config_path)

    # Configure logging
    logger.info(f"Visualizing attention maps for model: {model_name}")

    # Load model
    try:
        model_class = get_model_class(model_name)
    except ValueError as e:
        logger.error(f"Error: {e}")
        available_models = list(list_models().keys())
        logger.info(f"Available models: {', '.join(available_models)}")
        return None

    model = model_class(**cfg.model)

    # Load checkpoint if provided
    if checkpoint_path is not None:
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint file not found: {checkpoint_path}")
            return None

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            elif "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
            logger.info(f"Loaded checkpoint from: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return None

    # Load and preprocess image
    try:
        image = Image.open(image_path).convert("RGB")

        # Get preprocessing transforms
        preprocess = transforms.Compose(
            [
                transforms.Resize(
                    (cfg.preprocessing.resize[0], cfg.preprocessing.resize[1])
                ),
                transforms.CenterCrop(
                    (cfg.preprocessing.center_crop[0], cfg.preprocessing.center_crop[1])
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=cfg.preprocessing.normalize.mean,
                    std=cfg.preprocessing.normalize.std,
                ),
            ]
        )

        # Preprocess image
        input_tensor = preprocess(image)
        logger.info(f"Loaded image from: {image_path}")
    except Exception as e:
        logger.error(f"Error loading or preprocessing image: {e}")
        return None

    # Parse layers if provided
    layer_list = None
    if layers is not None:
        layer_list = layers.split(",")
        logger.info(f"Visualizing specific layers: {layer_list}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate attention visualization report
    try:
        report_path = generate_attention_report(
            model=model,
            image=input_tensor,
            output_dir=output_dir,
            layer_names=layer_list,
            title_prefix=f"{model_name} Attention",
        )

        if report_path:
            logger.info(f"Attention visualization report generated at: {report_path}")
            logger.info("Open the report in a web browser to view the visualizations.")
            return report_path
        else:
            logger.error("Failed to generate attention visualization report.")
            return None
    except Exception as e:
        logger.error(f"Error generating attention visualization: {e}")
        return None
