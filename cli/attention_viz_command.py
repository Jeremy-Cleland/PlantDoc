"""
Command-line interface for visualizing attention maps.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import torch
import typer
from PIL import Image
from torchvision import transforms

from core.models.registry import get_model_class, list_models
from core.visualization.attention_viz import generate_attention_report
from utils.config_utils import load_config
from utils.logger import configure_logging, get_logger

app = typer.Typer(help="CBAM Attention Visualization CLI")
logger = get_logger(__name__)

# Define options as module-level variables
MODEL_OPTION = typer.Option(..., "--model", "-m", help="Model name to visualize")
CONFIG_OPTION = typer.Option(
    "configs/config.yaml", "--config", "-c", help="Path to configuration file"
)
CHECKPOINT_OPTION = typer.Option(
    None, "--checkpoint", "-ckpt", help="Path to model checkpoint"
)
IMAGE_OPTION = typer.Option(..., "--image", "-i", help="Path to input image")
OUTPUT_OPTION = typer.Option(
    "outputs/attention_viz", "--output", "-o", help="Output directory for visualization"
)
LAYERS_OPTION = typer.Option(
    None, "--layers", "-l", help="Specific layers to visualize (e.g., layer1,layer2)"
)


@app.command()
def visualize(
    model_name: str = MODEL_OPTION,
    config_path: str = CONFIG_OPTION,
    checkpoint_path: Optional[str] = CHECKPOINT_OPTION,
    image_path: str = IMAGE_OPTION,
    output_dir: str = OUTPUT_OPTION,
    layers: Optional[str] = LAYERS_OPTION,
):
    """
    Visualize attention maps for a CBAM model.
    """
    # Load configuration
    cfg = load_config(config_path)

    # Configure logging
    logger = configure_logging(cfg)
    logger.info(f"Visualizing attention maps for model: {model_name}")

    # Load model
    try:
        model_class = get_model_class(model_name)
    except ValueError as e:
        logger.error(f"Error: {e}")
        available_models = list(list_models().keys())
        logger.info(f"Available models: {', '.join(available_models)}")
        raise typer.Exit(code=1)

    model = model_class(**cfg.model)

    # Load checkpoint if provided
    if checkpoint_path is not None:
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint file not found: {checkpoint_path}")
            raise typer.Exit(code=1)

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            logger.info(f"Loaded checkpoint from: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            raise typer.Exit(code=1)

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
        raise typer.Exit(code=1)

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
        else:
            logger.error("Failed to generate attention visualization report.")
            raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"Error generating attention visualization: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    # Fix path for imports
    sys.path.insert(0, str(Path(__file__).parents[1]))
    app()
