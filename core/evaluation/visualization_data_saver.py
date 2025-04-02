"""
Utility for saving visualization data from model evaluation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.logger import get_logger
from utils.paths import ensure_dir

logger = get_logger(__name__)


def save_attention_maps(
    model: nn.Module,
    dataloader: DataLoader,
    output_dir: Path,
    device: torch.device,
    max_samples: int = 20,
    class_names: Optional[List[str]] = None,
) -> None:
    """
    Extract and save attention maps from a trained model.

    Args:
        model: Trained model with attention mechanism
        dataloader: DataLoader for test data
        output_dir: Directory to save attention maps
        device: Device to run inference on
        max_samples: Maximum number of samples to save
        class_names: List of class names
    """
    # Check if model has attention mechanisms
    if not hasattr(model, "get_attention_maps") and not hasattr(
        model, "attention_maps"
    ):
        logger.warning(
            "Model does not have attention maps method. Skipping attention visualization."
        )
        return

    # Ensure output directory exists
    attention_dir = output_dir / "attention_maps"
    ensure_dir(attention_dir)

    # Set model to eval mode
    model.eval()

    # Dictionary to store results
    attention_data = {
        "attention_maps": [],
        "image_indices": [],
        "true_labels": [],
        "pred_labels": [],
        "confidences": [],
    }

    # Collect samples with their attention maps
    samples_count = 0
    with torch.no_grad():
        for batch in dataloader:
            # Get data from batch (handle different batch formats)
            if isinstance(batch, dict):
                inputs = batch["image"].to(device)
                targets = batch["label"].to(device)
                img_paths = batch.get("path", [""] * inputs.size(0))
            else:
                # Assume tuple/list format (inputs, targets) or (inputs, targets, paths)
                inputs = batch[0].to(device)
                targets = batch[1].to(device)
                img_paths = batch[2] if len(batch) > 2 else [""] * inputs.size(0)

            # Forward pass
            if hasattr(model, "return_attention") and callable(
                getattr(model, "return_attention")
            ):
                # Use return_attention=True flag if available
                outputs = model(inputs, return_attention=True)
            else:
                # Standard forward pass
                outputs = model(inputs)

            # Extract outputs and attention maps
            if isinstance(outputs, dict):
                if "logits" in outputs:
                    logits = outputs["logits"]
                elif "output" in outputs:
                    logits = outputs["output"]
                else:
                    logits = list(outputs.values())[0]  # Use first value as default

                # Get attention maps (could be in outputs)
                attention_maps = outputs.get("attention_maps", None)
            else:
                logits = outputs
                attention_maps = None

            # If attention_maps not in outputs, try to get them from model attributes
            if attention_maps is None:
                if hasattr(model, "get_attention_maps") and callable(
                    getattr(model, "get_attention_maps")
                ):
                    attention_maps = model.get_attention_maps()
                elif hasattr(model, "attention_maps"):
                    attention_maps = model.attention_maps

            # If still no attention maps, skip this batch
            if attention_maps is None:
                continue

            # Process attention maps (convert to numpy if needed)
            if isinstance(attention_maps, torch.Tensor):
                attention_maps = attention_maps.cpu().numpy()
            elif isinstance(attention_maps, list) and all(
                isinstance(x, torch.Tensor) for x in attention_maps
            ):
                attention_maps = [x.cpu().numpy() for x in attention_maps]

            # Get predictions and confidences
            probs = torch.softmax(logits, dim=1)
            confidences, preds = torch.max(probs, dim=1)

            # Add samples to attention_data
            for i in range(len(inputs)):
                if samples_count >= max_samples:
                    break

                # Extract attention map for this sample
                if isinstance(attention_maps, list):
                    # For multiple attention layers, just use the last one
                    sample_attention = attention_maps[-1][i]
                elif len(attention_maps.shape) == 4:  # [B, H, W, C] or [B, C, H, W]
                    sample_attention = attention_maps[i]
                else:
                    # Skip if format is unexpected
                    logger.warning(
                        f"Unexpected attention map format: {attention_maps.shape}"
                    )
                    continue

                # Add to results
                attention_data["attention_maps"].append(sample_attention)
                attention_data["image_indices"].append(samples_count)
                attention_data["true_labels"].append(targets[i].cpu().item())
                attention_data["pred_labels"].append(preds[i].cpu().item())
                attention_data["confidences"].append(confidences[i].cpu().item())

                samples_count += 1

            if samples_count >= max_samples:
                break

    # Convert lists to numpy arrays
    attention_data["attention_maps"] = np.array(attention_data["attention_maps"])
    attention_data["image_indices"] = np.array(attention_data["image_indices"])
    attention_data["true_labels"] = np.array(attention_data["true_labels"])
    attention_data["pred_labels"] = np.array(attention_data["pred_labels"])
    attention_data["confidences"] = np.array(attention_data["confidences"])

    # Save attention data
    np.save(attention_dir / "attention_data.npy", attention_data)
    logger.info(
        f"Saved attention maps for {samples_count} samples to {attention_dir / 'attention_data.npy'}"
    )

    # Also save a visualization-ready version in the evaluation_artifacts directory
    artifacts_dir = output_dir / "evaluation_artifacts"
    ensure_dir(artifacts_dir)
    np.save(artifacts_dir / "attention_maps.npy", attention_data)
    logger.info(f"Saved attention maps to {artifacts_dir / 'attention_maps.npy'}")
