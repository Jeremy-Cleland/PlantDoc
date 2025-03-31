"""
GradCAM visualization callback for models.

This callback generates GradCAM visualizations with confidence scores
at specified epochs throughout training.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from core.evaluation.interpretability import GradCAM
from utils.logger import get_logger

from .base import Callback

logger = get_logger(__name__)


class GradCAMCallback(Callback):
    """
    Callback to generate GradCAM visualizations with confidence scores.
    Visualizations are generated at specified epochs.

    Args:
        gradcam_dir: Directory to save GradCAM visualizations
        test_data: DataLoader or list of (image, label) tuples for visualization
        class_names: List of class names
        sample_indices: Indices of samples to visualize (if None, will use first n_samples)
        frequency: Generate visualization every n epochs
        n_samples: Number of samples to visualize
        target_layer: Target layer for GradCAM (if None, will be automatically detected)
        input_size: Size to which input images will be resized
        mean: Mean values for image normalization
        std: Std values for image normalization
    """

    priority = 150  # Run after other callbacks

    def __init__(
        self,
        gradcam_dir: Union[str, Path],
        test_data: Union[DataLoader, List[Tuple]],
        class_names: List[str],
        sample_indices: Optional[List[int]] = None,
        frequency: int = 20,
        n_samples: int = 5,
        target_layer: Optional[nn.Module] = None,
        input_size: Tuple[int, int] = (224, 224),
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
    ):
        super().__init__()
        self.gradcam_dir = Path(gradcam_dir)
        self.test_data = test_data
        self.class_names = class_names
        self.sample_indices = sample_indices
        self.frequency = frequency
        self.n_samples = n_samples
        self.target_layer = target_layer
        self.input_size = input_size
        self.mean = mean
        self.std = std

        # Create gradcam directory
        os.makedirs(self.gradcam_dir, exist_ok=True)

        # Extract test samples to visualize
        self.samples = self._extract_test_samples()

        logger.info(
            f"Initialized GradCAMCallback: "
            f"frequency={self.frequency}, "
            f"n_samples={len(self.samples)}"
        )

    def _extract_test_samples(self) -> List[Tuple]:
        """Extract test samples for visualization."""
        samples = []

        if isinstance(self.test_data, DataLoader):
            # Extract samples from DataLoader
            all_samples = []
            for batch in self.test_data:
                # Handle different DataLoader formats
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    images, labels = batch[0], batch[1]
                    for i in range(len(images)):
                        all_samples.append((images[i], labels[i]))
                else:
                    logger.warning(f"Unexpected batch format: {type(batch)}")

                if len(all_samples) >= self.n_samples * 2:  # Get more than needed
                    break

            # Select samples based on indices
            if self.sample_indices:
                samples = [
                    all_samples[i] for i in self.sample_indices if i < len(all_samples)
                ]
            else:
                samples = all_samples[: self.n_samples]
        else:
            # Assume test_data is already a list of (image, label) tuples
            if self.sample_indices:
                samples = [
                    self.test_data[i]
                    for i in self.sample_indices
                    if i < len(self.test_data)
                ]
            else:
                samples = self.test_data[: self.n_samples]

        return samples[: self.n_samples]  # Limit to n_samples

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Generate GradCAM visualizations at the end of specified epochs."""
        logs = logs or {}

        # Check if we should generate visualizations at this epoch
        if (
            epoch + 1
        ) % self.frequency != 0 and epoch + 1 != 1:  # First and every frequency epochs
            return

        logger.info(f"Generating GradCAM visualizations at epoch {epoch + 1}")

        # Create epoch directory
        epoch_dir = self.gradcam_dir / f"epoch_{epoch + 1:03d}"
        os.makedirs(epoch_dir, exist_ok=True)

        # Get the model
        model = logs.get("model")
        if model is None:
            logger.error(
                "Model not found in logs. Cannot generate GradCAM visualizations."
            )
            return

        # Set model to eval mode
        model.eval()

        try:
            # Initialize GradCAM
            gradcam = GradCAM(
                model=model,
                target_layer=self.target_layer,
                input_size=self.input_size,
                mean=self.mean,
                std=self.std,
            )

            # Generate visualizations for each sample
            for idx, (image, target) in enumerate(self.samples):
                sample_dir = epoch_dir / f"sample_{idx}"
                os.makedirs(sample_dir, exist_ok=True)

                # Ensure the image is on the same device as the model
                device = next(model.parameters()).device
                if isinstance(image, torch.Tensor):
                    image = image.to(device)

                # Forward pass to get predictions
                with torch.no_grad():
                    input_tensor = gradcam.preprocess_image(image)
                    output = model(input_tensor)
                    if isinstance(output, tuple):
                        output = output[0]

                    # Get probabilities
                    probs = torch.nn.functional.softmax(output, dim=1)[0]

                    # Get top-3 predictions
                    top_k = min(3, len(self.class_names))
                    top_probs, top_classes = torch.topk(probs, top_k)
                    top_probs = top_probs.cpu().numpy()
                    top_classes = top_classes.cpu().numpy()

                # Generate GradCAM for predicted class
                pred_class = int(top_classes[0])  # Convert to int
                pred_prob = top_probs[0]
                pred_path = sample_dir / f"gradcam_pred_{pred_class}.png"

                self._generate_visualization(
                    gradcam, image, pred_class, pred_prob, pred_path, "Predicted Class"
                )

                # Generate GradCAM for true class if available
                if target is not None:
                    # Convert target to Python int to avoid ListConfig issues
                    if hasattr(target, 'item'):  # For tensor
                        target_int = target.item()
                    elif hasattr(target, '__class__') and 'ListConfig' in target.__class__.__name__:
                        target_int = int(target)
                    else:
                        target_int = int(target)  # For native Python types

                    true_path = sample_dir / f"gradcam_true_{target_int}.png"
                    true_prob = probs[target_int].item()
                    self._generate_visualization(
                        gradcam, image, target_int, true_prob, true_path, "True Class"
                    )

                # Generate GradCAM for top-1 and top-2 if different from prediction
                for i, (cls, prob) in enumerate(zip(top_classes[1:3], top_probs[1:3])):
                    cls_int = int(cls)  # Convert to int
                    top_path = sample_dir / f"gradcam_top{i + 1}_{cls_int}.png"
                    self._generate_visualization(
                        gradcam, image, cls_int, prob, top_path, f"Top-{i + 1} Class"
                    )

            # Clean up GradCAM
            gradcam.cleanup()

            # Generate summary HTML
            self._generate_summary(epoch_dir, epoch + 1)

            logger.info(f"GradCAM visualizations saved to {epoch_dir}")

        except Exception as e:
            logger.error(f"Error generating GradCAM visualizations: {e}")

        finally:
            # Set model back to training mode
            model.train()

    def _generate_visualization(
        self,
        gradcam: GradCAM,
        image: Any,
        class_idx: int,
        probability: float,
        output_path: Path,
        title_prefix: str,
    ) -> None:
        """Generate and save a GradCAM visualization with confidence score."""
        try:
            # Ensure class_idx is a Python int
            class_idx = int(class_idx)
            
            # Get original image
            original_image = self._get_original_image(image)

            # Compute CAM using compute_cam method
            cam = gradcam.compute_cam(image, class_idx)

            # Create heatmap
            cmap = plt.get_cmap("jet")
            heatmap = cmap(cam)[:, :, :3]  # Remove alpha channel

            # Overlay heatmap on original image
            alpha = 0.5  # Transparency factor
            overlaid = original_image * (1 - alpha) + heatmap * 255 * alpha
            overlaid = overlaid.astype(np.uint8)

            # Get class name
            if class_idx < len(self.class_names):
                class_name = self.class_names[class_idx]
            else:
                class_name = f"Class {class_idx}"

            # Create figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(original_image)
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            axes[1].imshow(heatmap)
            axes[1].set_title("GradCAM Heatmap")
            axes[1].axis("off")

            axes[2].imshow(overlaid)
            axes[2].set_title(
                f"{title_prefix}: {class_name}\nConfidence: {probability:.2%}"
            )
            axes[2].axis("off")

            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

        except Exception as e:
            logger.error(f"Error generating visualization for class {class_idx}: {e}")

    def _get_original_image(self, image: Any) -> np.ndarray:
        """Convert image to numpy array for visualization."""
        if isinstance(image, str):
            return np.array(Image.open(image).convert("RGB").resize(self.input_size))
        elif isinstance(image, Image.Image):
            return np.array(image.resize(self.input_size))
        elif isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            return np.array(Image.fromarray(image).resize(self.input_size))
        elif isinstance(image, torch.Tensor):
            if image.ndim == 4:  # (B, C, H, W)
                img_tensor = image[0]
            else:  # (C, H, W)
                img_tensor = image

            # Denormalize and convert to numpy
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()

            # Handle normalized images
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)

            return np.array(Image.fromarray(img_np).resize(self.input_size))
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    def _generate_summary(self, epoch_dir: Path, epoch: int) -> None:
        """Generate an HTML summary of all visualizations for this epoch."""
        try:
            html_path = epoch_dir / "gradcam_summary.html"

            with open(html_path, "w") as f:
                f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>GradCAM Visualizations - Epoch {epoch}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .sample-container {{ margin-bottom: 40px; }}
        .visualization {{ margin-bottom: 20px; }}
        img {{ max-width: 100%; }}
        .filename {{ font-size: 0.8em; color: #666; }}
    </style>
</head>
<body>
    <h1>GradCAM Visualizations - Epoch {epoch}</h1>
""")

                # Add each sample
                sample_dirs = sorted(
                    [
                        d
                        for d in epoch_dir.iterdir()
                        if d.is_dir() and d.name.startswith("sample_")
                    ]
                )

                for sample_dir in sample_dirs:
                    f.write(f"""
    <div class="sample-container">
        <h2>{sample_dir.name}</h2>
""")

                    # Add each visualization
                    viz_files = sorted(
                        [f for f in sample_dir.iterdir() if f.suffix == ".png"]
                    )

                    for viz_file in viz_files:
                        rel_path = viz_file.relative_to(epoch_dir)
                        f.write(f"""
        <div class="visualization">
            <img src="{rel_path}" alt="{viz_file.stem}">
            <div class="filename">{viz_file.name}</div>
        </div>
""")

                    f.write("""
    </div>
""")

                f.write("""
</body>
</html>
""")

            logger.info(f"Generated summary HTML: {html_path}")

        except Exception as e:
            logger.error(f"Error generating summary HTML: {e}")
