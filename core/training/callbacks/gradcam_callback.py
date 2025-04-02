"""
GradCAM visualization callback for models.

This callback generates GradCAM visualizations with confidence scores
at specified epochs throughout training.
"""

import copy
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
        initial_frozen_epochs: Number of initial epochs to skip GradCAM generation
        sample_indices: Indices of samples to visualize (if None, will use first n_samples)
        frequency: Generate visualization every n epochs
        n_samples: Number of samples to visualize
        target_layer: Target layer for GradCAM (if None, will be automatically detected)
        input_size: Size to which input images will be resized
        mean: Mean values for image normalization
        std: Std values for image normalization
        debug: Whether to enable debug mode for GradCAM
    """

    priority = 150  # Run after other callbacks

    def __init__(
        self,
        gradcam_dir: Union[str, Path],
        test_data: Union[DataLoader, List[Tuple]],
        class_names: List[str],
        initial_frozen_epochs: int = 0,
        sample_indices: Optional[List[int]] = None,
        frequency: int = 20,
        n_samples: int = 5,
        target_layer: Optional[nn.Module] = None,
        input_size: Tuple[int, int] = (224, 224),
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        debug: bool = False,
    ):
        super().__init__()
        self.gradcam_dir = Path(gradcam_dir)
        self.test_data = test_data
        self.class_names = class_names
        self.initial_frozen_epochs = initial_frozen_epochs
        self.sample_indices = sample_indices
        self.frequency = frequency
        self.n_samples = n_samples
        self.target_layer = target_layer
        self.input_size = input_size
        self.mean = mean
        self.std = std
        self.debug_mode = debug

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

        # Skip GradCAM generation if not in fine-tuning mode
        is_fine_tuning = logs.get("is_fine_tuning", False)
        if not is_fine_tuning:
            logger.info(f"Skipping GradCAM for epoch {epoch + 1} (backbone frozen).")
            return

        # Check if we should generate visualizations at this epoch (frequency check)
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

        # Save current training state
        training_mode = model.training

        # Set model to eval mode for visualizations
        model.eval()

        # Check if the model has a frozen backbone
        model_has_frozen_backbone = False
        if hasattr(model, "frozen_backbone"):
            model_has_frozen_backbone = model.frozen_backbone

        # Create visualization model - use copy if model has frozen backbone
        if model_has_frozen_backbone:
            logger.info(
                "Creating a separate visualization model with unfrozen backbone"
            )
            # Clone the model for visualization to avoid modifying the training model
            try:
                vis_model = copy.deepcopy(model)
                if hasattr(vis_model, "unfreeze_backbone"):
                    vis_model.unfreeze_backbone()
                    logger.info("Visualization model backbone unfrozen successfully")
                else:
                    logger.warning(
                        "Visualization model has no unfreeze_backbone method"
                    )
            except Exception as e:
                logger.error(f"Error creating visualization model: {e}")
                vis_model = model  # Fallback to original model if cloning fails
        else:
            # Use original model if backbone is not frozen
            vis_model = model

        try:
            # Create debug output directory
            debug_dir = epoch_dir / "debug"
            os.makedirs(debug_dir, exist_ok=True)

            # Initialize GradCAM
            gradcam = GradCAM(
                model=vis_model,
                target_layer=self.target_layer,
                input_size=self.input_size,
                mean=self.mean,
                std=self.std,
                debug=self.debug_mode,
            )

            # Generate visualizations for each sample
            for idx, (image, target) in enumerate(self.samples):
                sample_dir = epoch_dir / f"sample_{idx}"
                os.makedirs(sample_dir, exist_ok=True)

                # Ensure the image is on the same device as the model
                device = next(vis_model.parameters()).device
                if isinstance(image, torch.Tensor):
                    image = image.to(device)

                # Forward pass to get predictions
                with torch.no_grad():
                    input_tensor = gradcam.preprocess_image(image)
                    output = vis_model(input_tensor)
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
                    if hasattr(target, "item"):  # For tensor
                        target_int = target.item()
                    elif (
                        hasattr(target, "__class__")
                        and "ListConfig" in target.__class__.__name__
                    ):
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
            logger.error(f"Error generating GradCAM visualizations: {e}", exc_info=True)

        finally:
            # Restore original training mode
            model.train(training_mode)

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

            # Set dark theme style
            plt.style.use("dark_background")

            # Create figure with dark background
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=400)
            fig.patch.set_facecolor("#09090b")

            # Set axes styling for dark theme
            for ax in axes:
                ax.set_facecolor("#09090b")
                ax.tick_params(colors="#f5f5f5")
                for spine in ax.spines.values():
                    spine.set_color("#23272e")

            # Plot original image
            axes[0].imshow(original_image)
            axes[0].set_title("Original Image", fontsize=12, color="#f8fafc")
            axes[0].axis("off")

            # Plot heatmap
            axes[1].imshow(heatmap)
            axes[1].set_title("GradCAM Heatmap", fontsize=12, color="#f8fafc")
            axes[1].axis("off")

            # Plot overlay
            axes[2].imshow(overlaid)
            title = f"{title_prefix}: {class_name}\nConfidence: {probability:.2%}"
            axes[2].set_title(title, fontsize=12, color="#b5fdbc")
            axes[2].axis("off")

            plt.tight_layout()

            # Save with high quality and dark theme
            plt.savefig(
                output_path,
                dpi=400,
                bbox_inches="tight",
                facecolor="#09090b",
                edgecolor="#23272e",
                pad_inches=0.1,
            )
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

            # Properly denormalize using ImageNet mean and std
            mean = torch.tensor(self.mean).view(3, 1, 1).to(img_tensor.device)
            std = torch.tensor(self.std).view(3, 1, 1).to(img_tensor.device)
            img_tensor = img_tensor * std + mean

            # Convert to numpy and proper RGB order
            img_np = img_tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)

            return np.array(Image.fromarray(img_np).resize(self.input_size))
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    def _generate_summary(self, epoch_dir: Path, epoch: int) -> None:
        """Generate an HTML summary of all visualizations for this epoch."""
        try:
            html_path = epoch_dir / "gradcam_summary.html"

            with open(html_path, "w") as f:
                f.write(
                    f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>GradCAM Visualizations - Epoch {epoch}</title>
    <link
      rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
    />
    <style>
      :root {{
        --bg-primary: #09090b;
        --bg-secondary: #09090b;
        --bg-tertiary: #09090b;
        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
        --text-tertiary: #94a3b8;
        --accent-primary: #777777;
        --accent-secondary: #ffffffb3;
        --accent-tertiary: #b5fdbc;
        --success: #b5fdbc;
        --warning: #f59e0b;
        --danger: #ef4444;
        --info: #3b82f6;
        --border-radius: 8px;
        --card-radius: 12px;
        --shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        --transition: all 0.3s ease;
        --border-color: #23272e;
      }}

      * {{
        margin: 0;
        padding: 0;
        box-sizing: border-box;
      }}

      body {{
        font-family: "SF Pro Display", -apple-system, BlinkMacSystemFont,
          "Segoe UI", Roboto, Oxygen, Ubuntu, sans-serif;
        background-color: var(--bg-primary);
        color: var(--text-primary);
        line-height: 1.6;
      }}

      .container {{
        max-width: 1400px;
        margin: 0 auto;
        padding: 0 1.5rem;
      }}

      .grid {{
        display: grid;
        grid-template-columns: 240px 1fr;
        gap: 1.5rem;
        min-height: 100vh;
      }}

      /* Sidebar */
      .sidebar {{
        background-color: var(--bg-secondary);
        border-right: 1px solid var(--border-color);
        padding: 1.5rem 0;
        position: sticky;
        top: 0;
        height: 100vh;
        overflow-y: auto;
      }}

      .sidebar-logo {{
        padding: 0 1.5rem;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
      }}

      .logo-icon {{
        width: 2.5rem;
        height: 2.5rem;
        background: linear-gradient(135deg, #777777, #b5fdbc);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        color: white;
      }}

      .logo-text {{
        font-size: 1.25rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        color: var(--text-primary);
      }}

      .nav-section {{
        margin-bottom: 1.5rem;
      }}

      .nav-heading {{
        padding: 0 1.5rem;
        margin-bottom: 0.75rem;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: var(--text-tertiary);
      }}

      .nav-items {{
        list-style-type: none;
      }}

      .nav-item {{
        padding: 0.75rem 1.5rem;
        cursor: pointer;
        transition: var(--transition);
        display: flex;
        align-items: center;
        gap: 0.75rem;
        border-left: 3px solid transparent;
      }}

      .nav-item:hover {{
        background-color: rgba(255, 255, 255, 0.05);
      }}

      .nav-item.active {{
        background-color: #23272e;
        border-left: 3px solid #b5fdbc;
      }}

      .nav-item i {{
        font-size: 1.1rem;
        color: #b5fdbc;
      }}

      .nav-link {{
        color: var(--text-primary);
        text-decoration: none;
        font-size: 0.95rem;
        font-weight: 500;
      }}

      /* Main Content */
      .main-content {{
        padding: 2rem 0;
      }}

      .header {{
        margin-bottom: 2rem;
      }}

      .header-title {{
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(
          90deg,
          var(--accent-primary),
          var(--accent-tertiary)
        );
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
      }}

      .header-subtitle {{
        color: var(--text-secondary);
        font-size: 1.1rem;
        max-width: 800px;
      }}

      .sample-container {{
        background-color: var(--bg-secondary);
        border-radius: var(--card-radius);
        padding: 1.5rem;
        box-shadow: var(--shadow);
        margin-bottom: 2rem;
      }}

      .sample-container h2 {{
        color: var(--accent-tertiary);
        margin-bottom: 1.5rem;
        font-size: 1.25rem;
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 0.5rem;
      }}

      .visualizations-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 1.5rem;
      }}

      .visualization {{
        background-color: var(--bg-tertiary);
        border-radius: var(--border-radius);
        overflow: hidden;
        transition: var(--transition);
      }}

      .visualization:hover {{
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.4);
      }}

      .visualization img {{
        width: 100%;
        display: block;
        border-radius: var(--border-radius) var(--border-radius) 0 0;
      }}

      .viz-info {{
        padding: 1rem;
        background-color: var(--bg-tertiary);
      }}

      .filename {{
        font-size: 0.9rem;
        color: var(--text-secondary);
        margin-top: 0.5rem;
      }}
    </style>
</head>
<body>
    <div class="grid">
        <!-- Sidebar -->
        <aside class="sidebar">
            <div class="sidebar-logo">
                <div class="logo-icon">
                    <i class="fas fa-eye"></i>
                </div>
                <div class="logo-text">GradCAM</div>
            </div>

            <div class="nav-section">
                <div class="nav-heading">Samples</div>
                <ul class="nav-items">
"""
                )

                # Add each sample to the sidebar
                sample_dirs = sorted(
                    [
                        d
                        for d in epoch_dir.iterdir()
                        if d.is_dir() and d.name.startswith("sample_")
                    ]
                )

                for i, sample_dir in enumerate(sample_dirs):
                    active = "active" if i == 0 else ""
                    f.write(
                        f"""                    <li class="nav-item {active}">
                        <i class="fas fa-image"></i>
                        <a href="#{sample_dir.name}" class="nav-link">{sample_dir.name}</a>
                    </li>
"""
                    )

                f.write(
                    """                </ul>
            </div>
        </aside>

        <!-- Main Content -->
        <main class="main-content">
            <div class="container">
                <header class="header">
                    <h1 class="header-title">GradCAM Visualizations</h1>
                    <p class="header-subtitle">
                        Epoch visualization of model attention areas for classification predictions.
                    </p>
                </header>
"""
                )

                # Add each sample
                for sample_dir in sample_dirs:
                    f.write(
                        f"""
                <div class="sample-container" id="{sample_dir.name}">
                    <h2>{sample_dir.name}</h2>
                    <div class="visualizations-grid">
"""
                    )

                    # Add each visualization
                    viz_files = sorted(
                        [f for f in sample_dir.iterdir() if f.suffix == ".png"]
                    )

                    for viz_file in viz_files:
                        rel_path = viz_file.relative_to(epoch_dir)
                        f.write(
                            f"""
                        <div class="visualization">
                            <img src="{rel_path}" alt="{viz_file.stem}">
                            <div class="viz-info">
                                <div class="filename">{viz_file.name}</div>
                            </div>
                        </div>
"""
                        )

                    f.write(
                        """
                    </div>
                </div>
"""
                    )

                f.write(
                    """
            </div>
        </main>
    </div>
    <script>
        // Add active class to sidebar items when clicked
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', () => {
                document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
                item.classList.add('active');
            });
        });
    </script>
</body>
</html>
"""
                )

            logger.info(f"Generated summary HTML: {html_path}")

        except Exception as e:
            logger.error(f"Error generating summary HTML: {e}")
