"""
SHAP (SHapley Additive exPlanations) implementation for model interpretability.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from utils.logger import get_logger
from utils.paths import ensure_dir

logger = get_logger(__name__)


class SHAPInterpreter:
    """
    SHAP (SHapley Additive exPlanations) interpreter for explaining model predictions.
    Compatible with PlantDoc's CBAM-ResNet18 architecture.
    """

    def __init__(
        self,
        model: nn.Module,
        background_data: Optional[torch.Tensor] = None,
        num_background_samples: int = 100,
        input_size: Tuple[int, int] = (224, 224),
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        device: Optional[torch.device] = None,
        debug: bool = True,
    ):
        """
        Initialize SHAP interpreter.

        Args:
            model: PyTorch model to explain
            background_data: Background data for DeepExplainer (if None, random samples will be used)
            num_background_samples: Number of background samples to use if background_data is None
            input_size: Input image size
            mean: Normalization mean
            std: Normalization std
            device: Device to use (if None, will use the device of the model)
            debug: Whether to enable debug logging
        """
        self.model = model
        self.model.eval()

        # Get device from model if not specified
        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = device

        self.input_size = input_size
        self.mean = mean
        self.std = std
        self.debug = debug

        # Create the SHAP explainer
        self._create_explainer(background_data, num_background_samples)

        logger.info(f"Initialized SHAP interpreter for {model.__class__.__name__}")

    def _create_explainer(
        self,
        background_data: Optional[torch.Tensor],
        num_background_samples: int,
    ) -> None:
        """
        Create SHAP explainer with appropriate type for the model.

        Args:
            background_data: Background data for DeepExplainer
            num_background_samples: Number of background samples to use if background_data is None
        """
        # Handle unfreezing for backpropagation
        if hasattr(self.model, "frozen_backbone") and self.model.frozen_backbone:
            logger.info(
                "Model has frozen backbone. Will temporarily unfreeze for SHAP."
            )
            self.model.unfreeze_backbone()
            self.needs_refreeze = True
        else:
            self.needs_refreeze = False

        # If background data is not provided, create random samples
        if background_data is None:
            logger.info(f"Creating {num_background_samples} random background samples")
            background_data = torch.randn(
                num_background_samples, 3, *self.input_size, device=self.device
            )

            # Apply normalization to match the model's preprocessing
            for i in range(3):  # For each RGB channel
                background_data[:, i] = (
                    background_data[:, i] * self.std[i]
                ) + self.mean[i]

        # Move background data to the correct device
        background_data = background_data.to(self.device)

        try:
            # Create a wrapper function for the model that handles preprocessing and postprocessing
            # This is required for PyTorch models as SHAP needs a function that maps inputs to outputs
            def model_wrapper(x):
                with torch.no_grad():
                    # Convert to tensor and normalize if inputs are numpy arrays
                    if isinstance(x, np.ndarray):
                        x = torch.tensor(x, dtype=torch.float32, device=self.device)

                    # Ensure the model is in eval mode
                    self.model.eval()

                    # Forward pass
                    outputs = self.model(x)

                    # Handle models that return tuples
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]  # Assume first element is the logits

                    return outputs.cpu().numpy()

            # Choose the appropriate explainer based on the model architecture
            if (
                hasattr(self.model, "backbone")
                and "resnet" in str(self.model.backbone.__class__).lower()
            ):
                logger.info("Using DeepExplainer for ResNet-based architecture")
                # Need to modify the model to accept tensor inputs for DeepExplainer
                # Convert background data to float32 to ensure compatibility
                if isinstance(background_data, torch.Tensor):
                    background_data = background_data.float()

                # Wrap model to handle input format issues
                def model_wrapper(x):
                    # Ensure input is processed correctly
                    if isinstance(x, np.ndarray):
                        x = torch.tensor(x, dtype=torch.float32, device=self.device)
                    elif isinstance(x, torch.Tensor):
                        x = x.float()  # Ensure float type

                    # Forward pass
                    return self.model(x)

                # Create explainer with the model itself, not the wrapper function
                # This is the key fix - SHAP expects a model object, not a function
                self.explainer = shap.DeepExplainer(
                    model=self.model, data=background_data
                )
            else:
                logger.info("Using KernelExplainer as fallback")
                # Fetch a sample of background data for KernelExplainer
                background_sample = shap.sample(
                    background_data.cpu().numpy(), num_background_samples
                )
                self.explainer = shap.KernelExplainer(
                    model=model_wrapper, data=background_sample
                )

            if self.debug:
                logger.debug(f"Created SHAP explainer: {type(self.explainer).__name__}")

        except Exception as e:
            logger.error(f"Error creating SHAP explainer: {e}", exc_info=True)
            raise

    def preprocess_image(
        self, image_input: Union[str, Image.Image, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        Preprocess image for SHAP analysis.

        Args:
            image_input: Input image as path, PIL image, numpy array, or tensor

        Returns:
            Preprocessed tensor ready for the model
        """
        try:
            # Handle different input types
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    raise FileNotFoundError(f"Image file not found: {image_input}")
                image = Image.open(image_input).convert("RGB")

            elif isinstance(image_input, Image.Image):
                image = image_input

            elif isinstance(image_input, np.ndarray):
                # Handle numpy array
                if image_input.dtype == np.uint8:
                    image = Image.fromarray(image_input)
                else:
                    image = Image.fromarray((image_input * 255).astype(np.uint8))

            elif isinstance(image_input, torch.Tensor):
                # If already a tensor, just ensure correct formatting
                if image_input.ndim == 4:
                    if image_input.size(0) != 1:
                        logger.warning(
                            "Batch size > 1 detected, using only the first image"
                        )
                    return image_input[:1].to(self.device)
                elif image_input.ndim == 3:
                    return image_input.unsqueeze(0).to(self.device)
                else:
                    raise ValueError(f"Unexpected tensor shape: {image_input.shape}")

            else:
                raise TypeError(f"Unsupported image input type: {type(image_input)}")

            # Apply transforms
            from torchvision import transforms

            transform = transforms.Compose(
                [
                    transforms.Resize(
                        self.input_size,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.mean, std=self.std),
                ]
            )

            input_tensor = transform(image).unsqueeze(0).to(self.device)

            if self.debug:
                logger.debug(f"Preprocessed tensor shape: {input_tensor.shape}")

            return input_tensor

        except Exception as e:
            logger.error(f"Error in preprocess_image: {e}", exc_info=True)
            raise

    def get_shap_values(
        self,
        image_input: Union[str, Image.Image, np.ndarray, torch.Tensor],
        target_class: Optional[int] = None,
        nsamples: int = 100,
    ) -> Tuple[np.ndarray, torch.Tensor, int]:
        """
        Calculate SHAP values for an image.

        Args:
            image_input: Input image as path, PIL image, numpy array, or tensor
            target_class: Target class index (if None, uses the predicted class)
            nsamples: Number of samples for KernelExplainer

        Returns:
            Tuple of (shap_values, input_tensor, target_class)
        """
        try:
            # Preprocess image
            input_tensor = self.preprocess_image(image_input)

            # Get prediction if target_class is not specified
            if target_class is None:
                with torch.no_grad():
                    output = self.model(input_tensor)
                    if isinstance(output, tuple):
                        output = output[0]
                    target_class = output.argmax(dim=1).item()

                    if self.debug:
                        logger.debug(
                            f"Using predicted class {target_class} with confidence {torch.softmax(output, dim=1)[0, target_class].item():.4f}"
                        )

            # Get SHAP values
            if isinstance(self.explainer, shap.DeepExplainer):
                # DeepExplainer works with tensors directly
                shap_values = self.explainer.shap_values(input_tensor)

                # Convert shap_values to numpy arrays if they're tensors
                if isinstance(shap_values, list):
                    # List of arrays (one per class)
                    shap_values = [
                        sv.cpu().numpy() if isinstance(sv, torch.Tensor) else sv
                        for sv in shap_values
                    ]

                    # If we only want a specific class, extract it
                    if target_class is not None:
                        shap_values = shap_values[target_class]
                else:
                    # Already a single array
                    if isinstance(shap_values, torch.Tensor):
                        shap_values = shap_values.cpu().numpy()

            elif isinstance(self.explainer, shap.KernelExplainer):
                # KernelExplainer expects numpy arrays
                input_np = input_tensor.cpu().numpy()
                shap_values = self.explainer.shap_values(input_np, nsamples=nsamples)

                # If target_class is specified, return only those values
                if target_class is not None and isinstance(shap_values, list):
                    shap_values = shap_values[target_class]

            else:
                raise TypeError(f"Unsupported explainer type: {type(self.explainer)}")

            return shap_values, input_tensor, target_class

        except Exception as e:
            logger.error(f"Error in get_shap_values: {e}", exc_info=True)
            raise

    def _prepare_input(self, image_input):
        """
        Prepare image input for SHAP analysis.

        Args:
            image_input: Image tensor, PIL Image, or path to image

        Returns:
            Tuple of (input_tensor, original_image)
        """
        import numpy as np
        import torch
        from PIL import Image
        from torchvision import transforms

        # If image_input is a string, assume it's a file path
        if isinstance(image_input, str):
            try:
                # Load image and convert to RGB
                original_image = Image.open(image_input).convert("RGB")

                # Create transform with normalization
                transform = transforms.Compose(
                    [
                        transforms.Resize(self.input_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=self.mean, std=self.std),
                    ]
                )

                # Apply transform and add batch dimension
                input_tensor = transform(original_image).unsqueeze(0)

                return input_tensor, original_image

            except Exception as e:
                logger.error(f"Error loading image from path: {e}")
                raise ValueError(f"Could not load image from path: {image_input}")

        # If image_input is a tensor
        elif isinstance(image_input, torch.Tensor):
            # Handle batch dimension
            if image_input.dim() == 3:
                input_tensor = image_input.unsqueeze(0)
            else:
                input_tensor = image_input

            # Convert to numpy for visualization
            if input_tensor.shape[1] == 3:  # CHW format
                img_np = input_tensor[0].permute(1, 2, 0).cpu().numpy()

                # Denormalize if needed
                if img_np.max() <= 1.0:
                    # Handle normalization
                    mean = np.array(self.mean).reshape(1, 1, 3)
                    std = np.array(self.std).reshape(1, 1, 3)
                    img_np = img_np * std + mean
                    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

                original_image = Image.fromarray(img_np)
            else:
                # Fallback if format is unknown
                original_image = Image.fromarray(
                    np.zeros(
                        (self.input_size[0], self.input_size[1], 3), dtype=np.uint8
                    )
                )

            return input_tensor, original_image

        # If image_input is a PIL Image
        elif hasattr(image_input, "convert"):  # PIL Image
            original_image = image_input

            # Create transform with normalization
            transform = transforms.Compose(
                [
                    transforms.Resize(self.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.mean, std=self.std),
                ]
            )

            # Apply transform and add batch dimension
            input_tensor = transform(original_image).unsqueeze(0)

            return input_tensor, original_image

        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

    def visualize_shap(
        self,
        image_input: Union[str, Image.Image, np.ndarray, torch.Tensor],
        target_class: Optional[int] = None,
        output_path: Optional[str] = None,
        show: bool = False,
        figsize: Tuple[int, int] = (15, 5),
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Visualize SHAP values for a single image and target class.

        Args:
            image_input: Input image (can be path, PIL Image, numpy array, or tensor)
            target_class: Target class index (if None, uses prediction)
            output_path: Path to save the visualization
            show: Whether to display the visualization
            figsize: Size of the figure
            class_names: List of class names for display

        Returns:
            Dictionary with SHAP values and visualization data
        """
        try:
            # Prepare inputs
            input_tensor, original_image = self._prepare_input(image_input)

            # Run model to get target class if not provided
            if target_class is None:
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    target_class = outputs.argmax(dim=1).item()

            # Get class name if available
            if class_names and target_class < len(class_names):
                class_name = class_names[target_class]
            else:
                class_name = f"Class {target_class}"

            # Generate SHAP values
            shap_values = self.explainer(input_tensor.to(self.device).cpu().numpy())

            # Process SHAP values
            if isinstance(shap_values, list):
                # Multiple classes, use the target class
                if target_class < len(shap_values):
                    target_shap_values = shap_values[target_class]
                else:
                    logger.warning(
                        f"Target class {target_class} is out of range. Using class 0."
                    )
                    target_shap_values = shap_values[0]
            else:
                # Already for the target class
                target_shap_values = shap_values

            # Create visualization with dark theme
            from core.visualization.base_visualization import (
                DEFAULT_THEME,
                apply_dark_theme,
            )

            theme = DEFAULT_THEME.copy()
            apply_dark_theme(theme)

            # Create subplot grid with dark theme
            fig, axes = plt.subplots(
                1, 3, figsize=figsize, facecolor=theme["background_color"]
            )

            for ax in axes:
                ax.set_facecolor(theme["background_color"])

            # Plot original image
            axes[0].imshow(original_image)
            axes[0].set_title("Original Image", color=theme["text_color"], fontsize=12)
            axes[0].axis("off")

            # Plot SHAP values
            # Remove any existing colorbar axes to avoid warnings
            for i in range(plt.gcf().get_axes()):
                if i >= len(axes):
                    plt.gcf().delaxes(plt.gcf().get_axes()[i])

            # Use matplotlib directly for the SHAP visualization
            abs_shap = np.abs(target_shap_values[0]).transpose(1, 2, 0).sum(axis=2)
            normalized_shap = abs_shap / abs_shap.max()

            # SHAP visualization
            im = axes[1].imshow(normalized_shap, cmap="hot")
            axes[1].set_title(
                f"SHAP Values ({class_name})", color=theme["text_color"], fontsize=12
            )
            axes[1].axis("off")
            cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            cbar.ax.tick_params(colors=theme["text_color"])

            # Create overlay of SHAP values on original image
            heatmap = plt.cm.hot(normalized_shap)[:, :, :3]  # Remove alpha channel
            alpha = 0.5
            overlaid = original_image * (1 - alpha) + heatmap * 255 * alpha
            overlaid = np.clip(overlaid, 0, 255).astype(np.uint8)

            axes[2].imshow(overlaid)
            axes[2].set_title(
                f"SHAP Overlay ({class_name})", color=theme["text_color"], fontsize=12
            )
            axes[2].axis("off")

            # Add overall title
            fig.suptitle(
                "SHAP Feature Attribution",
                fontsize=14,
                color=theme["text_color"],
                y=0.98,
            )

            plt.tight_layout()

            # Save if output path is provided
            if output_path:
                ensure_dir(Path(output_path).parent)
                plt.savefig(
                    output_path,
                    bbox_inches="tight",
                    dpi=400,
                    facecolor=theme["background_color"],
                )
                logger.info(f"Saved SHAP visualization to {output_path}")

            if not show:
                plt.close(fig)

            # Return results
            return {
                "shap_values": target_shap_values,
                "original_image": original_image,
                "normalized_shap": normalized_shap,
                "overlaid_image": overlaid,
                "target_class": target_class,
                "class_name": class_name,
            }

        except Exception as e:
            logger.error(f"Error in visualize_shap: {e}", exc_info=True)
            raise

    def visualize_multi_class(
        self,
        image_input: Union[str, Image.Image, np.ndarray, torch.Tensor],
        top_k: int = 3,
        output_path: Optional[str] = None,
        show: bool = False,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Visualize SHAP values for top-k predicted classes.

        Args:
            image_input: Input image as path, PIL image, numpy array, or tensor
            top_k: Number of top classes to visualize
            output_path: Path to save visualization (if None, doesn't save)
            show: Whether to show the plot
            class_names: Class names for visualization

        Returns:
            Dictionary with SHAP results and visualization data
        """
        try:
            # Preprocess image
            input_tensor = self.preprocess_image(image_input)

            # Get predictions for top-k classes
            with torch.no_grad():
                output = self.model(input_tensor)
                if isinstance(output, tuple):
                    output = output[0]

                probs = torch.softmax(output, dim=1)[0]
                top_probs, top_classes = torch.topk(probs, min(top_k, output.size(1)))

                top_probs = top_probs.cpu().numpy()
                top_classes = top_classes.cpu().numpy()

            # Get class names if available
            if class_names:
                top_class_names = [
                    class_names[idx] if idx < len(class_names) else f"Class {idx}"
                    for idx in top_classes
                ]
            else:
                top_class_names = [f"Class {idx}" for idx in top_classes]

            # Get SHAP values for all classes
            all_shap_values, _, _ = self.get_shap_values(image_input)

            # Get original image for visualization
            if isinstance(image_input, str):
                original_image = np.array(
                    Image.open(image_input).convert("RGB").resize(self.input_size)
                )
            elif isinstance(image_input, Image.Image):
                original_image = np.array(image_input.resize(self.input_size))
            elif isinstance(image_input, np.ndarray):
                original_image = np.array(
                    Image.fromarray(
                        image_input.astype(np.uint8)
                        if image_input.dtype != np.uint8
                        else image_input
                    ).resize(self.input_size)
                )
            elif isinstance(image_input, torch.Tensor):
                img_tensor = input_tensor[0].clone()
                for i in range(3):
                    img_tensor[i] = img_tensor[i] * self.std[i] + self.mean[i]
                img_np = img_tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                original_image = (img_np * 255).astype(np.uint8)

            # Create visualization grid
            nrows = len(top_classes) + 1  # +1 for the original image
            fig, axes = plt.subplots(nrows, 3, figsize=(15, 5 * nrows))

            # Plot original image in the first row
            axes[0, 0].imshow(original_image)
            axes[0, 0].set_title("Original Image")
            axes[0, 0].axis("off")

            # Hide other axes in the first row
            axes[0, 1].axis("off")
            axes[0, 2].axis("off")

            # Plot SHAP visualizations for each top class
            for i, (class_idx, class_name, prob) in enumerate(
                zip(top_classes, top_class_names, top_probs)
            ):
                row = i + 1  # Start from second row

                # Get SHAP values for this class
                if isinstance(all_shap_values, list):
                    # If we have a list of arrays (one per class)
                    if class_idx < len(all_shap_values):
                        target_shap_values = all_shap_values[class_idx]
                    else:
                        logger.warning(
                            f"Class index {class_idx} out of range for SHAP values"
                        )
                        continue
                else:
                    # If we have a single array
                    target_shap_values = all_shap_values

                # Process SHAP values - sum across channels and normalize
                abs_shap = np.abs(target_shap_values[0]).transpose(1, 2, 0).sum(axis=2)
                normalized_shap = abs_shap / abs_shap.max()

                # Original image (but only for the first class)
                if i == 0:
                    axes[row, 0].imshow(original_image)
                    axes[row, 0].set_title(
                        f"Predicted: {class_name}\nProbability: {prob:.4f}"
                    )
                else:
                    axes[row, 0].imshow(original_image)
                    axes[row, 0].set_title(
                        f"Predicted: {class_name}\nProbability: {prob:.4f}"
                    )
                axes[row, 0].axis("off")

                # SHAP heatmap
                im = axes[row, 1].imshow(normalized_shap, cmap="hot")
                axes[row, 1].set_title("SHAP Values")
                axes[row, 1].axis("off")
                plt.colorbar(im, ax=axes[row, 1], fraction=0.046, pad=0.04)

                # Overlay
                heatmap = plt.cm.hot(normalized_shap)[:, :, :3]
                alpha = 0.5
                overlaid = original_image * (1 - alpha) + heatmap * 255 * alpha
                overlaid = np.clip(overlaid, 0, 255).astype(np.uint8)

                axes[row, 2].imshow(overlaid)
                axes[row, 2].set_title("SHAP Overlay")
                axes[row, 2].axis("off")

            plt.tight_layout()

            # Save if output path is provided
            if output_path:
                ensure_dir(Path(output_path).parent)
                plt.savefig(output_path, bbox_inches="tight", dpi=400)
                logger.info(f"Saved multi-class SHAP visualization to {output_path}")

            if not show:
                plt.close(fig)

            # Return results
            return {
                "top_classes": [
                    {
                        "class_idx": idx,
                        "class_name": name,
                        "probability": prob,
                    }
                    for idx, name, prob in zip(top_classes, top_class_names, top_probs)
                ],
                "original_image": original_image,
                "shap_values": all_shap_values,
            }

        except Exception as e:
            logger.error(f"Error in visualize_multi_class: {e}", exc_info=True)
            raise

    def channel_shap(
        self,
        image_input: Union[str, Image.Image, np.ndarray, torch.Tensor],
        target_class: Optional[int] = None,
        output_path: Optional[str] = None,
        show: bool = False,
        figsize: Tuple[int, int] = (15, 5),
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Visualize SHAP values for each RGB channel separately.

        Args:
            image_input: Input image as path, PIL image, numpy array, or tensor
            target_class: Target class index (if None, uses predicted class)
            output_path: Path to save visualization (if None, doesn't save)
            show: Whether to show the plot
            figsize: Figure size
            class_names: Class names for visualization

        Returns:
            Dictionary with SHAP results and visualization data
        """
        try:
            # Get SHAP values
            shap_values, input_tensor, target_class = self.get_shap_values(
                image_input, target_class
            )

            # Get original image for visualization
            if isinstance(image_input, str):
                original_image = np.array(
                    Image.open(image_input).convert("RGB").resize(self.input_size)
                )
            elif isinstance(image_input, Image.Image):
                original_image = np.array(image_input.resize(self.input_size))
            elif isinstance(image_input, np.ndarray):
                original_image = np.array(
                    Image.fromarray(
                        image_input.astype(np.uint8)
                        if image_input.dtype != np.uint8
                        else image_input
                    ).resize(self.input_size)
                )
            elif isinstance(image_input, torch.Tensor):
                img_tensor = input_tensor[0].clone()
                for i in range(3):
                    img_tensor[i] = img_tensor[i] * self.std[i] + self.mean[i]
                img_np = img_tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                original_image = (img_np * 255).astype(np.uint8)

            # Get class name if available
            class_name = None
            if class_names and target_class < len(class_names):
                class_name = class_names[target_class]
            else:
                class_name = f"Class {target_class}"

            # Process SHAP values
            if isinstance(shap_values, list):
                if target_class < len(shap_values):
                    target_shap_values = shap_values[target_class]
                else:
                    logger.warning(
                        f"Target class {target_class} is out of range. Using class 0."
                    )
                    target_shap_values = shap_values[0]
            else:
                target_shap_values = shap_values

            # Get channel-wise SHAP values
            channel_names = ["Red", "Green", "Blue"]
            channel_shap_values = []

            # Extract each channel
            for c in range(3):
                channel_abs_shap = np.abs(target_shap_values[0, c])
                normalized = channel_abs_shap / (channel_abs_shap.max() + 1e-10)
                channel_shap_values.append(normalized)

            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=figsize)

            # Plot original image
            axes[0, 0].imshow(original_image)
            axes[0, 0].set_title(f"Original Image\nPredicted: {class_name}")
            axes[0, 0].axis("off")

            # Plot combined SHAP values
            abs_shap = np.abs(target_shap_values[0]).transpose(1, 2, 0).sum(axis=2)
            normalized_shap = abs_shap / (abs_shap.max() + 1e-10)

            im = axes[0, 1].imshow(normalized_shap, cmap="hot")
            axes[0, 1].set_title("Combined SHAP Values")
            axes[0, 1].axis("off")
            plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

            # Plot channel-wise SHAP values
            for c in range(3):
                # Create a blank RGB image
                channel_heatmap = np.zeros((*self.input_size, 3))

                # Set the values for this channel
                channel_heatmap[:, :, c] = channel_shap_values[c]

                # Add to the subplot at position (1, c)
                if c < 2:
                    im = axes[1, c].imshow(channel_shap_values[c], cmap="hot")
                    axes[1, c].set_title(f"{channel_names[c]} Channel SHAP")
                    axes[1, c].axis("off")
                    plt.colorbar(im, ax=axes[1, c], fraction=0.046, pad=0.04)

            # Create RGB composite visualization for the last subplot
            rgb_shap = np.stack(
                [
                    channel_shap_values[0],
                    channel_shap_values[1],
                    channel_shap_values[2],
                ],
                axis=2,
            )

            # Normalize the RGB composite
            rgb_shap = rgb_shap / (rgb_shap.max() + 1e-10)

            axes[1, 1].imshow(rgb_shap)
            axes[1, 1].set_title("RGB Composite SHAP")
            axes[1, 1].axis("off")

            plt.tight_layout()

            # Save if output path is provided
            if output_path:
                ensure_dir(Path(output_path).parent)
                plt.savefig(output_path, bbox_inches="tight", dpi=400)
                logger.info(f"Saved channel SHAP visualization to {output_path}")

            if not show:
                plt.close(fig)

            # Return results
            return {
                "shap_values": target_shap_values,
                "original_image": original_image,
                "normalized_shap": normalized_shap,
                "channel_shap_values": channel_shap_values,
                "target_class": target_class,
                "class_name": class_name,
            }

        except Exception as e:
            logger.error(f"Error in channel_shap: {e}", exc_info=True)
            raise

    def feature_importance(
        self,
        dataset: DataLoader,
        num_samples: int = 100,
        output_path: Optional[str] = None,
        show: bool = False,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate and visualize aggregate feature importance across a dataset.

        Args:
            dataset: DataLoader with images
            num_samples: Number of samples to use
            output_path: Path to save visualization (if None, doesn't save)
            show: Whether to show the plot
            class_names: Class names for visualization

        Returns:
            Dictionary with feature importance results
        """
        try:
            logger.info(f"Calculating feature importance across {num_samples} samples")

            # Collect samples
            samples = []
            sample_targets = []

            for batch in dataset:
                # Handle different batch formats
                if isinstance(batch, dict):
                    images = batch["image"]
                    targets = batch["label"]
                else:
                    images = batch[0]
                    targets = batch[1]

                # Add images to the list
                batch_size = images.size(0)
                for i in range(batch_size):
                    if len(samples) < num_samples:
                        samples.append(images[i])
                        sample_targets.append(targets[i].item())
                    else:
                        break

                if len(samples) >= num_samples:
                    break

            # Stack samples
            sample_tensor = torch.stack(samples).to(self.device)

            # Get SHAP values for all samples
            all_shap_values = self.explainer.shap_values(sample_tensor)

            # Process SHAP values based on the type of output
            if isinstance(all_shap_values, list):
                # List of arrays, one per class
                # Average across samples for each class
                class_importance = {}
                for class_idx, shap_values in enumerate(all_shap_values):
                    # Check if we have a matching class name
                    if class_names and class_idx < len(class_names):
                        class_name = class_names[class_idx]
                    else:
                        class_name = f"Class {class_idx}"

                    # Convert to numpy if it's a tensor
                    if isinstance(shap_values, torch.Tensor):
                        shap_values = shap_values.cpu().numpy()

                    # Get average absolute value across the samples
                    avg_abs_shap = np.abs(shap_values).mean(axis=0)

                    # Calculate channel-wise importance
                    channel_importance = {}
                    for c, channel_name in enumerate(["R", "G", "B"]):
                        channel_importance[channel_name] = float(avg_abs_shap[c].mean())

                    # Store the results
                    class_importance[class_name] = {
                        "channel_importance": channel_importance,
                        "spatial_importance": avg_abs_shap.sum(
                            axis=0
                        ),  # Sum across channels
                    }
            else:
                # Single array for all classes
                # Just average across samples
                avg_abs_shap = np.abs(all_shap_values).mean(axis=0)

                # Calculate channel-wise importance
                channel_importance = {}
                for c, channel_name in enumerate(["R", "G", "B"]):
                    channel_importance[channel_name] = float(avg_abs_shap[c].mean())

                # Store the results
                class_importance = {
                    "all_classes": {
                        "channel_importance": channel_importance,
                        "spatial_importance": avg_abs_shap.sum(
                            axis=0
                        ),  # Sum across channels
                    }
                }

            # Visualize feature importance
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))

            # Bar chart of channel importance for each class
            class_names_list = []
            r_values = []
            g_values = []
            b_values = []

            for class_name, importance in class_importance.items():
                # Skip if not a class importance dict
                if (
                    not isinstance(importance, dict)
                    or "channel_importance" not in importance
                ):
                    continue

                class_names_list.append(class_name)
                r_values.append(importance["channel_importance"]["R"])
                g_values.append(importance["channel_importance"]["G"])
                b_values.append(importance["channel_importance"]["B"])

            # Set colors for the RGB bars
            colors = ["red", "green", "blue"]

            # Create indices for the bars
            indices = np.arange(len(class_names_list))
            width = 0.25

            # Plot bars
            axes[0].bar(
                indices - width, r_values, width, color=colors[0], label="Red Channel"
            )
            axes[0].bar(
                indices, g_values, width, color=colors[1], label="Green Channel"
            )
            axes[0].bar(
                indices + width, b_values, width, color=colors[2], label="Blue Channel"
            )

            # Add labels and title
            axes[0].set_xlabel("Class")
            axes[0].set_ylabel("Average SHAP Value")
            axes[0].set_title("Channel Importance by Class")
            axes[0].set_xticks(indices)
            axes[0].set_xticklabels(class_names_list, rotation=45, ha="right")
            axes[0].legend()

            # Create a spatial heatmap of importance
            # Average across all classes
            avg_spatial_importance = np.mean(
                [
                    importance["spatial_importance"]
                    for importance in class_importance.values()
                    if isinstance(importance, dict)
                    and "spatial_importance" in importance
                ],
                axis=0,
            )

            # Normalize for visualization
            avg_spatial_importance = avg_spatial_importance / (
                avg_spatial_importance.max() + 1e-10
            )

            # Plot spatial importance
            im = axes[1].imshow(avg_spatial_importance, cmap="hot")
            axes[1].set_title("Spatial Feature Importance")
            axes[1].axis("off")
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

            plt.tight_layout()

            # Save if output path is provided
            if output_path:
                ensure_dir(Path(output_path).parent)
                plt.savefig(output_path, bbox_inches="tight", dpi=400)
                logger.info(f"Saved feature importance visualization to {output_path}")

            if not show:
                plt.close(fig)

            # Return results
            return {
                "class_importance": class_importance,
                "avg_spatial_importance": avg_spatial_importance.tolist(),
                "channel_importance": {
                    "R": float(np.mean(r_values)),
                    "G": float(np.mean(g_values)),
                    "B": float(np.mean(b_values)),
                },
            }

        except Exception as e:
            logger.error(f"Error in feature_importance: {e}", exc_info=True)
            raise

    def cleanup(self):
        """
        Clean up resources.
        """
        # Re-freeze backbone if it was initially frozen
        if self.needs_refreeze and hasattr(self.model, "freeze_backbone"):
            logger.info("Restoring frozen backbone state")
            self.model.freeze_backbone()
            self.needs_refreeze = False

    def __del__(self):
        """
        Cleanup when the object is deleted.
        """
        try:
            self.cleanup()
        except:
            pass


# Helper functions for easier usage


def explain_image_with_shap(
    model: nn.Module,
    image_path: str,
    target_class: Optional[int] = None,
    output_path: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    background_data: Optional[torch.Tensor] = None,
    num_background_samples: int = 100,
) -> Dict[str, Any]:
    """
    Explain an image with SHAP.

    Args:
        model: PyTorch model
        image_path: Path to the image
        target_class: Target class index (if None, uses predicted class)
        output_path: Path to save visualization (if None, doesn't save)
        class_names: Class names for visualization
        background_data: Background data for DeepExplainer
        num_background_samples: Number of background samples to use if background_data is None

    Returns:
        Dictionary with SHAP results
    """
    interpreter = SHAPInterpreter(
        model=model,
        background_data=background_data,
        num_background_samples=num_background_samples,
    )

    try:
        return interpreter.visualize_shap(
            image_input=image_path,
            target_class=target_class,
            output_path=output_path,
            class_names=class_names,
        )
    finally:
        interpreter.cleanup()


def compare_gradcam_and_shap(
    model: nn.Module,
    image_path: str,
    target_class: Optional[int] = None,
    output_path: Optional[str] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compare GradCAM and SHAP explanations for an image.

    Args:
        model: PyTorch model
        image_path: Path to the image
        target_class: Target class index (if None, uses predicted class)
        output_path: Path to save visualization (if None, doesn't save)
        class_names: Class names for visualization

    Returns:
        Dictionary with comparison results
    """
    # Import GradCAM
    from core.evaluation.interpretability import GradCAM

    # Create interpreters
    shap_interpreter = SHAPInterpreter(model=model)
    grad_cam = GradCAM(model=model)

    try:
        # Get SHAP values
        shap_result = shap_interpreter.visualize_shap(
            image_input=image_path,
            target_class=target_class,
            output_path=None,  # Don't save individual visualization
            class_names=class_names,
        )

        # Get GradCAM
        original_image, cam_heatmap, cam_overlaid = grad_cam.visualize(
            image_input=image_path,
            target_category=target_class,
            output_path=None,  # Don't save individual visualization
        )

        # Create comparison visualization
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 3, 1)
        plt.imshow(original_image)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(2, 3, 2)
        plt.imshow(shap_result["normalized_shap"], cmap="hot")
        plt.title("SHAP Heatmap")
        plt.axis("off")

        plt.subplot(2, 3, 3)
        plt.imshow(shap_result["overlaid_image"])
        plt.title("SHAP Overlay")
        plt.axis("off")

        plt.subplot(2, 3, 5)
        plt.imshow(cam_heatmap)
        plt.title("GradCAM Heatmap")
        plt.axis("off")

        plt.subplot(2, 3, 6)
        plt.imshow(cam_overlaid)
        plt.title("GradCAM Overlay")
        plt.axis("off")

        plt.tight_layout()

        # Save if output path is provided
        if output_path:
            ensure_dir(Path(output_path).parent)
            plt.savefig(output_path, bbox_inches="tight", dpi=400)
            logger.info(f"Saved comparison visualization to {output_path}")
            plt.close()

        # Return results
        return {
            "shap_result": shap_result,
            "gradcam_result": {
                "original_image": original_image,
                "heatmap": cam_heatmap,
                "overlaid": cam_overlaid,
            },
            "target_class": shap_result["target_class"],
            "class_name": shap_result["class_name"],
        }

    finally:
        shap_interpreter.cleanup()
        grad_cam.cleanup()


def batch_explain_with_shap(
    model: nn.Module,
    dataset: DataLoader,
    output_dir: Path,
    num_samples: int = 10,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Generate SHAP explanations for a batch of images.

    Args:
        model: PyTorch model
        dataset: DataLoader with images
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
        class_names: Class names for visualization

    Returns:
        Dictionary with batch explanation results
    """
    # Create interpreter
    interpreter = SHAPInterpreter(model=model)

    try:
        # Create output directory
        ensure_dir(output_dir)

        # Collect samples
        samples = []
        sample_targets = []
        sample_paths = []

        for batch in dataset:
            # Handle different batch formats
            if isinstance(batch, dict):
                images = batch["image"]
                targets = batch["label"]
                paths = batch.get("path", [""] * len(images))
            else:
                images = batch[0]
                targets = batch[1]
                paths = batch[2] if len(batch) > 2 else [""] * len(images)

            # Add images to the list
            batch_size = images.size(0)
            for i in range(batch_size):
                if len(samples) < num_samples:
                    samples.append(images[i])
                    sample_targets.append(targets[i].item())
                    sample_paths.append(paths[i])
                else:
                    break

            if len(samples) >= num_samples:
                break

        # Generate explanations
        results = []

        for i, (image, target, path) in enumerate(
            zip(samples, sample_targets, sample_paths)
        ):
            # Create a descriptive filename
            if path:
                base_name = os.path.basename(path)
                file_name = f"{i:02d}_{base_name}"
            else:
                file_name = f"{i:02d}_class_{target}.png"

            # Get target class name
            target_class_name = (
                class_names[target]
                if class_names and target < len(class_names)
                else f"Class {target}"
            )

            logger.info(
                f"Generating SHAP for sample {i + 1}/{len(samples)}: {target_class_name}"
            )

            # Generate visualization
            output_path = output_dir / f"shap_{file_name}"

            try:
                result = interpreter.visualize_shap(
                    image_input=image,
                    target_class=target,
                    output_path=str(output_path),
                    class_names=class_names,
                )

                # Store results
                results.append(
                    {
                        "index": i,
                        "target_class": target,
                        "target_class_name": target_class_name,
                        "output_path": str(output_path),
                    }
                )

                # Also generate channel-wise visualization
                channel_output_path = output_dir / f"shap_channels_{file_name}"
                interpreter.channel_shap(
                    image_input=image,
                    target_class=target,
                    output_path=str(channel_output_path),
                    class_names=class_names,
                )

            except Exception as e:
                logger.error(f"Error generating SHAP for sample {i}: {e}")

        # Generate feature importance visualization
        importance_output_path = output_dir / "feature_importance.png"
        interpreter.feature_importance(
            dataset=dataset,
            num_samples=min(100, len(dataset.dataset)),
            output_path=str(importance_output_path),
            class_names=class_names,
        )

        return {
            "num_explained_samples": len(results),
            "results": results,
            "visualization_dir": str(output_dir),
        }

    finally:
        interpreter.cleanup()


class ShapInterpreter:
    """
    Generate SHAP visualizations for model explanations.

    Args:
        model: PyTorch model to explain
        dataset: Dataset containing examples to explain
        class_names: List of class names for visualization
        device: Device to run the model on ('cpu', 'cuda', 'mps')
        output_dir: Directory to save visualizations
        num_samples: Number of background samples for SHAP explainer
        batch_size: Batch size for processing
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dataset: Dataset,
        class_names: List[str],
        device: str = "cpu",
        output_dir: Optional[Union[str, Path]] = None,
        num_samples: int = 100,
        batch_size: int = 32,
    ):
        self.model = model
        self.dataset = dataset
        self.class_names = class_names
        self.device = device
        self.num_samples = min(num_samples, len(dataset))
        self.batch_size = batch_size

        # Set output directory
        if output_dir is None:
            self.output_dir = Path("outputs/reports/plots/shap_analysis")
        else:
            self.output_dir = Path(output_dir) / "reports" / "plots" / "shap_analysis"

        ensure_dir(self.output_dir)
        logger.info(f"SHAP visualizations will be saved to {self.output_dir}")

        # Put model in eval mode
        self.model.eval()
        self.model.to(self.device)

    def _prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for SHAP analysis.

        Returns:
            Tuple of (background data, test data, test labels)
        """
        # Sample background data for SHAP
        indices = np.random.choice(len(self.dataset), self.num_samples, replace=False)
        background_dataset = Subset(self.dataset, indices)

        # Create a smaller test dataset for explanations
        test_size = min(20, len(self.dataset))
        test_indices = np.random.choice(len(self.dataset), test_size, replace=False)
        test_dataset = Subset(self.dataset, test_indices)

        # Create dataloaders
        background_loader = DataLoader(
            background_dataset, batch_size=self.batch_size, shuffle=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,  # Process one at a time for explanation
            shuffle=False,
        )

        # Extract background data
        background_data = []

        for batch in background_loader:
            # Handle different batch formats
            if isinstance(batch, dict):
                # Dataset returns a dictionary (like PlantDiseaseDataset)
                inputs = batch["image"]
            elif isinstance(batch, (list, tuple)) and len(batch) >= 1:
                # Dataset returns a tuple (input, target)
                inputs = batch[0]
            else:
                logger.error(f"Unexpected batch format: {type(batch)}")
                raise ValueError(f"Unexpected batch format: {type(batch)}")

            background_data.append(inputs.numpy())

        background_data = np.vstack(background_data)

        # Extract test data
        test_data = []
        test_labels = []

        for batch in test_loader:
            # Handle different batch formats
            if isinstance(batch, dict):
                # Dictionary format (like PlantDiseaseDataset)
                inputs = batch["image"]
                targets = batch["label"]
            elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                # Tuple format (inputs, targets)
                inputs = batch[0]
                targets = batch[1]
            else:
                logger.error(f"Unexpected batch format: {type(batch)}")
                raise ValueError(f"Unexpected batch format: {type(batch)}")

            test_data.append(inputs.numpy())

            # Handle different target formats
            if isinstance(targets, (list, tuple)):
                targets = torch.tensor(targets)

            test_labels.append(
                targets.numpy() if isinstance(targets, torch.Tensor) else targets
            )

        test_data = np.vstack(test_data)
        test_labels = (
            np.concatenate(test_labels) if len(test_labels) > 0 else np.array([])
        )

        return background_data, test_data, test_labels

    def _model_predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Model prediction function for SHAP explainer.

        Args:
            inputs: Batch of input images (numpy array or tensor)

        Returns:
            Model predictions
        """
        with torch.no_grad():
            # Convert to tensor if it's a numpy array
            if isinstance(inputs, np.ndarray):
                inputs_tensor = torch.tensor(
                    inputs, device=self.device, dtype=torch.float32
                )
            else:
                # If already a tensor, just ensure it's on the right device
                inputs_tensor = inputs.to(self.device)

            # Handle 2D input (single sample) by adding batch dimension
            if len(inputs_tensor.shape) == 2 or len(inputs_tensor.shape) == 3:
                inputs_tensor = inputs_tensor.unsqueeze(0)

            # Forward pass through the model
            outputs = self.model(inputs_tensor)

            # Handle different output formats
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Take first element if it's a tuple

            # Return as numpy array
            return outputs.cpu().numpy()

    def generate_visualizations(self) -> None:
        """
        Generate and save SHAP visualizations.
        """
        logger.info("Preparing data for SHAP analysis...")
        background_data, test_data, test_labels = self._prepare_data()

        logger.info("Creating SHAP explainer...")
        # Use GradientExplainer instead of DeepExplainer - it's compatible with PyTorch without TensorFlow
        try:
            # Convert numpy arrays to PyTorch tensors
            background_tensor = torch.tensor(
                background_data, dtype=torch.float32, device=self.device
            )
            test_tensor = torch.tensor(
                test_data, dtype=torch.float32, device=self.device
            )

            # Use PyTorch-compatible explainer
            explainer = shap.GradientExplainer(self.model, background_tensor)

            logger.info("Computing SHAP values with GradientExplainer...")
            # Get SHAP values for test data
            shap_values = explainer.shap_values(test_tensor)

            # Save summary plot
            logger.info("Generating summary plot...")
            plt.figure(figsize=(12, 8))

            # Handle different shapes of shap_values
            if isinstance(shap_values, list):
                # For multi-class models, shap_values is a list of arrays (one per class)
                shap.summary_plot(
                    shap_values,
                    test_data,
                    feature_names=[f"Channel {i}" for i in range(test_data.shape[1])],
                    class_names=self.class_names,
                    show=False,
                )
            else:
                # For single-output models, shap_values is a single array
                shap.summary_plot(
                    shap_values,
                    test_data,
                    feature_names=[f"Channel {i}" for i in range(test_data.shape[1])],
                    show=False,
                )

            plt.tight_layout()
            plt.savefig(
                self.output_dir / "shap_summary.png", bbox_inches="tight", dpi=400
            )
            plt.close()

            # Generate class-specific visualizations
            if isinstance(shap_values, list):
                # If we have class-specific SHAP values
                for i, class_name in enumerate(self.class_names):
                    if i < len(shap_values):
                        logger.info(f"Generating visualization for class: {class_name}")
                        # Generate class-specific visualization
                        plt.figure(figsize=(12, 8))

                        # Use absolute values to show overall feature importance
                        shap.summary_plot(
                            shap_values[i],
                            test_data,
                            feature_names=[
                                f"Channel {c}" for c in range(test_data.shape[1])
                            ],
                            plot_type="bar",
                            show=False,
                        )
                        plt.title(f"SHAP Feature Importance for {class_name}")
                        plt.tight_layout()
                        plt.savefig(
                            self.output_dir
                            / f"shap_importance_{i}_{class_name.replace(' ', '_')}.png",
                            bbox_inches="tight",
                            dpi=400,
                        )
                        plt.close()

            # Generate individual explanations for a few examples
            try:
                # Try to use image specific visualization
                for i in range(min(5, len(test_data))):
                    logger.info(f"Generating explanation for example {i}")

                    # Get the predicted class
                    with torch.no_grad():
                        pred = self.model(
                            torch.tensor(test_data[i : i + 1], device=self.device)
                        )
                        pred_class = torch.argmax(pred, dim=1).item()

                    true_class = test_labels[i]
                    true_class_name = (
                        self.class_names[true_class]
                        if true_class < len(self.class_names)
                        else f"Class {true_class}"
                    )
                    pred_class_name = (
                        self.class_names[pred_class]
                        if pred_class < len(self.class_names)
                        else f"Class {pred_class}"
                    )

                    # Create a figure with the original image and SHAP values
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                    # Original image - transpose from (C,H,W) to (H,W,C) for display
                    img = test_data[i].transpose(1, 2, 0)
                    # Denormalize if needed
                    if img.min() < 0 or img.max() > 1:
                        std = np.array([0.229, 0.224, 0.225])
                        mean = np.array([0.485, 0.456, 0.406])
                        img = img * std + mean
                        img = np.clip(img, 0, 1)

                    axes[0].imshow(img)
                    axes[0].set_title(f"Original\nTrue: {true_class_name}")
                    axes[0].axis("off")

                    # SHAP values for the predicted class
                    if isinstance(shap_values, list) and pred_class < len(shap_values):
                        sv = shap_values[pred_class][i]
                        # Sum across channels for a single image
                        sv_mag = np.abs(sv).sum(axis=0)
                        # Normalize for visualization
                        sv_norm = sv_mag / sv_mag.max()

                        # SHAP heatmap
                        axes[1].imshow(sv_norm, cmap="hot")
                        axes[1].set_title(f"SHAP Values\nPredicted: {pred_class_name}")
                        axes[1].axis("off")

                        # Overlay on original
                        heatmap = plt.cm.hot(sv_norm)
                        # Convert to RGB, dropping alpha channel
                        heatmap = heatmap[..., :3]

                        # Create a blended overlay
                        alpha = 0.7
                        overlay = img * alpha + heatmap * (1 - alpha)
                        overlay = np.clip(overlay, 0, 1)

                        axes[2].imshow(overlay)
                        axes[2].set_title("SHAP Overlay")
                        axes[2].axis("off")
                    else:
                        # If we don't have class-specific SHAP values
                        axes[1].text(
                            0.5,
                            0.5,
                            "SHAP values not available\nfor this class",
                            ha="center",
                            va="center",
                        )
                        axes[1].axis("off")
                        axes[2].text(
                            0.5, 0.5, "Overlay not available", ha="center", va="center"
                        )
                        axes[2].axis("off")

                    plt.tight_layout()
                    plt.savefig(
                        self.output_dir / f"shap_example_{i}.png",
                        bbox_inches="tight",
                        dpi=400,
                    )
                    plt.close()

            except Exception as e:
                logger.error(f"Error generating individual examples: {e}")
                # Fall back to simpler visualization if needed
                for i in range(min(5, len(test_data))):
                    try:
                        logger.info(f"Generating fallback explanation for example {i}")
                        plt.figure(figsize=(10, 6))

                        # Show original image with text information
                        img = test_data[i].transpose(1, 2, 0)
                        if img.min() < 0 or img.max() > 1:
                            std = np.array([0.229, 0.224, 0.225])
                            mean = np.array([0.485, 0.456, 0.406])
                            img = img * std + mean
                            img = np.clip(img, 0, 1)

                        plt.imshow(img)
                        plt.title(
                            f"Example {i}\nTrue: {self.class_names[test_labels[i]] if test_labels[i] < len(self.class_names) else f'Class {test_labels[i]}'}"
                        )
                        plt.axis("off")
                        plt.tight_layout()
                        plt.savefig(
                            self.output_dir / f"shap_simple_example_{i}.png",
                            bbox_inches="tight",
                            dpi=400,
                        )
                        plt.close()
                    except Exception as nested_e:
                        logger.error(
                            f"Error with fallback visualization for example {i}: {nested_e}"
                        )

        except Exception as e:
            logger.error(f"Error in SHAP visualization: {e}")
            # Create a simplified image to show there was an error
            plt.figure(figsize=(10, 6))
            plt.text(
                0.5,
                0.5,
                f"SHAP visualization error:\n{str(e)}",
                ha="center",
                va="center",
                fontsize=12,
            )
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(self.output_dir / "shap_error.png", dpi=400)
            plt.close()
            raise

        logger.info(f"SHAP visualizations saved to {self.output_dir}")


def generate_shap_visualizations(
    model: torch.nn.Module,
    dataset: Dataset,
    class_names: List[str],
    experiment_dir: Union[str, Path],
    device: str = "cpu",
    num_samples: int = 100,
    batch_size: int = 32,
    theme: str = "dark",
) -> None:
    """
    Generate SHAP visualizations for a model and save them to the experiment directory.

    Args:
        model: PyTorch model to explain
        dataset: Dataset containing examples to explain
        class_names: List of class names for visualization
        experiment_dir: Path to experiment directory
        device: Device to run the model on ('cpu', 'cuda', 'mps')
        num_samples: Number of background samples for SHAP explainer
        batch_size: Batch size for processing
        theme: Theme to use for visualization ('dark' or 'light')
    """
    output_dir = Path(experiment_dir) / "shap_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating SHAP visualizations with {theme} theme")

    # Create an interpreter object
    interpreter = SHAPInterpreter(
        model=model,
        num_background_samples=50,
    )

    # Prepare output directory
    ensure_dir(output_dir)

    # Sample images from the dataset for explanation
    logger.info(f"Sampling {num_samples} images for SHAP analysis")

    # List to store samples and their targets
    samples = []
    sample_targets = []
    sample_paths = []

    # Use a dataloader to batch-process
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    # Get images and labels for visualization
    for batch_images, batch_targets in dataloader:
        for i in range(len(batch_images)):
            samples.append(batch_images[i])
            sample_targets.append(batch_targets[i].item())

            # Try to get file path if available in dataset
            path = None
            if hasattr(dataset, "samples") and i < len(dataset.samples):
                path = dataset.samples[i][0]
            elif hasattr(dataset, "imgs") and i < len(dataset.imgs):
                path = dataset.imgs[i][0]
            elif hasattr(dataset, "filenames") and i < len(dataset.filenames):
                path = dataset.filenames[i]

            sample_paths.append(path)

            if len(samples) >= num_samples:
                break

        if len(samples) >= num_samples:
            break

    # Generate explanations
    results = []

    for i, (image, target, path) in enumerate(
        zip(samples, sample_targets, sample_paths)
    ):
        # Create a descriptive filename
        if path:
            base_name = os.path.basename(path)
            file_name = f"{i:02d}_{base_name}"
        else:
            file_name = f"{i:02d}_class_{target}.png"

        # Get target class name
        target_class_name = (
            class_names[target]
            if class_names and target < len(class_names)
            else f"Class {target}"
        )

        logger.info(
            f"Generating SHAP for sample {i + 1}/{len(samples)}: {target_class_name}"
        )

        # Generate visualization with selected theme
        output_path = output_dir / f"shap_{file_name}"

        try:
            # Apply dark theme through the visualize_shap method
            result = interpreter.visualize_shap(
                image_input=image,
                target_class=target,
                output_path=str(output_path),
                class_names=class_names,
            )

            # Store results
            results.append(
                {
                    "index": i,
                    "target_class": target,
                    "target_class_name": target_class_name,
                    "output_path": str(output_path),
                }
            )

            # Also generate channel-wise visualization
            channel_output_path = output_dir / f"shap_channels_{file_name}"
            interpreter.channel_shap(
                image_input=image,
                target_class=target,
                output_path=str(channel_output_path),
                class_names=class_names,
            )

        except Exception as e:
            logger.error(f"Error generating SHAP for sample {i}: {e}")

    logger.info(f"Generated SHAP visualizations for {len(results)} samples")

    return {
        "num_explained_samples": len(results),
        "results": results,
        "visualization_dir": str(output_dir),
    }
