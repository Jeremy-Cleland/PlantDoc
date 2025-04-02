"""
SHAP Evaluatation for model interpretability.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader

from core.evaluation.shap_interpreter import (
    SHAPInterpreter,
    batch_explain_with_shap,
    compare_gradcam_and_shap,
)
from utils.logger import get_logger
from utils.paths import ensure_dir

logger = get_logger(__name__)


def evaluate_with_shap(
    model: torch.nn.Module,
    data_loader: DataLoader,
    num_samples: int = 10,
    output_dir: Optional[Union[str, Path]] = None,
    class_names: Optional[List[str]] = None,
    num_background_samples: int = 100,
    compare_with_gradcam: bool = True,
) -> Dict:
    """
    Evaluate a model using SHAP for interpretability.

    Args:
        model: PyTorch model to evaluate
        data_loader: DataLoader for evaluation data
        num_samples: Number of sample images to explain
        output_dir: Directory to save visualizations
        class_names: List of class names
        num_background_samples: Number of background samples for SHAP
        compare_with_gradcam: Whether to compare SHAP with GradCAM

    Returns:
        Dictionary with evaluation results
    """
    # Prepare model
    model.eval()
    device = next(model.parameters()).device

    # Create output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        ensure_dir(output_dir)
        shap_dir = output_dir / "shap_visualizations"
        ensure_dir(shap_dir)

        if compare_with_gradcam:
            comparison_dir = output_dir / "shap_gradcam_comparison"
            ensure_dir(comparison_dir)

    # Get class names if not provided
    if class_names is None:
        if hasattr(data_loader.dataset, "classes"):
            class_names = data_loader.dataset.classes
        elif hasattr(data_loader.dataset, "class_names"):
            class_names = data_loader.dataset.class_names
        else:
            logger.warning("No class names found. Using index numbers as class names.")
            if hasattr(model, "num_classes"):
                class_names = [f"Class_{i}" for i in range(model.num_classes)]
            else:
                class_names = [
                    f"Class_{i}" for i in range(100)
                ]  # Default to 100 classes max

    # Initialize SHAP interpreter
    logger.info("Initializing SHAP interpreter for model interpretation")
    interpreter = SHAPInterpreter(
        model=model,
        num_background_samples=num_background_samples,
    )

    try:
        # Generate batch explanations
        logger.info(f"Generating SHAP explanations for {num_samples} samples")

        batch_results = batch_explain_with_shap(
            model=model,
            dataset=data_loader,
            output_dir=shap_dir,
            num_samples=num_samples,
            class_names=class_names,
        )

        # Compare with GradCAM if requested
        if compare_with_gradcam:
            logger.info("Comparing SHAP with GradCAM")

            # Import GradCAM
            from core.evaluation.interpretability import (
                evaluate_model as evaluate_with_gradcam,
            )

            # Run GradCAM evaluation
            gradcam_results = evaluate_with_gradcam(
                model=model,
                data_loader=data_loader,
                num_samples=num_samples,
                output_dir=output_dir,
                class_names=class_names,
            )

            # Generate comparisons for the first few samples
            comparison_results = []

            # Collect sample images for visualization
            samples = []
            sample_targets = []
            sample_paths = []

            for batch in data_loader:
                if len(samples) >= num_samples:
                    break

                # Handle different batch formats
                if isinstance(batch, dict):
                    images = batch["image"]
                    targets = batch["label"]
                    paths = batch.get("path", [""] * len(images))
                else:
                    images = batch[0]
                    targets = batch[1]
                    paths = batch[2] if len(batch) > 2 else [""] * len(images)

                # Add images to visualization list
                batch_size = images.size(0)
                for i in range(batch_size):
                    if len(samples) < num_samples:
                        samples.append(images[i].cpu())
                        sample_targets.append(targets[i].item())
                        sample_paths.append(paths[i])
                    else:
                        break

            # Generate comparisons
            for idx, (image, target, img_path) in enumerate(
                zip(samples, sample_targets, sample_paths)
            ):
                if idx >= num_samples // 2:  # Only do comparisons for a subset
                    break

                # Create a descriptive filename
                if img_path:
                    base_name = os.path.basename(img_path)
                    file_name = f"{idx:02d}_{base_name}"
                else:
                    file_name = f"{idx:02d}_class_{target}.png"

                # Get the target class name
                target_class_name = (
                    class_names[target]
                    if target < len(class_names)
                    else f"Unknown_{target}"
                )

                try:
                    # Generate comparison
                    logger.info(
                        f"Generating comparison for sample {idx + 1}/{len(samples)}: {target_class_name}"
                    )

                    output_path = comparison_dir / f"comparison_{file_name}"

                    result = compare_gradcam_and_shap(
                        model=model,
                        image_path=image,
                        target_class=target,
                        output_path=str(output_path),
                        class_names=class_names,
                    )

                    comparison_results.append(
                        {
                            "index": idx,
                            "target_class": target,
                            "target_class_name": target_class_name,
                            "output_path": str(output_path),
                        }
                    )

                except Exception as e:
                    logger.error(f"Error generating comparison for sample {idx}: {e}")

            batch_results["comparisons"] = comparison_results

        logger.info("SHAP model interpretation completed")
        return batch_results

    finally:
        # Clean up
        interpreter.cleanup()
