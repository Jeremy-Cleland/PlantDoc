"""
Model evaluation functionality for plant disease classification.
"""

import time
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.logger import get_logger
from utils.paths import ensure_dir

logger = get_logger(__name__)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    checkpoint_path: Optional[Union[str, Path]] = None,
    cfg: Optional[DictConfig] = None,
    output_dir: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
) -> Dict:
    """
    Evaluate a model on a dataset.

    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader for evaluation data
        checkpoint_path: Path to model checkpoint to load
        cfg: Configuration object
        output_dir: Directory to save evaluation results
        device: Device to run evaluation on ('cpu', 'cuda', 'mps')

    Returns:
        Dictionary of evaluation metrics
    """
    # Set up device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if not torch.cuda.is_available() and torch.backends.mps.is_available():
            device = "mps"
    device = torch.device(device)
    logger.info(f"Evaluating model on device: {device}")

    # Load checkpoint if provided
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        logger.info(f"Loading model checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        elif isinstance(checkpoint, dict) and all(
            k.startswith("model.") for k in checkpoint.keys()
        ):
            # Lightning checkpoint format
            state_dict = {
                k[6:]: v for k, v in checkpoint.items() if k.startswith("model.")
            }
            model.load_state_dict(state_dict)
        else:
            # Assume it's a direct state dict
            model.load_state_dict(checkpoint)

        logger.info("Checkpoint loaded successfully")

    # Move model to device
    model.to(device)
    model.eval()

    # Create output directory if needed
    if output_dir is not None:
        output_dir = Path(output_dir)
        ensure_dir(output_dir)

    # Prepare for evaluation
    all_preds = []
    all_targets = []
    all_probs = []
    all_image_paths = []
    all_features = []
    all_labels = []
    all_scores = []
    all_inputs = []

    # Disable gradient computation for evaluation
    with torch.no_grad():
        start_time = time.time()
        for batch in tqdm(dataloader, desc="Evaluating"):
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
            outputs = model(inputs)

            # Handle different output formats (logits or dict with logits)
            if isinstance(outputs, dict) and "logits" in outputs:
                logits = outputs["logits"]
            else:
                logits = outputs

            # Get predictions and probabilities
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            # Store results
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_image_paths.extend(img_paths)

            # Extract features
            if hasattr(model, "features"):
                features = model.features(inputs)
                all_features.append(features.cpu())
                all_labels.append(targets.cpu())
                all_scores.append(logits.cpu())
                all_inputs.append(inputs.cpu())

    # Combine results
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)

    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="weighted"
    )

    # Create report dictionary
    report = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "eval_time": time.time() - start_time,
        "num_samples": len(all_targets),
    }

    # Get per-class metrics
    if hasattr(dataloader.dataset, "class_names") or (
        cfg and hasattr(cfg, "data") and hasattr(cfg.data, "class_names")
    ):
        # Get class names
        if hasattr(dataloader.dataset, "class_names"):
            class_names = dataloader.dataset.class_names
        elif hasattr(dataloader.dataset, "classes"):
            class_names = dataloader.dataset.classes
        else:
            class_names = cfg.data.class_names

        # Generate detailed classification report
        class_report = classification_report(
            all_targets, all_preds, target_names=class_names, output_dict=True
        )

        # Add per-class metrics to report
        report["per_class_metrics"] = class_report

        # Calculate confusion matrix
        conf_matrix = confusion_matrix(all_targets, all_preds)
        report["confusion_matrix"] = conf_matrix.tolist()

        # Save detailed report if output directory is provided
        if output_dir is not None:
            # Save confusion matrix
            np.save(output_dir / "confusion_matrix.npy", conf_matrix)

            # Save predictions
            np.save(output_dir / "predictions.npy", all_preds)
            np.save(output_dir / "targets.npy", all_targets)
            np.save(output_dir / "probabilities.npy", all_probs)

            # Save paths for error analysis
            if any(all_image_paths):
                with open(output_dir / "image_paths.txt", "w") as f:
                    for path in all_image_paths:
                        f.write(f"{path}\n")

                # Save misclassifications for error analysis
                errors = [
                    (path, int(true), int(pred))
                    for path, true, pred in zip(all_image_paths, all_targets, all_preds)
                    if true != pred
                ]

                if errors:
                    with open(output_dir / "misclassifications.csv", "w") as f:
                        f.write("path,true_label,pred_label,true_class,pred_class\n")
                        for path, true, pred in errors:
                            true_class = (
                                class_names[true]
                                if true < len(class_names)
                                else f"Unknown_{true}"
                            )
                            pred_class = (
                                class_names[pred]
                                if pred < len(class_names)
                                else f"Unknown_{pred}"
                            )
                            f.write(f"{path},{true},{pred},{true_class},{pred_class}\n")

    # Save features for visualization if needed
    if isinstance(all_features, torch.Tensor):
        features_path = output_dir / "features.npy"
        # Convert features to numpy (move to CPU first if needed)
        if all_features.is_cuda or all_features.device.type == "mps":
            all_features = all_features.cpu()
        features_np = all_features.detach().numpy()
        np.save(features_path, features_np)
        logger.info(
            f"Saved {len(features_np)} features with shape {features_np.shape} to {features_path}"
        )

    # Save true labels if needed for visualizations
    if isinstance(all_labels, torch.Tensor):
        labels_path = output_dir / "true_labels.npy"
        # Convert to numpy (move to CPU first if needed)
        if all_labels.is_cuda or all_labels.device.type == "mps":
            all_labels = all_labels.cpu()
        labels_np = all_labels.detach().numpy()
        np.save(labels_path, labels_np)
        logger.info(
            f"Saved {len(labels_np)} true labels with shape {labels_np.shape} to {labels_path}"
        )

    # Save predictions and scores for further analysis
    predictions_path = output_dir / "predictions.npy"
    np.save(predictions_path, all_preds)
    logger.info(f"Saved {len(all_preds)} predictions to {predictions_path}")

    if all_scores is not None:
        scores_path = output_dir / "scores.npy"
        np.save(scores_path, all_scores)
        logger.info(f"Saved {len(all_scores)} prediction scores to {scores_path}")

    # Save a small set of test images for visualization
    if isinstance(all_inputs, torch.Tensor):
        max_images = 50  # Save up to 50 sample images
        if len(all_inputs) > max_images:
            # Use evenly spaced indices to get representative samples
            indices = np.linspace(0, len(all_inputs) - 1, max_images, dtype=int)
            sample_images = all_inputs[indices]
        else:
            sample_images = all_inputs

        # Move to CPU and convert to numpy
        if sample_images.is_cuda or sample_images.device.type == "mps":
            sample_images = sample_images.cpu()
        sample_images = sample_images.detach().numpy()

        # Save images
        images_path = output_dir / "test_images.npy"
        np.save(images_path, sample_images)
        logger.info(f"Saved {len(sample_images)} test images to {images_path}")

    else:
        logger.warning("Could not save test images: all_inputs is not a tensor")

    # Log results
    logger.info(f"Evaluation completed in {report['eval_time']:.2f} seconds")
    logger.info(
        f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
    )

    return report
