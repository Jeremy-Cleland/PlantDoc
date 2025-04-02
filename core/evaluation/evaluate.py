"""
Model evaluation functionality for plant disease classification.
"""

import time
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from sklearn.calibration import calibration_curve
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

try:
    from sklearn.manifold import TSNE

    TSNE_AVAILABLE = True
except ImportError:
    logger.warning(
        "TSNE not available. Install scikit-learn for dimensionality reduction."
    )
    TSNE_AVAILABLE = False

try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    logger.warning(
        "UMAP not available. Install umap-learn for improved dimensionality reduction."
    )
    UMAP_AVAILABLE = False


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

    # Process features if collected
    if all_features:
        all_features_tensor = torch.cat(all_features, dim=0)
        all_labels_tensor = torch.cat(all_labels, dim=0)
        all_scores_tensor = torch.cat(all_scores, dim=0) if all_scores else None

        # Keep a limited number of test images for visualization
        max_test_images = 50
        if all_inputs:
            # Sample evenly across batches
            total_samples = sum(inputs.size(0) for inputs in all_inputs)
            if total_samples > max_test_images:
                # Calculate indices to keep
                indices = np.linspace(0, total_samples - 1, max_test_images, dtype=int)

                # Concatenate then sample
                all_inputs_tensor = torch.cat(all_inputs, dim=0)
                all_inputs_tensor = all_inputs_tensor[indices]
            else:
                all_inputs_tensor = torch.cat(all_inputs, dim=0)
        else:
            all_inputs_tensor = None
    else:
        all_features_tensor = None
        all_labels_tensor = None
        all_scores_tensor = None
        all_inputs_tensor = None

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
            # Create evaluation_artifacts directory
            artifacts_dir = output_dir / "evaluation_artifacts"
            ensure_dir(artifacts_dir)

            # Save confusion matrix
            np.save(
                artifacts_dir / "confusion_matrix.npy",
                conf_matrix,
            )

            # Save predictions
            np.save(artifacts_dir / "predictions.npy", all_preds)
            np.save(artifacts_dir / "targets.npy", all_targets)
            np.save(artifacts_dir / "probabilities.npy", all_probs)

            # Save misclassified indices
            misclassified_indices = np.where(all_preds != all_targets)[0]
            np.save(artifacts_dir / "misclassified_indices.npy", misclassified_indices)
            logger.info(
                f"Saved {len(misclassified_indices)} misclassified indices to {artifacts_dir / 'misclassified_indices.npy'}"
            )

            # Save per-class metrics
            per_class_metrics = {
                class_name: {
                    k: v
                    for k, v in metrics.items()
                    if k in ["precision", "recall", "f1-score", "support"]
                }
                for class_name, metrics in class_report.items()
                if class_name not in ["accuracy", "macro avg", "weighted avg"]
            }
            np.save(artifacts_dir / "per_class_metrics.npy", per_class_metrics)
            logger.info(
                f"Saved per-class metrics for {len(per_class_metrics)} classes to {artifacts_dir / 'per_class_metrics.npy'}"
            )

            # Save calibration data
            try:
                # Compute calibration curves for each class
                n_classes = len(class_names)
                calibration_data = {}

                # For each class, compute the calibration curve
                for class_idx in range(n_classes):
                    # For binary case (convert to binary - this class vs. rest)
                    y_true_binary = (all_targets == class_idx).astype(int)
                    y_prob_binary = all_probs[:, class_idx]

                    # Compute calibration curve
                    prob_true, prob_pred = calibration_curve(
                        y_true_binary, y_prob_binary, n_bins=10, strategy="uniform"
                    )

                    # Store data
                    calibration_data[class_names[class_idx]] = {
                        "true_probs": prob_true,
                        "pred_probs": prob_pred,
                    }

                # Save calibration data
                np.save(artifacts_dir / "calibration_data.npy", calibration_data)
                logger.info(
                    f"Saved calibration data for {n_classes} classes to {artifacts_dir / 'calibration_data.npy'}"
                )
            except Exception as e:
                logger.error(f"Error computing calibration data: {e}")

            # Compute and save embeddings if features are available
            if (
                isinstance(all_features_tensor, torch.Tensor)
                and len(all_features_tensor) > 0
            ):
                try:
                    # Stack features if they're in a list
                    features_array = all_features_tensor.cpu().numpy()

                    # Check we have enough data for meaningful dimensionality reduction
                    min_samples = 5 * len(
                        class_names
                    )  # Rule of thumb - at least 5 samples per class

                    if len(features_array) >= min_samples:
                        # Create embeddings using available method
                        embeddings = None
                        method_used = "none"

                        # Try UMAP first if available (better quality)
                        if UMAP_AVAILABLE:
                            try:
                                reducer = umap.UMAP(
                                    n_components=2,
                                    n_neighbors=30,
                                    min_dist=0.1,
                                    metric="euclidean",
                                    random_state=42,
                                )
                                embeddings = reducer.fit_transform(features_array)
                                method_used = "umap"
                            except Exception as e:
                                logger.warning(
                                    f"UMAP failed: {e}. Falling back to t-SNE."
                                )

                        # Fall back to t-SNE if UMAP not available or failed
                        if embeddings is None and TSNE_AVAILABLE:
                            try:
                                tsne = TSNE(
                                    n_components=2,
                                    perplexity=min(30, len(features_array) - 1),
                                    random_state=42,
                                )
                                embeddings = tsne.fit_transform(features_array)
                                method_used = "t-sne"
                            except Exception as e:
                                logger.warning(f"t-SNE failed: {e}")

                        # Save embeddings if generated
                        if embeddings is not None:
                            embedding_data = {
                                "embeddings": embeddings,
                                "labels": all_labels_tensor.cpu().numpy(),
                                "method": method_used,
                            }
                            np.save(artifacts_dir / "embeddings.npy", embedding_data)
                            logger.info(
                                f"Saved 2D embeddings ({method_used}) for {len(embeddings)} samples to {artifacts_dir / 'embeddings.npy'}"
                            )
                    else:
                        logger.warning(
                            f"Not enough samples ({len(features_array)}) for dimensionality reduction. Need at least {min_samples}."
                        )
                except Exception as e:
                    logger.error(f"Error computing embeddings: {e}")

            # Save paths for error analysis
            if any(all_image_paths):
                with open(artifacts_dir / "image_paths.txt", "w") as f:
                    for path in all_image_paths:
                        f.write(f"{path}\n")

                # Save misclassifications for error analysis
                errors = [
                    (path, int(true), int(pred))
                    for path, true, pred in zip(all_image_paths, all_targets, all_preds)
                    if true != pred
                ]

                if errors:
                    with open(
                        artifacts_dir / "misclassifications.csv",
                        "w",
                    ) as f:
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
    if all_features_tensor is not None:
        features_path = output_dir / "evaluation_artifacts" / "features.npy"
        # Convert features to numpy (move to CPU first if needed)
        if all_features_tensor.is_cuda or all_features_tensor.device.type == "mps":
            all_features_tensor = all_features_tensor.cpu()
        features_np = all_features_tensor.detach().numpy()
        np.save(features_path, features_np)
        logger.info(
            f"Saved {len(features_np)} features with shape {features_np.shape} to {features_path}"
        )

    # Save true labels if needed for visualizations
    if all_labels_tensor is not None:
        labels_path = output_dir / "evaluation_artifacts" / "true_labels.npy"
        # Convert to numpy (move to CPU first if needed)
        if all_labels_tensor.is_cuda or all_labels_tensor.device.type == "mps":
            all_labels_tensor = all_labels_tensor.cpu()
        labels_np = all_labels_tensor.detach().numpy()
        np.save(labels_path, labels_np)
        logger.info(
            f"Saved {len(labels_np)} true labels with shape {labels_np.shape} to {labels_path}"
        )

    # Save predictions and scores for further analysis
    predictions_path = output_dir / "evaluation_artifacts" / "predictions.npy"
    np.save(predictions_path, all_preds)
    logger.info(f"Saved {len(all_preds)} predictions to {predictions_path}")

    if all_scores_tensor is not None:
        scores_path = output_dir / "evaluation_artifacts" / "scores.npy"
        scores_np = all_scores_tensor.detach().numpy()
        np.save(scores_path, scores_np)
        logger.info(f"Saved {len(scores_np)} prediction scores to {scores_path}")

    # Save a small set of test images for visualization
    if all_inputs_tensor is not None:
        # Save images to evaluation_artifacts directory
        artifacts_dir = output_dir / "evaluation_artifacts"
        ensure_dir(artifacts_dir)
        images_path = artifacts_dir / "test_images.npy"

        # Move to CPU and convert to numpy
        if all_inputs_tensor.is_cuda or all_inputs_tensor.device.type == "mps":
            all_inputs_tensor = all_inputs_tensor.cpu()
        sample_images = all_inputs_tensor.detach().numpy()

        np.save(images_path, sample_images)
        logger.info(f"Saved {len(sample_images)} test images to {images_path}")
    else:
        logger.warning("Could not save test images: all_inputs is not a tensor")

    # Log results
    logger.info(f"Evaluation completed in {report['eval_time']:.2f} seconds")
    logger.info(
        f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
    )

    # Try to extract and save attention maps if model has attention mechanisms
    try:
        from core.evaluation.visualization_data_saver import save_attention_maps

        logger.info("Checking if model has attention mechanisms...")
        save_attention_maps(
            model=model,
            dataloader=dataloader,
            output_dir=output_dir,
            device=device,
            max_samples=20,
            class_names=class_names if "class_names" in locals() else None,
        )
    except Exception as e:
        logger.warning(f"Could not extract attention maps: {e}")

    # Save a subset of true_labels with the same size as test_images
    # This prevents size mismatch errors in generate_plots.py
    try:
        true_labels_path = output_dir / "evaluation_artifacts" / "true_labels.npy"
        test_images_path = output_dir / "evaluation_artifacts" / "test_images.npy"
        
        if true_labels_path.exists() and test_images_path.exists():
            true_labels = np.load(true_labels_path)
            test_images = np.load(test_images_path)
            
            if len(true_labels) != len(test_images):
                logger.info(f"Size mismatch: test_images {len(test_images)} vs true_labels {len(true_labels)}")
                subset_labels_path = output_dir / "evaluation_artifacts" / "subset_true_labels.npy"
                subset_size = min(len(test_images), len(true_labels))
                np.save(subset_labels_path, true_labels[:subset_size])
                logger.info(f"Saved subset of {subset_size} true labels to match test_images size")
    except Exception as e:
        logger.warning(f"Could not create subset of true labels: {e}")

    return report
