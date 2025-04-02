"""
Callback to save data needed for enhanced visualizations in reports.

This callback saves the necessary files for our enhanced visualization functions:
- predictions.npy - Model predictions on the validation set
- features.npy - Feature vectors from the model
- true_labels.npy - Ground truth labels from the validation set
- scores.npy - Raw model scores/logits for confidence analysis
- test_images.npy - Sample images for classification examples grid
"""

import random
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch

from core.training.callbacks.base import Callback
from utils.logger import get_logger
from utils.paths import ensure_dir

logger = get_logger(__name__)


class VisualizationDataSaver(Callback):
    """
    Callback that saves data for enhanced visualizations at the end of training.

    This callback collects and saves:
    - Model predictions on the validation set
    - Feature vectors (if available)
    - Ground truth labels
    - Raw model scores/confidence
    - Sample test images for visualization

    Args:
        experiment_dir: Path to the experiment directory
        num_test_images: Number of test images to save for visualization examples
        model_type: Type of model for feature extraction
        save_augmentation_examples: Whether to save augmentation examples
    """

    def __init__(
        self,
        experiment_dir: Union[str, Path],
        num_test_images: int = 20,
        model_type: str = "resnet",
        save_augmentation_examples: bool = True,
    ):
        super().__init__()
        self.experiment_dir = Path(experiment_dir)
        self.num_test_images = num_test_images
        self.model_type = model_type
        self.save_augmentation_examples = save_augmentation_examples

        # Initialize trainer context attributes
        self.model = None
        self.train_loader = None
        self.val_loader = None

        # Create the augmentation examples directory if needed
        if save_augmentation_examples:
            self.aug_examples_dir = self.experiment_dir / "augmentation_examples"
            ensure_dir(self.aug_examples_dir)

        # Run at the end of training
        self.on_train_end_flag = True
        self.priority = 500  # Run after other callbacks

        # Keep track of validation data for feature extraction
        self.val_outputs = None
        self.val_targets = None
        self.val_features = None
        self.val_inputs = None

        logger.info(
            "VisualizationDataSaver initialized with experiment_dir: "
            + str(self.experiment_dir)
        )

    def set_trainer_context(self, context):
        """Directly set all context attributes from the trainer."""
        # Extract context items
        self.model = context.get("model")
        self.train_loader = context.get("train_loader")
        self.val_loader = context.get("val_loader")

        # Log which attributes were set
        logger.info(
            f"VisualizationDataSaver received trainer context: model={self.model is not None}, "
            f"train_loader={self.train_loader is not None}, val_loader={self.val_loader is not None}"
        )

    def _get_feature_extractor(self, model):
        """Get a feature extraction hook based on model type."""
        features = []

        def hook_fn(module, input, output):
            # Store features
            features.append(output.detach().cpu())

        # Log the model architecture for debugging
        logger.debug(f"Model structure: {model}")

        # Different model types may have different feature extraction points
        if self.model_type.lower() == "resnet":
            # For ResNet models, extract from the avgpool layer
            if hasattr(model, "avgpool"):
                logger.info("Using avgpool layer for feature extraction")
                model.avgpool.register_forward_hook(hook_fn)
            elif hasattr(model, "backbone") and hasattr(model.backbone, "avgpool"):
                logger.info("Using backbone.avgpool layer for feature extraction")
                model.backbone.avgpool.register_forward_hook(hook_fn)
            else:
                logger.warning(
                    "ResNet model structure not found - trying to find alternative layers"
                )
                # Fallback to adaptive_avg_pool2d if it exists
                if hasattr(model, "adaptive_avg_pool2d"):
                    logger.info(
                        "Using adaptive_avg_pool2d layer for feature extraction"
                    )
                    model.adaptive_avg_pool2d.register_forward_hook(hook_fn)
                # Try to find the last convolutional layer
                elif hasattr(model, "layer4") and hasattr(
                    model.layer4, "register_forward_hook"
                ):
                    logger.info("Using layer4 for feature extraction")
                    model.layer4.register_forward_hook(hook_fn)
        elif self.model_type.lower() == "efficientnet":
            # For EfficientNet, extract from the classifier[0] layer
            if hasattr(model, "classifier") and len(model.classifier) > 0:
                logger.info("Using classifier[0] layer for feature extraction")
                model.classifier[0].register_forward_hook(hook_fn)
        elif self.model_type.lower() == "vit":
            # For Vision Transformer, extract from the norm layer
            if hasattr(model, "norm"):
                logger.info("Using norm layer for feature extraction")
                model.norm.register_forward_hook(hook_fn)
        elif self.model_type.lower() == "cbam":
            # For CBAM models, try to get the layer after attention
            if hasattr(model, "avgpool"):
                logger.info("Using avgpool layer for feature extraction")
                model.avgpool.register_forward_hook(hook_fn)
            elif hasattr(model, "backbone") and hasattr(model.backbone, "avgpool"):
                logger.info("Using backbone.avgpool layer for feature extraction")
                model.backbone.avgpool.register_forward_hook(hook_fn)

        return features

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        """Store validation outputs on epoch end for later saving."""
        if "val_outputs" in logs and "val_targets" in logs:
            # Store the validation outputs and targets
            self.val_outputs = logs["val_outputs"]
            self.val_targets = logs["val_targets"]
            logger.info(
                f"Epoch {epoch + 1}: Captured validation outputs and targets for visualization."
            )

            # Store input images from this batch if available
            if "inputs" in logs:
                self.val_inputs = logs["inputs"]
                logger.info("Captured input images for visualization.")

    def on_train_end(self, logs: Dict[str, Any]) -> None:
        """Save all visualization data at the end of training."""
        logger.info("Saving data for enhanced visualizations...")

        # Get model and validation data
        model = logs.get("model") or self.model
        train_loader = getattr(self, "train_loader", None)
        val_loader = getattr(self, "val_loader", None)

        # Detailed debugging logs
        model_info = f"Model: {type(model).__name__ if model else None}"
        train_loader_info = (
            f"Train loader: {type(train_loader).__name__ if train_loader else None}"
        )
        val_loader_info = (
            f"Val loader: {type(val_loader).__name__ if val_loader else None}"
        )
        logger.info(
            f"Data available for visualization: {model_info}, {train_loader_info}, {val_loader_info}"
        )

        if model is None:
            logger.error("Model not available, can't extract features")
            return

        if val_loader is None:
            logger.error("Validation loader not available, can't get validation data")
            # Add info about how the callback was initialized
            logger.error(
                "Check if val_loader was provided to the Trainer or set manually on the callback"
            )
            return

        # Save predictions and targets if available from logs
        if self.val_outputs is not None and self.val_targets is not None:
            # Get predictions from outputs
            outputs = self.val_outputs
            targets = self.val_targets

            # Convert to numpy and save
            predictions_path = self.experiment_dir / "predictions.npy"
            true_labels_path = self.experiment_dir / "true_labels.npy"
            scores_path = self.experiment_dir / "scores.npy"

            # Convert to numpy with appropriate format
            if isinstance(outputs, torch.Tensor):
                # Get raw scores for confidence analysis
                scores = outputs.numpy()
                # Get predicted class indices
                predictions = np.argmax(scores, axis=1)
            else:
                logger.warning(f"Unexpected outputs type: {type(outputs)}")
                return

            if isinstance(targets, torch.Tensor):
                # Convert targets to numpy
                true_labels = targets.numpy()
                # One-hot encode if necessary
                if len(true_labels.shape) == 1:
                    # Convert to one-hot for ROC curve calculation
                    num_classes = outputs.shape[1]
                    true_labels_onehot = np.zeros((true_labels.size, num_classes))
                    true_labels_onehot[np.arange(true_labels.size), true_labels] = 1
                    true_labels = true_labels_onehot
            else:
                logger.warning(f"Unexpected targets type: {type(targets)}")
                return

            # Save the files
            np.save(predictions_path, predictions)
            np.save(true_labels_path, true_labels)
            np.save(scores_path, scores)

            logger.info(
                f"Saved predictions, true labels, and scores for {len(predictions)} samples"
            )
        else:
            logger.warning(
                "Validation outputs or targets not available - trying to run a validation pass"
            )
            # Try to run a validation pass manually
            try:
                if model is not None and val_loader is not None:
                    model.eval()
                    all_outputs = []
                    all_targets = []
                    logger.info("Running manual validation pass to collect data...")

                    with torch.no_grad():
                        for batch in val_loader:
                            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                                inputs, targets = batch[0], batch[1]
                            elif (
                                isinstance(batch, dict)
                                and "image" in batch
                                and "label" in batch
                            ):
                                inputs, targets = batch["image"], batch["label"]
                            else:
                                logger.error(f"Unexpected batch format: {type(batch)}")
                                continue

                            # Move to device
                            device = next(model.parameters()).device
                            inputs = inputs.to(device)
                            targets = targets.to(device)

                            # Get predictions
                            outputs = model(inputs)
                            if isinstance(outputs, tuple) and len(outputs) == 2:
                                outputs, _ = outputs

                            # Store
                            all_outputs.append(outputs.cpu())
                            all_targets.append(targets.cpu())

                            # Just get one batch for now
                            break

                    if all_outputs and all_targets:
                        logger.info("Successfully collected manual validation data")
                        # Concat and save
                        outputs = torch.cat(all_outputs, dim=0)
                        targets = torch.cat(all_targets, dim=0)

                        # Save the predictions
                        predictions_path = self.experiment_dir / "predictions.npy"
                        true_labels_path = self.experiment_dir / "true_labels.npy"
                        scores_path = self.experiment_dir / "scores.npy"

                        scores = outputs.numpy()
                        predictions = np.argmax(scores, axis=1)

                        # One-hot encode targets
                        true_labels = targets.numpy()
                        if len(true_labels.shape) == 1:
                            num_classes = outputs.shape[1]
                            true_labels_onehot = np.zeros(
                                (true_labels.size, num_classes)
                            )
                            true_labels_onehot[
                                np.arange(true_labels.size), true_labels
                            ] = 1
                            true_labels = true_labels_onehot

                        # Save files
                        np.save(predictions_path, predictions)
                        np.save(true_labels_path, true_labels)
                        np.save(scores_path, scores)

                        logger.info(
                            f"Saved predictions, true labels, and scores for {len(predictions)} samples from manual pass"
                        )
            except Exception as e:
                logger.error(f"Error during manual validation pass: {e}")

        # Extract features from the model on validation data
        try:
            if val_loader is not None and model is not None:
                # Set up feature extraction
                features_list = self._get_feature_extractor(model)
                logger.info(
                    f"Set up feature extraction hooks for model type: {self.model_type}"
                )

                # Run a forward pass on some validation data
                model.eval()
                with torch.no_grad():
                    # Log that we're starting feature extraction
                    logger.info("Running validation batch to extract features...")

                    # Get a batch of data
                    batch_processed = False
                    for batch in val_loader:
                        try:
                            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                                inputs, _ = batch[0], batch[1]
                            elif isinstance(batch, dict) and "image" in batch:
                                inputs = batch["image"]
                            else:
                                logger.error(f"Unexpected batch format: {type(batch)}")
                                continue

                            # Move to the same device as model
                            device = next(model.parameters()).device
                            inputs = inputs.to(device)
                            logger.info(
                                f"Processing batch with {len(inputs)} inputs on device {device}"
                            )

                            # Forward pass
                            model(inputs)
                            batch_processed = True

                            # Check if we got features
                            if features_list and len(features_list) > 0:
                                logger.info("Successfully extracted features.")
                                break
                            else:
                                logger.warning(
                                    "No features captured by the hook - feature list is empty."
                                )
                        except Exception as e:
                            logger.error(
                                f"Error processing batch for feature extraction: {e}"
                            )
                            break

                    if not batch_processed:
                        logger.error(
                            "No batches were successfully processed for feature extraction"
                        )

                # Process and save features if available
                if features_list and len(features_list) > 0:
                    # Flatten the features
                    features = features_list[0]
                    logger.info(f"Raw feature tensor shape: {features.shape}")

                    # Reshape to 2D: [batch_size, feature_dim]
                    if len(features.shape) > 2:
                        if len(features.shape) == 4:
                            # Handle the case of [batch_size, channels, height, width]
                            features = features.mean(dim=(2, 3))
                            logger.info(
                                f"Averaged features over spatial dimensions to shape: {features.shape}"
                            )
                        else:
                            # Otherwise flatten everything after the first dimension
                            features = features.flatten(1)
                            logger.info(
                                f"Flattened features to shape: {features.shape}"
                            )

                    # Save features
                    features_path = self.experiment_dir / "features.npy"
                    np.save(features_path, features.numpy())
                    logger.info(
                        f"Saved {len(features)} feature vectors with dimension {features.shape[1]}"
                    )
                else:
                    logger.warning("No features extracted from the model")
            else:
                logger.error(
                    "Can't extract features: "
                    + f"val_loader {'available' if val_loader else 'missing'}, "
                    + f"model {'available' if model else 'missing'}"
                )
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            import traceback

            logger.error(traceback.format_exc())

        # Save test images if input data is available
        try:
            if val_loader is not None:
                # Collect test images and their labels
                test_images = []
                test_labels = []

                # Get samples from the validation dataset
                if hasattr(val_loader, "dataset"):
                    dataset = val_loader.dataset
                    indices = random.sample(
                        range(len(dataset)), min(self.num_test_images, len(dataset))
                    )

                    for idx in indices:
                        sample = dataset[idx]

                        # Extract images and labels based on the format
                        if isinstance(sample, (list, tuple)) and len(sample) >= 2:
                            image, label = sample[0], sample[1]
                        elif (
                            isinstance(sample, dict)
                            and "image" in sample
                            and "label" in sample
                        ):
                            image, label = sample["image"], sample["label"]
                        else:
                            logger.warning(f"Unexpected sample format: {type(sample)}")
                            continue

                        # Add to our collections
                        test_images.append(
                            image.numpy() if isinstance(image, torch.Tensor) else image
                        )
                        test_labels.append(
                            label.numpy() if isinstance(label, torch.Tensor) else label
                        )

                    # Convert to arrays
                    test_images = np.array(test_images)
                    test_labels = np.array(test_labels)

                    # Save the test images and labels
                    test_images_path = self.experiment_dir / "test_images.npy"
                    np.save(test_images_path, test_images)
                    logger.info(
                        f"Saved {len(test_images)} test images for visualization"
                    )
                else:
                    logger.warning(
                        "Validation loader does not have a dataset attribute"
                    )
        except Exception as e:
            logger.error(f"Error saving test images: {e}")

        # Save augmentation examples if requested
        if self.save_augmentation_examples and train_loader is not None:
            try:
                # Try to get the transforms from the training dataset
                if hasattr(train_loader, "dataset") and hasattr(
                    train_loader.dataset, "transform"
                ):
                    transform = train_loader.dataset.transform
                    logger.info(
                        f"Found transforms for augmentation examples: {transform}"
                    )

                    # Get a few samples from the dataset
                    dataset = train_loader.dataset
                    indices = random.sample(range(len(dataset)), min(5, len(dataset)))

                    for i, idx in enumerate(indices):
                        sample = dataset[idx]

                        # Extract the original image
                        if isinstance(sample, (list, tuple)) and len(sample) >= 2:
                            image = sample[0]
                        elif isinstance(sample, dict) and "image" in sample:
                            image = sample["image"]
                        else:
                            continue

                        # Convert to numpy if tensor
                        if isinstance(image, torch.Tensor):
                            image = (
                                image.numpy().transpose(1, 2, 0)
                                if image.ndim == 3
                                else image.numpy()
                            )

                        # Save the original image
                        from PIL import Image

                        if isinstance(image, np.ndarray):
                            # Normalize to [0, 255] if in [0, 1]
                            if image.max() <= 1.0:
                                image = (image * 255).astype(np.uint8)
                            Image.fromarray(image).save(
                                self.aug_examples_dir / f"original_{i}.png"
                            )

                        # We could save augmented versions here, but that requires knowing the exact transforms
                        # I'll leave this as a placeholder for now

                    logger.info(
                        f"Saved augmentation examples to {self.aug_examples_dir}"
                    )
                else:
                    logger.warning("Training dataset does not have transform attribute")
            except Exception as e:
                logger.error(f"Error saving augmentation examples: {e}")

        logger.info("Finished saving data for enhanced visualizations")
