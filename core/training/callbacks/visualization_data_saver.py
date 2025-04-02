"""
Callback to save data needed for enhanced visualizations in reports.

This callback saves the necessary files for our enhanced visualization functions:
- predictions.npy - Model predictions on the validation set
- features.npy - Feature vectors from the model
- true_labels.npy - Ground truth labels from the validation set
- scores.npy - Raw model scores/logits for confidence analysis
- test_images.npy - Sample images for classification examples grid
"""

import os
import random
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

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
            self.aug_examples_dir = (
                self.experiment_dir / "reports" / "plots" / "augmentation"
            )
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
        elif "cbam" in self.model_type.lower():
            # For CBAM models, extract from the last layer of the backbone and after global pooling
            if hasattr(model, "backbone") and hasattr(model.backbone, "layer4"):
                logger.info("Using backbone.layer4[-1] for CBAM feature extraction")
                # Get the last residual block in layer4
                last_layer = model.backbone.layer4[-1]
                last_layer.register_forward_hook(hook_fn)
            elif hasattr(model, "backbone") and hasattr(model.backbone, "backbone"):
                # For nested backbone structures
                if hasattr(model.backbone.backbone, "layer4"):
                    logger.info(
                        "Using backbone.backbone.layer4[-1] for CBAM feature extraction"
                    )
                    model.backbone.backbone.layer4[-1].register_forward_hook(hook_fn)

            # Also try to hook the global pooling if available
            if hasattr(model, "global_pool"):
                logger.info("Using global_pool for feature extraction")
                model.global_pool.register_forward_hook(hook_fn)
            elif hasattr(model, "backbone") and hasattr(model.backbone, "global_pool"):
                logger.info("Using backbone.global_pool for feature extraction")
                model.backbone.global_pool.register_forward_hook(hook_fn)

            # For models with a feature_fusion module
            if hasattr(model, "feature_fusion") and not isinstance(
                model.feature_fusion, bool
            ):
                logger.info("Using feature_fusion for feature extraction")
                model.feature_fusion.register_forward_hook(hook_fn)
            elif (
                hasattr(model, "backbone")
                and hasattr(model.backbone, "feature_fusion")
                and not isinstance(model.backbone.feature_fusion, bool)
            ):
                logger.info("Using backbone.feature_fusion for feature extraction")
                model.backbone.feature_fusion.register_forward_hook(hook_fn)
        elif hasattr(model, "features"):
            # Generic approach for models with a features module (DenseNet, EfficientNet, etc.)
            logger.info("Using generic 'features' module for feature extraction")
            model.features.register_forward_hook(hook_fn)
        else:
            # Fallback for other model architectures
            logger.warning(
                f"No suitable feature extraction point found for {self.model_type}"
            )
            for name, module in model.named_modules():
                # Try to find a convolutional layer near the end (before classification head)
                if isinstance(module, nn.Conv2d) and "layer" in name and "4" in name:
                    logger.info(f"Using {name} for feature extraction")
                    module.register_forward_hook(hook_fn)
                    break

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
        """
        Save visualization data at the end of training.

        Args:
            logs: Training logs
        """
        # Log available information
        logger.info(
            "VisualizationDataSaver: Saving visualization data at the end of training"
        )
        logger.info(f"Available keys: {logs.keys() if logs else 'No logs available'}")

        # Get model and loaders from context
        model = getattr(self, "model", None)
        val_loader = getattr(self, "val_loader", None)

        if model is None or val_loader is None:
            logger.warning("Model or validation dataloader not available")
            return

        # Create evaluation_artifacts directory directly in experiment_dir
        artifacts_dir = self.experiment_dir / "evaluation_artifacts"
        ensure_dir(artifacts_dir)

        # For backward compatibility, also ensure metrics/evaluation_artifacts exists
        metrics_artifacts_dir = self.experiment_dir / "metrics" / "evaluation_artifacts"
        ensure_dir(metrics_artifacts_dir)

        # Helper function to save artifacts with backward compatibility
        def save_artifact(name, data):
            # Primary location (correct one)
            primary_path = artifacts_dir / f"{name}.npy"
            np.save(primary_path, data)

            # Legacy location (for backward compatibility)
            legacy_path = metrics_artifacts_dir / f"{name}.npy"
            np.save(legacy_path, data)

            logger.info(f"Saved {name} to {primary_path} and legacy location")
            return primary_path

        # Save predictions and targets if available from logs
        if self.val_outputs is not None and self.val_targets is not None:
            # Get predictions from outputs
            outputs = self.val_outputs
            targets = self.val_targets

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
            save_artifact("predictions", predictions)
            save_artifact("true_labels", true_labels)
            save_artifact("scores", scores)

            # Also save a subset of true labels that matches test_images size
            # This prevents the size mismatch warning in generate_plots.py
            subset_size = min(self.num_test_images, len(true_labels))
            if len(true_labels) > subset_size:
                save_artifact("subset_true_labels", true_labels[:subset_size])
                logger.info(
                    f"Saved subset of true labels ({subset_size}) to match test_images size"
                )

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

                        # Save using the helper function
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

                        # Save files using the helper function
                        save_artifact("predictions", predictions)
                        save_artifact("true_labels", true_labels)
                        save_artifact("scores", scores)

                        # Also save a subset of true labels that matches test_images size
                        subset_size = min(self.num_test_images, len(true_labels))
                        if len(true_labels) > subset_size:
                            save_artifact(
                                "subset_true_labels", true_labels[:subset_size]
                            )
                            logger.info(
                                f"Saved subset of true labels ({subset_size}) to match test_images size"
                            )

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
                    logger.info("Running validation batches to extract features...")

                    # Get multiple batches of data to match the number of labels
                    batch_processed = False
                    batch_count = 0
                    max_batches = 5  # Process up to 5 batches to get more features

                    # Calculate how many samples we need to match true_labels
                    true_labels_path = (
                        self.experiment_dir / "evaluation_artifacts" / "true_labels.npy"
                    )
                    required_samples = 0
                    if true_labels_path.exists():
                        try:
                            true_labels = np.load(true_labels_path)
                            required_samples = len(true_labels)
                            logger.info(
                                f"Need to extract features for {required_samples} samples to match true_labels"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Couldn't load true_labels for size matching: {e}"
                            )

                    collected_samples = 0

                    for batch in val_loader:
                        if batch_count >= max_batches:
                            break

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
                            batch_size = len(inputs)
                            collected_samples += batch_size
                            logger.info(
                                f"Processing batch {batch_count + 1} with {batch_size} inputs on device {device}"
                            )

                            # Forward pass
                            model(inputs)
                            batch_processed = True
                            batch_count += 1

                            # Check if we got features
                            if features_list and len(features_list) > 0:
                                logger.info(
                                    f"Successfully extracted features from batch {batch_count}"
                                )

                                # If we have enough samples to match true_labels
                                if (
                                    required_samples > 0
                                    and collected_samples >= required_samples
                                ):
                                    logger.info(
                                        f"Collected enough samples ({collected_samples}) to match true_labels ({required_samples})"
                                    )
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
                    else:
                        logger.info(
                            f"Processed {batch_count} batches with approx. {collected_samples} samples for feature extraction"
                        )

                # Process and save features if available
                if features_list and len(features_list) > 0:
                    # Flatten the features
                    features = features_list[0]
                    if features.shape[0] > 0:
                        # Save features to numpy file
                        save_artifact("features", features.cpu().numpy())
                        logger.info(f"Features shape: {features.shape}")
                    else:
                        logger.warning("No features extracted from the model")
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

                    # Save the test images and labels using the helper function
                    save_artifact("test_images", test_images)
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
        if (
            hasattr(self, "save_augmentation_examples")
            and isinstance(self.save_augmentation_examples, bool)
            and self.save_augmentation_examples
            and self.train_loader is not None
        ):
            try:
                self._save_augmentation_examples()
            except Exception as e:
                logger.error(f"Error saving augmentation examples: {e}")
                import traceback

                logger.error(traceback.format_exc())

        logger.info("Finished saving data for enhanced visualizations")

    def _save_augmentation_examples(self):
        """Save augmentation examples for visualization."""
        if not hasattr(self, "train_loader") or self.train_loader is None:
            logger.warning("No train_loader available for augmentation examples")
            return

        # Get transforms if available
        transforms = None
        if hasattr(self.train_loader.dataset, "transform"):
            transforms = self.train_loader.dataset.transform
            logger.info(f"Found transforms for augmentation examples: {transforms}")
        else:
            logger.warning("No transforms found in dataset for augmentation examples")
            return

        # Save some examples of augmentations
        try:
            # Get a few random samples
            samples = []
            for i, batch in enumerate(self.train_loader):
                if i >= 1:  # Just process one batch
                    break

                # Extract inputs from batch, handling different formats
                if isinstance(batch, (list, tuple)):
                    # Handle tuple/list format
                    inputs = batch[0]  # First element is typically inputs
                elif isinstance(batch, dict) and "image" in batch:
                    # Handle dictionary format
                    inputs = batch["image"]
                else:
                    logger.error(f"Unexpected batch format: {type(batch)}")
                    continue

                # Get a subset of the batch for visualization
                inputs = inputs[: min(5, len(inputs))]

                # Convert inputs to appropriate format to avoid data type issues
                inputs_list = []
                for img_tensor in inputs:
                    # Convert to uint8 numpy array for consistent format
                    if img_tensor.dim() == 3:  # CHW format
                        # Convert CHW to HWC
                        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()

                        # Handle different normalization scenarios
                        if img_np.max() <= 1.0:
                            img_np = (img_np * 255).astype(np.uint8)
                        else:
                            img_np = img_np.astype(np.uint8)

                        # Ensure 3 channels (RGB)
                        if img_np.shape[2] == 1:
                            img_np = np.repeat(img_np, 3, axis=2)

                        inputs_list.append(img_np)
                    else:
                        logger.warning(f"Unexpected tensor shape: {img_tensor.shape}")

                samples.extend(inputs_list)

            # Save augmentation examples in the experiment directory
            output_dir = self.experiment_dir / "reports" / "plots" / "augmentation"
            os.makedirs(output_dir, exist_ok=True)

            # Process each sample
            for i, img_array in enumerate(samples):
                if i >= 5:  # Limit to 5 samples
                    break

                # Convert to PIL Image for augmentation
                img = Image.fromarray(img_array)

                # Save original image with augmentations
                output_path = output_dir / f"augmentation_example_{i + 1}.png"

                # Import here to avoid circular imports
                from scripts.generate_augmentations import create_augmentation_grid

                # Save the image first since create_augmentation_grid expects a path
                img_path = output_dir / f"original_{i + 1}.png"
                img.save(img_path)

                # Now create the augmentation grid
                create_augmentation_grid(str(img_path), output_path, theme="plantdoc")

                # Copy the image to the augmentation directory

                logger.info(f"Saved augmentation example to {output_path}")

        except Exception as e:
            logger.error(f"Error saving augmentation examples: {e}")
            import traceback

            logger.error(traceback.format_exc())
