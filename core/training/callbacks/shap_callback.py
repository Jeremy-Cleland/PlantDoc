# Create this file: core/training/callbacks/shap_callback.py

"""
SHAP callback for model interpretability after training.
"""

import os
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader, TensorDataset

from core.evaluation.shap_evaluation import evaluate_with_shap
from core.training.callbacks.base import Callback
from utils.logger import get_logger
from utils.paths import ensure_dir

logger = get_logger(__name__)


class SHAPCallback(Callback):
    """
    Callback to run SHAP analysis at the end of training.
    """

    def __init__(
        self,
        num_samples: int = 10,
        compare_with_gradcam: bool = True,
        num_background_samples: int = 50,
        output_subdir: str = "shap_analysis",
        dataset_split: str = "test",
        every_n_epochs: int = 10,
        enabled: bool = True,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize SHAP callback.

        Args:
            num_samples: Number of samples to analyze
            compare_with_gradcam: Whether to compare with GradCAM
            num_background_samples: Number of background samples for SHAP
            output_subdir: Subdirectory name to save results
            dataset_split: Dataset split to use ('train', 'val', or 'test')
            every_n_epochs: How often to run SHAP analysis during training
            enabled: Whether the callback is enabled
            output_dir: Directory to save SHAP results (if None, uses experiment_dir/output_subdir)
        """
        self.num_samples = num_samples
        self.compare_with_gradcam = compare_with_gradcam
        self.num_background_samples = num_background_samples
        self.output_subdir = output_subdir
        self.dataset_split = dataset_split
        self.every_n_epochs = every_n_epochs
        self.enabled = enabled
        self.output_dir = output_dir
        self.current_epoch = 0
        self.experiment_dir = None

        logger.info(  # Use the correctly initialized logger
            f"Initialized SHAP callback: samples={self.num_samples}, "
            f"background_samples={self.num_background_samples}, "
            f"dataset={self.dataset_split}"
        )

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of training."""
        if logs and "experiment_dir" in logs:
            self.experiment_dir = logs["experiment_dir"]
            logger.info(
                f"SHAP callback will save results to {self.experiment_dir}/{self.output_subdir}"
            )

    def _get_output_dir(self, logs: Optional[Dict[str, Any]] = None) -> str:
        """Get the output directory for SHAP analysis."""
        if self.output_dir is not None:
            return self.output_dir

        # Use experiment_dir if available
        if self.experiment_dir is not None:
            # Changed to use reports/plots directory structure
            return os.path.join(
                self.experiment_dir, "reports", "plots", self.output_subdir
            )

        # Fallback - use current working directory
        return os.path.join(os.getcwd(), self.output_subdir)

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Run SHAP analysis at the end of training.

        Args:
            logs: Dictionary containing training information including model and data
        """
        if not self.enabled:
            logger.info("SHAP callback is disabled, skipping analysis")
            return

        if logs is None:
            logger.warning("No logs provided to on_train_end, skipping SHAP analysis")
            return

        # Extract necessary components from logs
        model = logs.get("model")
        if model is None:
            logger.error("No model found in logs for SHAP analysis")
            return

        try:
            logger.info("Running SHAP analysis at the end of training...")

            # Ensure model is in eval mode
            model.eval()

            # Try to get the appropriate dataloader
            # Extract dataloaders from logs
            dataloader = None

            # Check if we have direct access to dataloaders in logs
            if self.dataset_split == "train" and "train_loader" in logs:
                dataloader = logs["train_loader"]
                logger.info("Using training dataset for SHAP analysis")
            elif self.dataset_split == "val" and "val_loader" in logs:
                dataloader = logs["val_loader"]
                logger.info("Using validation dataset for SHAP analysis")
            elif self.dataset_split == "test" and "test_loader" in logs:
                dataloader = logs["test_loader"]
                logger.info("Using test dataset for SHAP analysis")
            else:
                # Try to find a data_module object in logs
                data_module = logs.get("data_module")
                if data_module is not None:
                    if self.dataset_split == "train" and hasattr(
                        data_module, "train_dataloader"
                    ):
                        dataloader = data_module.train_dataloader()
                        logger.info(
                            "Using training dataset from data_module for SHAP analysis"
                        )
                    elif self.dataset_split == "val" and hasattr(
                        data_module, "val_dataloader"
                    ):
                        dataloader = data_module.val_dataloader()
                        logger.info(
                            "Using validation dataset from data_module for SHAP analysis"
                        )
                    elif hasattr(data_module, "test_dataloader"):
                        dataloader = data_module.test_dataloader()
                        logger.info(
                            "Using test dataset from data_module for SHAP analysis"
                        )
                    else:
                        logger.error(
                            f"Could not find {self.dataset_split} dataloader in data_module"
                        )
                else:
                    logger.warning(
                        f"No dataloader found for {self.dataset_split} split and no data_module available"
                    )

                    # Fallback options in order of preference
                    if "val_loader" in logs:
                        dataloader = logs["val_loader"]
                        logger.info(
                            "Falling back to validation dataset for SHAP analysis"
                        )
                    elif "train_loader" in logs:
                        dataloader = logs["train_loader"]
                        logger.info(
                            "Falling back to training dataset for SHAP analysis"
                        )

            if dataloader is None:
                # Try to create a small dataset from a sample batch
                try:
                    # Look for any input tensor data in logs to create a small dataset
                    if "sample_batch" in logs:
                        batch = logs["sample_batch"]
                        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                            x, y = batch[0], batch[1]

                            # Create a small dataset if we have tensors
                            if isinstance(x, torch.Tensor) and isinstance(
                                y, torch.Tensor
                            ):
                                dataset = TensorDataset(x, y)
                                dataloader = DataLoader(
                                    dataset, batch_size=16, shuffle=False
                                )
                                logger.info(
                                    "Created dataloader from sample batch for SHAP analysis"
                                )
                except Exception as e:
                    logger.warning(f"Could not create dataloader from sample: {e}")

            if dataloader is None:
                logger.error("No dataloader available for SHAP analysis, aborting")
                return

            # Create output directory
            output_dir = self._get_output_dir(logs)
            ensure_dir(output_dir)
            logger.info(f"SHAP results will be saved to: {output_dir}")

            # Try to get class names from various sources
            class_names = None

            # Check logs for class_names directly
            if "class_names" in logs:
                class_names = logs["class_names"]
                logger.info(f"Using {len(class_names)} class names from logs")
            # Check if we have a data_module with class_names
            elif "data_module" in logs and hasattr(logs["data_module"], "class_names"):
                class_names = logs["data_module"].class_names
                logger.info(f"Using {len(class_names)} class names from data_module")
            # Try to get from the dataloader's dataset
            elif hasattr(dataloader, "dataset"):
                dataset = dataloader.dataset
                if hasattr(dataset, "class_names"):
                    class_names = dataset.class_names
                    logger.info(f"Using {len(class_names)} class names from dataset")
                elif hasattr(dataset, "classes"):
                    class_names = dataset.classes
                    logger.info(
                        f"Using {len(class_names)} class names from dataset.classes"
                    )

            if class_names is None:
                logger.warning(
                    "No class names found for SHAP analysis, visualization may be less interpretable"
                )

            # Ensure input format is correct
            logger.info("Checking input tensor format before SHAP analysis...")
            try:
                dataloader_iter = iter(dataloader)
                sample_batch = next(dataloader_iter)

                # Verify data format and shapes
                if isinstance(sample_batch, (tuple, list)) and len(sample_batch) >= 2:
                    x_sample, y_sample = sample_batch[0], sample_batch[1]
                    logger.info(
                        f"Input tensor shape: {x_sample.shape}, dtype: {x_sample.dtype}"
                    )
                    logger.info(
                        f"Target tensor shape: {y_sample.shape}, dtype: {y_sample.dtype}"
                    )
                elif isinstance(sample_batch, dict):
                    x_keys = [
                        k
                        for k in sample_batch.keys()
                        if k in ("input", "x", "image", "images")
                    ]
                    y_keys = [
                        k
                        for k in sample_batch.keys()
                        if k in ("target", "y", "label", "labels")
                    ]

                    if x_keys and y_keys:
                        x_sample = sample_batch[x_keys[0]]
                        y_sample = sample_batch[y_keys[0]]
                        logger.info(
                            f"Input tensor shape: {x_sample.shape}, dtype: {x_sample.dtype}"
                        )
                        logger.info(
                            f"Target tensor shape: {y_sample.shape}, dtype: {y_sample.dtype}"
                        )

            except Exception as e:
                logger.warning(f"Could not check input format: {e}")

            # Run SHAP evaluation with proper error handling
            try:
                results = evaluate_with_shap(
                    model=model,
                    data_loader=dataloader,
                    num_samples=self.num_samples,
                    output_dir=output_dir,
                    class_names=class_names,
                    num_background_samples=self.num_background_samples,
                    compare_with_gradcam=self.compare_with_gradcam,
                )
                logger.info(f"SHAP analysis complete. Results saved to {output_dir}")
            except Exception as e:
                logger.error(f"Error in SHAP callback: {e}", exc_info=True)
                # Create a minimal results object to prevent None returns
                results = {
                    "error": str(e),
                    "num_explained_samples": 0,
                    "visualization_dir": str(output_dir),
                    "status": "failed"
                }

            # Save a summary of results
            with open(os.path.join(output_dir, "summary.txt"), "w") as f:
                f.write("SHAP Analysis Summary\n")
                f.write("====================\n\n")
                f.write(
                    f"Number of samples analyzed: {results.get('num_explained_samples', 0)}\n"
                )
                f.write(
                    f"Comparisons with GradCAM: {len(results.get('comparisons', []))}\n"
                )
                f.write(f"See visualizations in: {output_dir}\n")

            return results

        except Exception as e:
            logger.error(f"Error in SHAP callback: {e}", exc_info=True)
            return None

    def on_validation_end(self, model, val_loader=None, epoch=None, logs=None):
        """
        Generate SHAP visualizations after validation.

        Args:
            model: The PyTorch model to analyze
            val_loader: Optional validation dataloader
            epoch: Current epoch number (if None, uses self.current_epoch)
            logs: Dictionary with additional information
        """
        if not self.enabled:
            return

        # Use provided epoch or self.current_epoch
        current_epoch = epoch if epoch is not None else self.current_epoch

        # Check if we should run SHAP analysis
        if current_epoch % self.every_n_epochs != 0:
            return

        logger.info(f"Running SHAP analysis at epoch {current_epoch}")

        # Get output directory
        output_dir = self._get_output_dir(logs)
        ensure_dir(output_dir)
        logger.info(f"SHAP results will be saved to: {output_dir}")

        # Try to use provided val_loader
        dataloader = val_loader

        # If no dataloader, try to find one in logs
        if dataloader is None and logs is not None:
            if "val_loader" in logs:
                dataloader = logs["val_loader"]
                logger.info("Using validation dataloader from logs")
            elif "test_loader" in logs:
                dataloader = logs["test_loader"]
                logger.info("Using test dataloader from logs")
            elif "train_loader" in logs:
                dataloader = logs["train_loader"]
                logger.info("Using train dataloader from logs")

        # If still no dataloader, try to create a small dataset from the training data
        if dataloader is None:
            try:
                train_loader = logs.get("train_loader") if logs else None
                if train_loader is not None:
                    # Take a small subset of the training data
                    x_list = []
                    y_list = []
                    device = next(model.parameters()).device

                    # Collect a few batches
                    for i, batch in enumerate(train_loader):
                        if isinstance(batch, (list, tuple)):
                            x, y = batch[0], batch[1]
                        else:
                            # Assume dictionary with 'input' and 'target' keys
                            x, y = (
                                batch.get(
                                    "input", batch.get("x", batch.get("image", None))
                                ),
                                batch.get(
                                    "target", batch.get("y", batch.get("label", None))
                                ),
                            )

                        if x is None or y is None:
                            logger.warning(
                                "Could not extract inputs and targets from batch"
                            )
                            continue

                        x_list.append(x)
                        y_list.append(y)

                        if i >= 2:  # Just use 2-3 batches
                            break

                    if x_list and y_list:
                        # Create a simple dataloader
                        x_tensor = torch.cat(x_list, dim=0)
                        y_tensor = torch.cat(y_list, dim=0)

                        # Limit to a reasonable number of samples
                        max_samples = min(50, len(x_tensor))
                        indices = torch.randperm(len(x_tensor))[:max_samples]
                        x_tensor, y_tensor = x_tensor[indices], y_tensor[indices]

                        dataset = TensorDataset(x_tensor, y_tensor)
                        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
                        logger.info(
                            f"Created temporary dataloader with {len(dataset)} samples for SHAP analysis"
                        )
            except Exception as e:
                logger.error(f"Failed to create temporary dataloader: {e}")

        if dataloader is None:
            logger.error("No dataloader available for SHAP analysis, aborting")
            return

        # Try to get class names from dataloader
        class_names = None
        if hasattr(dataloader, "dataset"):
            dataset = dataloader.dataset
            if hasattr(dataset, "class_names"):
                class_names = dataset.class_names
            elif hasattr(dataset, "classes"):
                class_names = dataset.classes

        if class_names is None and logs is not None:
            class_names = logs.get("class_names")

        # Set model to eval mode
        model.eval()

        try:
            # Ensure input format is correct
            logger.info("Checking input tensor format before SHAP analysis...")
            try:
                dataloader_iter = iter(dataloader)
                sample_batch = next(dataloader_iter)

                # Verify data format and shapes
                if isinstance(sample_batch, (tuple, list)) and len(sample_batch) >= 2:
                    x_sample, y_sample = sample_batch[0], sample_batch[1]
                    logger.info(
                        f"Input tensor shape: {x_sample.shape}, dtype: {x_sample.dtype}"
                    )
                    logger.info(
                        f"Target tensor shape: {y_sample.shape}, dtype: {y_sample.dtype}"
                    )
                elif isinstance(sample_batch, dict):
                    x_keys = [
                        k
                        for k in sample_batch.keys()
                        if k in ("input", "x", "image", "images")
                    ]
                    y_keys = [
                        k
                        for k in sample_batch.keys()
                        if k in ("target", "y", "label", "labels")
                    ]

                    if x_keys and y_keys:
                        x_sample = sample_batch[x_keys[0]]
                        y_sample = sample_batch[y_keys[0]]
                        logger.info(
                            f"Input tensor shape: {x_sample.shape}, dtype: {x_sample.dtype}"
                        )
                        logger.info(
                            f"Target tensor shape: {y_sample.shape}, dtype: {y_sample.dtype}"
                        )

            except Exception as e:
                logger.warning(f"Could not check input format: {e}")

            # Run SHAP evaluation with proper error handling
            try:
                results = evaluate_with_shap(
                    model=model,
                    data_loader=dataloader,
                    num_samples=self.num_samples,
                    output_dir=output_dir,
                    class_names=class_names,
                    num_background_samples=self.num_background_samples,
                    compare_with_gradcam=self.compare_with_gradcam,
                )
                logger.info(f"SHAP analysis complete. Results saved to {output_dir}")
            except Exception as e:
                logger.error(f"Error in SHAP callback: {e}", exc_info=True)
                # Create a minimal results object to prevent None returns
                results = {
                    "error": str(e),
                    "num_explained_samples": 0,
                    "visualization_dir": str(output_dir),
                    "status": "failed"
                }
            return results
        except Exception as e:
            logger.error(f"Error in SHAP validation analysis: {e}", exc_info=True)
            return None
