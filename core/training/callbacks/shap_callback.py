# Create this file: core/training/callbacks/shap_callback.py

"""
SHAP callback for model interpretability after training.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from core.evaluation.shap_evaluation import evaluate_with_shap
from core.training.callbacks.base import Callback
from utils.logger import get_logger

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
    ):
        """
        Initialize SHAP callback.

        Args:
            num_samples: Number of samples to analyze
            compare_with_gradcam: Whether to compare with GradCAM
            num_background_samples: Number of background samples for SHAP
            output_subdir: Subdirectory name to save results
            dataset_split: Dataset split to use ('train', 'val', or 'test')
        """
        self.num_samples = num_samples
        self.compare_with_gradcam = compare_with_gradcam
        self.num_background_samples = num_background_samples
        self.output_subdir = output_subdir
        self.dataset_split = dataset_split

        logger.info(
            f"Initialized SHAP callback: samples={num_samples}, "
            f"background_samples={num_background_samples}, "
            f"dataset={dataset_split}"
        )

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of training."""
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of each epoch."""
        pass

    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of each batch."""
        pass

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of each batch."""
        pass

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Run SHAP analysis at the end of training.

        Args:
            logs: Dictionary containing training information including model and data
        """
        if logs is None:
            logger.warning("No logs provided to on_train_end, skipping SHAP analysis")
            return

        # Extract necessary components from logs
        model = logs.get("model")
        if model is None:
            logger.error("No model found in logs for SHAP analysis")
            return

        # In the training script, the data_module might not be directly passed to logs
        # We need to extract the relevant part containing dataloaders
        experiment_dir = logs.get("experiment_dir", "")
        if isinstance(experiment_dir, Path):
            experiment_dir = str(experiment_dir)
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

                    # Fallback: If test loader is requested but not available, use validation loader instead
                    if self.dataset_split == "test" and "val_loader" in logs:
                        dataloader = logs["val_loader"]
                        logger.info(
                            "Falling back to validation dataset for SHAP analysis"
                        )
                    elif "train_loader" in logs:
                        # Last resort fallback to training loader
                        dataloader = logs["train_loader"]
                        logger.info(
                            "Falling back to training dataset for SHAP analysis"
                        )

            if dataloader is None:
                logger.error("No dataloader available for SHAP analysis, aborting")
                return

            # Create output directory
            output_dir = os.path.join(experiment_dir, self.output_subdir)
            Path(output_dir).mkdir(parents=True, exist_ok=True)

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

            # Run SHAP evaluation
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
