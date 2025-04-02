"""
Metrics logger callback for saving training metrics to disk.
"""

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf

from utils.logger import get_logger
from utils.paths import ensure_dir

from .base import Callback

logger = get_logger(__name__)


def _convert_omegaconf_to_python(obj):
    """
    Convert OmegaConf objects (DictConfig, ListConfig) to native Python types for JSON serialization.

    Args:
        obj: Object to convert

    Returns:
        Object with OmegaConf types converted to native Python types
    """
    if isinstance(obj, DictConfig):
        return {k: _convert_omegaconf_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, ListConfig) or isinstance(obj, (list, tuple)):
        return [_convert_omegaconf_to_python(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: _convert_omegaconf_to_python(v) for k, v in obj.items()}
    else:
        return obj


class MetricsLogger(Callback):
    """
    Callback to save training and validation metrics to disk.

    Args:
        metrics_dir: Directory to save metrics files
        filename: Base filename (extensions added based on format)
        save_format: Format to save metrics in ('json', 'jsonl', 'csv')
        overwrite: Whether to overwrite existing files
        experiment_dir: Directory to save additional report files
        class_names: List of class names for class-specific metrics
    """

    priority = 70  # Run after most other callbacks

    def __init__(
        self,
        metrics_dir: Union[str, Path],
        filename: str = "training_metrics",
        save_format: str = "json",
        overwrite: bool = True,
        experiment_dir: Optional[Union[str, Path]] = None,
        class_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.metrics_dir = Path(metrics_dir)
        self.filename = filename
        self.save_format = save_format.lower()
        self.overwrite = overwrite
        self.experiment_dir = Path(experiment_dir) if experiment_dir else None
        self.class_names = class_names
        self.config = None  # Initialize config attribute to None

        # Validate format
        if self.save_format not in ["json", "jsonl", "csv"]:
            raise ValueError(
                f"Format must be 'json', 'jsonl', or 'csv', got {self.save_format}"
            )

        # Create metrics directory
        os.makedirs(self.metrics_dir, exist_ok=True)

        # If experiment_dir is provided, ensure it exists
        if self.experiment_dir:
            os.makedirs(self.experiment_dir, exist_ok=True)

        # Initialize metrics history
        self.history = []
        self.has_saved = False

        logger.info(
            f"Initialized MetricsLogger: "
            f"format='{self.save_format}', "
            f"overwrite={self.overwrite}"
        )

    def set_trainer_context(self, context: Dict[str, Any]) -> None:
        """
        Set context from the trainer. Provides access to model, optimizer, etc.

        Args:
            context: Dictionary containing training context information
        """
        if "config" in context:
            self.config = context["config"]
            logger.info("MetricsLogger received config from trainer context")
        if "experiment_dir" in context and context["experiment_dir"] is not None:
            self.experiment_dir = Path(context["experiment_dir"])
            logger.info(f"MetricsLogger received experiment_dir: {self.experiment_dir}")

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Extract training parameters from config and save to a file.
        This runs once at the beginning of training.
        """
        # Check if we have access to the config
        if not hasattr(self, "config"):
            logger.warning(
                "No config attribute found in MetricsLogger. Cannot extract training parameters."
            )
            self._create_default_training_params()
            return

        if self.config is None:
            logger.warning(
                "Config attribute exists but is None in MetricsLogger. Cannot extract training parameters."
            )
            self._create_default_training_params()
            return

        try:
            # Extract config - use OmegaConf.to_container if needed
            cfg = self.config

            # Extract relevant training parameters
            training_params = {}

            # Model parameters
            if "model" in cfg:
                training_params["model"] = {
                    "name": cfg.model.get("name", "unknown"),
                    "num_classes": cfg.model.get("num_classes", 0),
                    "pretrained": cfg.model.get("pretrained", False),
                    "input_size": cfg.model.get("input_size", [224, 224]),
                }

                # Optional model parameters
                for param in ["head_type", "hidden_dim", "dropout_rate", "backbone"]:
                    if param in cfg.model:
                        # Use OmegaConf.to_container for nested configs
                        value = cfg.model[param]
                        if isinstance(value, (DictConfig, ListConfig)):
                            value = OmegaConf.to_container(value)
                        training_params["model"][param] = value

            # Training parameters
            if "training" in cfg:
                training_params["training"] = {
                    "epochs": cfg.training.get("epochs", 0),
                    "batch_size": cfg.training.get("batch_size", 0),
                }

                # Optional training parameters
                for param in [
                    "learning_rate",
                    "weight_decay",
                    "use_mixed_precision",
                    "gradient_clip_val",
                    "precision",
                ]:
                    if param in cfg.training:
                        # Use OmegaConf.to_container for nested configs
                        value = cfg.training[param]
                        if isinstance(value, (DictConfig, ListConfig)):
                            value = OmegaConf.to_container(value)
                        training_params["training"][param] = value

            # Optimizer parameters
            if "optimizer" in cfg:
                training_params["optimizer"] = {
                    "name": cfg.optimizer.get("name", "adam"),
                }

                # Optional optimizer parameters
                for param in ["lr", "weight_decay", "momentum", "beta1", "beta2"]:
                    if param in cfg.optimizer:
                        # Use OmegaConf.to_container for nested configs
                        value = cfg.optimizer[param]
                        if isinstance(value, (DictConfig, ListConfig)):
                            value = OmegaConf.to_container(value)
                        training_params["optimizer"][param] = value

            # Scheduler parameters
            if "scheduler" in cfg:
                training_params["scheduler"] = {
                    "name": cfg.scheduler.get("name", "none"),
                }

                # Optional scheduler parameters
                for param in [
                    "monitor",
                    "factor",
                    "patience",
                    "min_lr",
                    "step_size",
                    "gamma",
                ]:
                    if param in cfg.scheduler:
                        # Use OmegaConf.to_container for nested configs
                        value = cfg.scheduler[param]
                        if isinstance(value, (DictConfig, ListConfig)):
                            value = OmegaConf.to_container(value)
                        training_params["scheduler"][param] = value

            # Loss parameters
            if "loss" in cfg:
                training_params["loss"] = {
                    "name": cfg.loss.get("name", "cross_entropy"),
                }

                # Optional loss parameters
                for param in ["components", "weights"]:
                    if param in cfg.loss:
                        # Use OmegaConf.to_container for nested configs
                        value = cfg.loss[param]
                        if isinstance(value, (DictConfig, ListConfig)):
                            value = OmegaConf.to_container(value)
                        training_params["loss"][param] = value

            # Data parameters
            if "data" in cfg:
                training_params["data"] = {}

                if "train_val_test_split" in cfg.data:
                    # Use OmegaConf.to_container for nested configs
                    value = cfg.data.train_val_test_split
                    if isinstance(value, (DictConfig, ListConfig)):
                        value = OmegaConf.to_container(value)
                    training_params["data"]["train_val_test_split"] = value

            # Preprocessing parameters
            if "preprocessing" in cfg:
                training_params["data"]["preprocessing"] = {}
                for param in ["resize", "center_crop", "normalize"]:
                    if param in cfg.preprocessing:
                        # Use OmegaConf.to_container for nested configs
                        value = cfg.preprocessing[param]
                        if isinstance(value, (DictConfig, ListConfig)):
                            value = OmegaConf.to_container(value)
                        training_params["data"]["preprocessing"][param] = value

            # Augmentation parameters
            if "augmentation" in cfg and "train" in cfg.augmentation:
                training_params["data"]["augmentation"] = {}
                for param in ["rand_augment", "cutmix", "mixup"]:
                    if param in cfg.augmentation.train:
                        # Use OmegaConf.to_container for nested configs
                        value = cfg.augmentation.train[param]
                        if isinstance(value, (DictConfig, ListConfig)):
                            value = OmegaConf.to_container(value)
                        training_params["data"]["augmentation"][param] = value

            # Save to file
            self._save_training_params(training_params)

        except Exception as e:
            logger.error(f"Error extracting and saving training parameters: {e}")
            import traceback

            logger.error(traceback.format_exc())
            self._create_default_training_params()

    def _create_default_training_params(self):
        """Create a default training_params.json file with basic information."""
        logger.info("Creating default training parameters file")

        # Try to extract some information from logs if available
        model_name = "unknown"
        num_classes = 0

        # Try to extract model name and classes from self.model
        if hasattr(self, "model"):
            model = self.model
            if hasattr(model, "__class__"):
                model_name = model.__class__.__name__
            if hasattr(model, "num_classes"):
                num_classes = model.num_classes

        # Create basic training parameters
        training_params = {
            "model": {
                "name": model_name,
                "num_classes": num_classes,
                "pretrained": True,
                "input_size": [224, 224],
            },
            "training": {
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 0.001,
                "weight_decay": 1e-5,
                "use_mixed_precision": True,
            },
            "optimizer": {
                "name": "adam",
                "lr": 0.001,
                "weight_decay": 1e-5,
            },
            "scheduler": {
                "name": "reduce_on_plateau",
                "monitor": "val_loss",
                "factor": 0.1,
                "patience": 3,
                "min_lr": 1e-6,
            },
            "loss": {
                "name": "cross_entropy",
            },
        }

        # Save the default training parameters
        self._save_training_params(training_params)

    def _save_training_params(self, training_params):
        """Save training parameters to file."""
        # Convert OmegaConf objects to Python types
        training_params = _convert_omegaconf_to_python(training_params)

        # Save to file
        output_path = self.metrics_dir / "training_params.json"
        # Also save to experiment_dir if it exists
        exp_output_path = (
            self.experiment_dir / "metrics" / "training_params.json"
            if self.experiment_dir
            else None
        )

        try:
            # Save the file
            with open(output_path, "w") as f:
                json.dump(training_params, f, indent=4)
            logger.info(f"Saved training parameters to {output_path}")

            # Also save to experiment_dir if it exists
            if exp_output_path:
                os.makedirs(exp_output_path.parent, exist_ok=True)
                with open(exp_output_path, "w") as f:
                    json.dump(training_params, f, indent=4)
                logger.info(f"Saved training parameters to {exp_output_path}")
        except Exception as e:
            logger.error(f"Error saving training parameters: {e}")

    def _get_filepath(self) -> Path:
        """Get the full path to the metrics file with appropriate extension."""
        if self.save_format == "json":
            return self.metrics_dir / f"{self.filename}.json"
        elif self.save_format == "jsonl":
            return self.metrics_dir / f"{self.filename}.jsonl"
        else:  # csv
            return self.metrics_dir / f"{self.filename}.csv"

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Save metrics at the end of each epoch."""
        logs = logs or {}

        # Filter logs to only include scalar values (numbers, booleans, strings)
        epoch_metrics = {
            "epoch": epoch + 1
        }  # 1-based epoch for user-friendly numbering
        for key, value in logs.items():
            if isinstance(value, (int, float, bool, str)):
                epoch_metrics[key] = value

        # Add to history
        self.history.append(epoch_metrics)

        # Save metrics
        self._save_metrics()

    def _save_metrics(self) -> None:
        """Save metrics to disk in the specified format."""
        filepath = self._get_filepath()

        # Check if file exists and we're not supposed to overwrite
        if filepath.exists() and not self.overwrite and self.has_saved:
            return

        try:
            if self.save_format == "json":
                self._save_json(filepath)
            elif self.save_format == "jsonl":
                self._save_jsonl(filepath)
            else:  # csv
                self._save_csv(filepath)

            self.has_saved = True
        except Exception as e:
            logger.error(f"Error saving metrics to {filepath}: {e}")

    def _save_json(self, filepath: Path) -> None:
        """Save metrics in JSON format."""
        # Convert history to JSON-serializable format
        history = _convert_omegaconf_to_python(self.history)
        with open(filepath, "w") as f:
            json.dump({"history": history}, f, indent=2)

    def _save_jsonl(self, filepath: Path) -> None:
        """Save metrics in JSONL format (one JSON object per line)."""
        mode = "w" if self.overwrite or not self.has_saved else "a"
        with open(filepath, mode) as f:
            for metrics in self.history:
                # Convert each metrics dict to JSON-serializable format
                serializable_metrics = _convert_omegaconf_to_python(metrics)
                f.write(json.dumps(serializable_metrics) + "\n")

    def _save_csv(self, filepath: Path) -> None:
        """Save metrics in CSV format."""
        if not self.history:
            return

        # Get all field names from all epochs
        fieldnames = set()
        for metrics in self.history:
            fieldnames.update(metrics.keys())
        fieldnames = sorted(list(fieldnames))

        mode = "w" if self.overwrite or not self.has_saved else "a"
        with open(filepath, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            # Write header only once
            if mode == "w":
                writer.writeheader()

            # Write rows
            for metrics in self.history:
                writer.writerow(metrics)

    def _convert_history_format(self):
        """Convert history from list of dicts to dict of lists for compatibility with report generator."""
        converted_history = {}
        for entry in self.history:
            for key, value in entry.items():
                if key not in converted_history:
                    converted_history[key] = []
                converted_history[key].append(value)
        return converted_history

    def _save_class_names(self):
        """Save class names to a file if provided."""
        if not self.class_names or not self.experiment_dir:
            return

        try:
            class_names_path = self.experiment_dir / "class_names.txt"
            with open(class_names_path, "w") as f:
                for class_name in self.class_names:
                    f.write(f"{class_name}\n")
            logger.info(f"Saved class names to {class_names_path}")
        except Exception as e:
            logger.error(f"Error saving class names: {e}")

    def _save_report_files(self, final_metrics: Dict[str, Any]):
        """Save files needed for report generation."""
        try:
            if self.experiment_dir is None:
                logger.warning("No experiment_dir provided, skipping report files")
                return

            # Create metrics directory if needed
            metrics_dir = self.experiment_dir / "metrics"
            ensure_dir(metrics_dir)

            # Process class metrics if present
            class_metrics = {}
            if (
                "class_metrics" in final_metrics
                and final_metrics["class_metrics"] is not None
            ):
                logger.info(
                    "Found class metrics in final metrics, processing them for report"
                )
                class_metrics_data = _convert_omegaconf_to_python(
                    final_metrics["class_metrics"]
                )

                # Check if we have dict or list structure for class metrics
                if isinstance(class_metrics_data, dict):
                    # Already dictionary format
                    class_metrics = class_metrics_data
                elif isinstance(class_metrics_data, list) and self.class_names:
                    # List structure, convert to dict using class names
                    for i, class_name in enumerate(self.class_names):
                        if i < len(class_metrics_data):
                            class_metrics[class_name] = class_metrics_data[i]
                else:
                    logger.warning(
                        f"Unrecognized class_metrics format: {type(class_metrics_data)}"
                    )

                logger.info(f"Processed metrics for {len(class_metrics)} classes")

            # Copy basic metrics to be saved
            report_metrics = {
                k: _convert_omegaconf_to_python(v)
                for k, v in final_metrics.items()
                if k
                not in ["class_metrics", "confusion_matrix", "final_metrics", "model"]
            }

            # Remove model key if it exists to avoid serialization issues
            if "model" in report_metrics:
                del report_metrics["model"]

            # Add class metrics
            report_metrics.update(class_metrics)

            # Save metrics.json for report
            metrics_path = metrics_dir / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(report_metrics, f, indent=2)
            logger.info(f"Saved metrics.json to {metrics_path}")

            # Save history.json in correct format
            history_dict = self._convert_history_format()
            history_dict = _convert_omegaconf_to_python(history_dict)

            history_path = metrics_dir / "history.json"
            with open(history_path, "w") as f:
                json.dump(history_dict, f, indent=2)
            logger.info(f"Saved history.json to {history_path}")

            # Save class names
            self._save_class_names()

            # Save confusion matrix if it exists
            if (
                "confusion_matrix" in final_metrics
                and final_metrics["confusion_matrix"] is not None
            ):
                # Convert to numpy array and handle any OmegaConf objects
                confusion_matrix = _convert_omegaconf_to_python(
                    final_metrics["confusion_matrix"]
                )
                cm_path = metrics_dir / "evaluation_artifacts" / "confusion_matrix.npy"
                np.save(cm_path, np.array(confusion_matrix))
                logger.info(f"Saved confusion matrix to {cm_path}")

        except Exception as e:
            import traceback

            logger.error(f"Error saving report files: {e}")
            logger.error(traceback.format_exc())

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Final save of metrics at the end of training."""
        logs = logs or {}

        # Add best values to the final metrics if available
        final_metrics = {}
        for key in ["best_val_loss", "best_val_acc", "best_epoch", "total_time"]:
            if key in logs:
                final_metrics[key] = logs[key]

        # Add overall metrics
        for key in ["accuracy", "precision", "recall", "f1", "val_loss"]:
            for metrics in reversed(self.history):
                if key in metrics:
                    final_metrics[key] = metrics[key]
                    break

        # Add class metrics if available in logs
        if "class_metrics" in logs and logs["class_metrics"]:
            final_metrics["class_metrics"] = logs["class_metrics"]
            logger.info("Added class metrics from logs to final metrics")

        # Add confusion matrix if available in logs
        if "confusion_matrix" in logs and logs["confusion_matrix"] is not None:
            final_metrics["confusion_matrix"] = logs["confusion_matrix"]
            logger.info("Added confusion matrix from logs to final metrics")

        if final_metrics:
            # Add to history
            self.history.append({"final_metrics": True, **final_metrics})

            # Save metrics
            self._save_metrics()

            # Save files needed for report generation
            self._save_report_files(final_metrics)

            logger.info(f"Final metrics saved to {self._get_filepath()}")
