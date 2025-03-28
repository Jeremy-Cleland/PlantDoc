"""
Metrics logger callback for saving training metrics to disk.
"""

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from utils.logger import get_logger

from .base import Callback

logger = get_logger(__name__)


class MetricsLogger(Callback):
    """
    Callback to save training and validation metrics to disk.

    Args:
        metrics_dir: Directory to save metrics files
        filename: Base filename (extensions added based on format)
        save_format: Format to save metrics in ('json', 'jsonl', 'csv')
        overwrite: Whether to overwrite existing files
    """

    priority = 70  # Run after most other callbacks

    def __init__(
        self,
        metrics_dir: Union[str, Path],
        filename: str = "training_metrics",
        save_format: str = "json",
        overwrite: bool = True,
    ):
        super().__init__()
        self.metrics_dir = Path(metrics_dir)
        self.filename = filename
        self.save_format = save_format.lower()
        self.overwrite = overwrite

        # Validate format
        if self.save_format not in ["json", "jsonl", "csv"]:
            raise ValueError(
                f"Format must be 'json', 'jsonl', or 'csv', got {self.save_format}"
            )

        # Create metrics directory
        os.makedirs(self.metrics_dir, exist_ok=True)

        # Initialize metrics history
        self.history = []
        self.has_saved = False

        logger.info(
            f"Initialized MetricsLogger: "
            f"format='{self.save_format}', "
            f"overwrite={self.overwrite}"
        )

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
        with open(filepath, "w") as f:
            json.dump({"history": self.history}, f, indent=2)

    def _save_jsonl(self, filepath: Path) -> None:
        """Save metrics in JSONL format (one JSON object per line)."""
        mode = "w" if self.overwrite or not self.has_saved else "a"
        with open(filepath, mode) as f:
            for metrics in self.history:
                f.write(json.dumps(metrics) + "\n")

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

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Final save of metrics at the end of training."""
        logs = logs or {}

        # Add best values to the final metrics if available
        final_metrics = {}
        for key in ["best_val_loss", "best_val_acc", "best_epoch", "total_time"]:
            if key in logs:
                final_metrics[key] = logs[key]

        if final_metrics:
            # Add to history
            self.history.append({"final_metrics": True, **final_metrics})

            # Save metrics
            self._save_metrics()

            logger.info(f"Final metrics saved to {self._get_filepath()}")
