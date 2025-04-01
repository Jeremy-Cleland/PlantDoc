"""
ConfidenceMonitorCallback for tracking model confidence and calibration during training.
"""

import numbers
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from core.training.callbacks.base import Callback
from utils.logger import get_logger

logger = get_logger(__name__)


class ConfidenceMonitorCallback(Callback):
    """
    Monitors model confidence and calibration metrics during validation.

    Tracks confidence distributions, Expected Calibration Error (ECE),
    and warns about abnormally high or low confidence levels.

    Args:
        **kwargs: Configuration parameters. Expected keys:
            monitor_frequency (int): How often to calculate metrics (epochs). Default: 1.
            threshold_warning (float): Threshold for confidence warnings. Default: 0.7.
            ece_bins (int): Number of bins for ECE calculation. Default: 10.
            log_per_class (bool): Whether to log per-class confidence. Default: True.
    """

    priority = 40

    def __init__(self, **kwargs):
        super().__init__()
        kwargs.pop("enabled", None)
        self.monitor_frequency = kwargs.get("monitor_frequency", 1)
        self.threshold_warning = kwargs.get("threshold_warning", 0.7)
        self.ece_bins = kwargs.get("ece_bins", 10)
        self.log_per_class = kwargs.get("log_per_class", True)

        # Validate parameters
        if self.monitor_frequency < 1:
            raise ValueError("monitor_frequency must be >= 1")
        if not (0 < self.threshold_warning < 1):
            raise ValueError("threshold_warning must be between 0 and 1")
        if self.ece_bins < 1:
            raise ValueError("ece_bins must be >= 1")

        self._epoch_scores: List[Tuple[float, int, int]] = []
        self.history: List[Dict[str, Any]] = []

        logger.info(
            f"Initialized ConfidenceMonitorCallback: Freq={self.monitor_frequency}, "
            f"Warn={self.threshold_warning:.2f}, ECE Bins={self.ece_bins}, "
            f"PerClass={self.log_per_class}"
        )

        unused_keys = set(kwargs.keys()) - {
            "monitor_frequency",
            "threshold_warning",
            "ece_bins",
            "log_per_class",
        }
        if unused_keys:
            logger.warning(
                f"Unused config keys for ConfidenceMonitorCallback: {list(unused_keys)}"
            )

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Collect confidence scores during validation."""
        logs = logs or {}
        is_validation = logs.get("val_phase", False) or logs.get("is_validation", False)
        if not is_validation:
            return

        outputs, targets = logs.get("outputs"), logs.get("targets")
        if outputs is None or targets is None:
            return
        if not isinstance(outputs, torch.Tensor) or not isinstance(
            targets, torch.Tensor
        ):
            return
        if outputs.ndim < 2 or targets.ndim < 1:
            return
        if outputs.shape[0] != targets.shape[0]:
            return

        try:
            probs = F.softmax(outputs.float(), dim=1)
            confidence, predicted = torch.max(probs, dim=1)
            confidence_np = confidence.detach().cpu().numpy()
            predicted_np = predicted.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy().astype(int)
            self._epoch_scores.extend(zip(confidence_np, predicted_np, targets_np))
        except Exception as e:
            logger.error(
                f"ConfidenceMonitor: Error processing batch: {e}", exc_info=True
            )

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Calculate confidence metrics at the end of validation epochs."""
        logs = logs or {}
        epoch_1based = epoch + 1
        if epoch_1based % self.monitor_frequency != 0:
            return
        if not self._epoch_scores:
            logger.warning(
                f"ConfidenceMonitor (Epoch {epoch_1based}): No validation scores collected."
            )
            return

        num_samples = len(self._epoch_scores)
        logger.info(
            f"ConfidenceMonitor: Processing {num_samples} validation samples for epoch {epoch_1based}."
        )

        confidence_values = np.array([s[0] for s in self._epoch_scores])
        predictions = np.array([s[1] for s in self._epoch_scores])
        targets = np.array([s[2] for s in self._epoch_scores])

        mean_confidence = np.mean(confidence_values)
        median_confidence = np.median(confidence_values)
        min_confidence = np.min(confidence_values)
        accuracy = np.mean(predictions == targets)

        low_conf_count = np.sum(confidence_values < self.threshold_warning)
        low_conf_percent = (
            (low_conf_count / num_samples) * 100 if num_samples > 0 else 0
        )

        ece = self._calculate_ece(confidence_values, predictions, targets, num_samples)

        per_class_stats = (
            self._calculate_per_class_confidence(self._epoch_scores)
            if self.log_per_class
            else {}
        )

        epoch_summary = {
            "epoch": epoch_1based,
            "num_samples": num_samples,
            "mean_confidence": mean_confidence,
            "median_confidence": median_confidence,
            "min_confidence": min_confidence,
            "accuracy": accuracy,
            "low_conf_percent": low_conf_percent,
            "ece": ece,
            **{f"class_{c}_mean_conf": s["mean"] for c, s in per_class_stats.items()},
            **{f"class_{c}_count": s["count"] for c, s in per_class_stats.items()},
        }
        self.history.append(epoch_summary)

        logger.info(f"--- Confidence Metrics (Epoch {epoch_1based}) ---")
        logger.info(f"  Accuracy:             {accuracy:.4f}")
        logger.info(
            f"  Mean Confidence:      {mean_confidence:.4f} | Median: {median_confidence:.4f} | Min: {min_confidence:.4f}"
        )
        logger.info(
            f"  Low Conf (<{self.threshold_warning:.2f}):    {low_conf_percent:.2f}% ({low_conf_count}/{num_samples})"
        )
        logger.info(f"  ECE ({self.ece_bins} bins):         {ece:.4f}")

        if self.log_per_class and per_class_stats:
            logger.info("  Per-Class Mean Confidence (Correct Predictions):")
            [
                logger.info(f"    Class {c}: {s['mean']:.4f} (Count: {s['count']})")
                for c, s in sorted(per_class_stats.items())
            ]

        if mean_confidence < self.threshold_warning:
            logger.warning(
                f"Epoch {epoch_1based}: Mean confidence ({mean_confidence:.4f}) below threshold ({self.threshold_warning:.2f})!"
            )

        logs.update(
            {f"val_{k}": v for k, v in epoch_summary.items() if k != "epoch"}
        )  # Update logs with val_ prefix
        self._epoch_scores = []

    def _calculate_ece(self, confidences, predictions, targets, num_samples):
        """Calculate Expected Calibration Error."""
        if num_samples == 0:
            return 0.0

        ece = 0.0
        bin_boundaries = np.linspace(0, 1, self.ece_bins + 1)

        for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            in_bin |= (confidences == 1.0) & (bin_upper == 1.0)
            bin_n = np.sum(in_bin)

            if bin_n > 0:
                ece += abs(
                    np.mean(predictions[in_bin] == targets[in_bin])
                    - np.mean(confidences[in_bin])
                ) * (bin_n / num_samples)

        return ece

    def _calculate_per_class_confidence(self, scores):
        """Calculate per-class confidence statistics."""
        pc = defaultdict(list)
        targets_seen = set()

        for conf, pred, target in scores:
            targets_seen.add(int(target))
            if pred == target:  # Only include correct predictions
                pc[int(target)].append(conf)

        return {
            c: {"mean": np.mean(pc[c]) if pc[c] else 0.0, "count": len(pc[c])}
            for c in sorted(list(targets_seen))
        }
