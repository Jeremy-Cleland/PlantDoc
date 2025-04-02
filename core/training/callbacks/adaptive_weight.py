"""
Adaptive Weight Adjustment for loss functions during training.

This callback monitors model confidence and adjusts loss function parameters
such as Focal Loss gamma and Center Loss weight to improve training dynamics.
"""

from typing import Any, Dict, Optional

from core.training.callbacks.base import Callback
from core.training.callbacks.confidence_monitor import ConfidenceMonitorCallback
from utils.logger import get_logger

logger = get_logger(__name__)


class AdaptiveWeightAdjustmentCallback(Callback):
    """
    Adaptively adjusts loss weights based on confidence monitoring.

    Requires a ConfidenceMonitorCallback to be present in the trainer's callbacks.
    Can adjust Focal Loss gamma and/or other loss component weights.

    Args:
        **kwargs: Configuration parameters. Expected keys:
            confidence_callback (ConfidenceMonitorCallback, optional): Explicit reference. If None, attempts to find it.
            focal_loss_gamma_range (Tuple[float, float]): Min/max gamma for Focal Loss. Default: (2.0, 5.0).
            center_loss_weight_range (Tuple[float, float]): Min/max weight for Center Loss. Default: (0.5, 2.0).
            adjust_frequency (int): Adjust weights every N epochs. Default: 5.
    """

    priority = 35  # Lower than ConfidenceMonitorCallback (prio 40)

    def __init__(self, **kwargs):
        super().__init__()  # Priority set as class attribute

        # Pop 'enabled' - caller should handle this
        kwargs.pop("enabled", None)

        # Explicitly allow passing the confidence callback instance
        self.confidence_callback = kwargs.get("confidence_callback")
        if self.confidence_callback is not None and not isinstance(
            self.confidence_callback, ConfidenceMonitorCallback
        ):
            logger.warning(
                f"Provided 'confidence_callback' is not an instance of ConfidenceMonitorCallback (got {type(self.confidence_callback).__name__}). Will try to find one."
            )
            self.confidence_callback = None  # Reset if wrong type

        # Extract other parameters
        self.focal_loss_gamma_range = tuple(
            kwargs.get("focal_loss_gamma_range", (2.0, 5.0))
        )
        self.center_loss_weight_range = tuple(
            kwargs.get("center_loss_weight_range", (0.5, 2.0))
        )
        self.adjust_frequency = kwargs.get("adjust_frequency", 5)

        # Validate ranges and frequency
        if not (
            isinstance(self.focal_loss_gamma_range, tuple)
            and len(self.focal_loss_gamma_range) == 2
            and self.focal_loss_gamma_range[0] <= self.focal_loss_gamma_range[1]
        ):
            raise ValueError("focal_loss_gamma_range must be a tuple of (min, max)")
        if not (
            isinstance(self.center_loss_weight_range, tuple)
            and len(self.center_loss_weight_range) == 2
            and self.center_loss_weight_range[0] <= self.center_loss_weight_range[1]
        ):
            raise ValueError("center_loss_weight_range must be a tuple of (min, max)")
        if self.adjust_frequency < 1:
            raise ValueError("adjust_frequency must be >= 1")

        # Internal state
        self._focal_loss_criterion = None  # Reference to the loss object with 'gamma'
        self._center_loss_criterion = None  # Reference to the CenterLoss object
        self._found_components = False

        logger.info(
            f"Initialized AdaptiveWeightAdjustmentCallback: adjust_freq={self.adjust_frequency}, "
            f"gamma_range={self.focal_loss_gamma_range}, center_weight_range={self.center_loss_weight_range}"
        )

        # Warn about unused keys (excluding confidence_callback handled above)
        unused_keys = set(kwargs.keys()) - {
            "confidence_callback",
            "focal_loss_gamma_range",
            "center_loss_weight_range",
            "adjust_frequency",
        }
        if unused_keys:
            logger.warning(
                f"Unused config keys for AdaptiveWeightAdjustmentCallback: {list(unused_keys)}"
            )

    def _find_components(self, logs: Dict[str, Any]):
        """Internal helper to find relevant callbacks and loss components."""
        if self._found_components:  # Don't search repeatedly
            return

        # --- Find ConfidenceMonitorCallback ---
        if self.confidence_callback is None:
            callbacks = logs.get("callbacks", [])
            for callback in callbacks:
                if isinstance(callback, ConfidenceMonitorCallback):
                    self.confidence_callback = callback
                    logger.info(
                        "AdaptiveWeightAdjustment: Found ConfidenceMonitorCallback instance."
                    )
                    break
            if self.confidence_callback is None:
                logger.warning(
                    "AdaptiveWeightAdjustment: ConfidenceMonitorCallback not found in trainer callbacks. Cannot perform adjustments."
                )
                # No need to search for loss components if monitor is missing
                self._found_components = (
                    True  # Mark search as done (even if unsuccessful)
                )
                return

        # --- Find Loss Components ---
        criterion = logs.get("criterion")
        if criterion is None:
            logger.warning(
                "AdaptiveWeightAdjustment: 'criterion' not found in logs. Cannot find loss components."
            )
            self._found_components = True
            return

        # Check for Focal Loss gamma (simplified check)
        # Needs improvement if Focal Loss is wrapped deeper
        if hasattr(criterion, "gamma"):
            self._focal_loss_criterion = criterion
            logger.info(
                "AdaptiveWeightAdjustment: Found potential Focal Loss with 'gamma' attribute directly on criterion."
            )
        # Check for combined loss with focal component
        elif hasattr(criterion, "focal_loss") and hasattr(
            criterion.focal_loss, "gamma"
        ):
            self._focal_loss_criterion = criterion.focal_loss
            logger.info(
                "AdaptiveWeightAdjustment: Found Focal Loss via 'focal_loss' attribute."
            )

        # Check for Center Loss component
        # Direct usage as criterion - assuming our CenterLoss has a 'weight' attribute
        if hasattr(criterion, "weight") and hasattr(
            criterion, "centers"
        ):  # Basic CenterLoss check
            self._center_loss_criterion = criterion
            logger.info(
                "AdaptiveWeightAdjustment: Found potential CenterLoss directly as criterion."
            )
        # Check for center loss as a component
        elif hasattr(criterion, "center_loss") and hasattr(
            criterion.center_loss, "weight"
        ):
            self._center_loss_criterion = criterion.center_loss
            logger.info(
                "AdaptiveWeightAdjustment: Found CenterLoss via 'center_loss' attribute."
            )
        # For our CombinedLoss implementation
        elif hasattr(criterion, "loss_weights") and hasattr(criterion, "loss_fns"):
            # Search through components in a combined loss setup
            for i, loss_fn in enumerate(criterion.loss_fns):
                # Check if this component might be a center loss
                if hasattr(loss_fn, "centers") and hasattr(loss_fn, "weight"):
                    self._center_loss_criterion = loss_fn
                    logger.info(
                        f"AdaptiveWeightAdjustment: Found CenterLoss at index {i} of combined loss."
                    )
                # Check if this component might be a focal loss
                elif hasattr(loss_fn, "gamma"):
                    self._focal_loss_criterion = loss_fn
                    logger.info(
                        f"AdaptiveWeightAdjustment: Found Focal Loss at index {i} of combined loss."
                    )

        if self._focal_loss_criterion is None:
            logger.info(
                "AdaptiveWeightAdjustment: No Focal Loss component with 'gamma' found."
            )
        if self._center_loss_criterion is None:
            logger.info("AdaptiveWeightAdjustment: No CenterLoss component found.")

        self._found_components = True  # Mark search complete

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Find components at the start of training."""
        self._found_components = False  # Reset search flag
        if logs:
            self._find_components(logs)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Perform weight adjustments based on confidence."""
        epoch_1based = epoch + 1

        # Ensure components are found (might happen on first epoch_end if not in train_begin logs)
        if not self._found_components and logs:
            self._find_components(logs)

        # Check frequency and required components
        if epoch_1based % self.adjust_frequency != 0:
            return
        if self.confidence_callback is None:
            return  # Already warned if missing
        if self._focal_loss_criterion is None and self._center_loss_criterion is None:
            return

        # Check confidence history
        history = getattr(self.confidence_callback, "history", [])
        if len(history) < 2:
            logger.debug(
                f"AdaptiveWeightAdjustment (Epoch {epoch_1based}): Not enough confidence history ({len(history)} records) to calculate trend."
            )
            return

        # Get current and previous confidence metrics
        try:
            current_metrics = history[-1]
            previous_metrics = history[-2]
            current_confidence = current_metrics.get("mean_confidence")
            previous_confidence = previous_metrics.get("mean_confidence")

            if current_confidence is None or previous_confidence is None:
                logger.warning(
                    "AdaptiveWeightAdjustment: Missing 'mean_confidence' in confidence history. Skipping adjustment."
                )
                return

            confidence_change = current_confidence - previous_confidence
            conf_threshold = getattr(
                self.confidence_callback, "threshold_warning", 0.7
            )  # Use actual threshold from monitor

        except (IndexError, KeyError, AttributeError) as e:
            logger.warning(
                f"AdaptiveWeightAdjustment: Error accessing confidence history: {e}. Skipping adjustment."
            )
            return

        logger.info(
            f"AdaptiveWeightAdjustment (Epoch {epoch_1based}): Current confidence={current_confidence:.4f}, Change={confidence_change:+.4f}"
        )

        # --- Adjust Focal Loss Gamma ---
        if self._focal_loss_criterion is not None:
            current_gamma = float(getattr(self._focal_loss_criterion, "gamma", -1.0))
            if current_gamma < 0:
                logger.warning(
                    "AdaptiveWeightAdjustment: Could not read current focal loss gamma."
                )
            else:
                min_gamma, max_gamma = self.focal_loss_gamma_range
                new_gamma = current_gamma  # Start with current value

                # Logic: If confidence is low and not improving much, increase gamma (focus more on hard examples)
                if current_confidence < conf_threshold and confidence_change < 0.01:
                    new_gamma = min(current_gamma * 1.1, max_gamma)
                # Logic: If confidence is high, decrease gamma slightly (less focus on hard examples)
                elif current_confidence > (conf_threshold + 0.1):  # Add buffer
                    new_gamma = max(current_gamma * 0.95, min_gamma)

                if (
                    abs(new_gamma - current_gamma) > 1e-4
                ):  # Only update if changed significantly
                    try:
                        self._focal_loss_criterion.gamma = new_gamma
                        logger.info(
                            f"  Adjusted Focal Loss gamma: {current_gamma:.2f} -> {new_gamma:.2f}"
                        )
                    except Exception as e:
                        logger.error(f"  Failed to set focal loss gamma: {e}")

        # --- Adjust Center Loss Weight ---
        if self._center_loss_criterion is not None:
            current_weight = float(getattr(self._center_loss_criterion, "weight", -1.0))
            if current_weight < 0:
                logger.warning(
                    "AdaptiveWeightAdjustment: Could not read current center loss weight."
                )
            else:
                min_weight, max_weight = self.center_loss_weight_range
                new_weight = current_weight  # Start with current value

                # Logic: If confidence is low, increase center loss weight (pull features closer)
                if current_confidence < conf_threshold:
                    new_weight = min(current_weight * 1.2, max_weight)
                # Logic: If confidence is high, decrease center loss weight (allow more spread if classes are separable)
                elif current_confidence > (conf_threshold + 0.1):  # Add buffer
                    new_weight = max(current_weight * 0.9, min_weight)

                if (
                    abs(new_weight - current_weight) > 1e-4
                ):  # Only update if changed significantly
                    try:
                        self._center_loss_criterion.weight = new_weight
                        logger.info(
                            f"  Adjusted Center Loss weight: {current_weight:.2f} -> {new_weight:.2f}"
                        )
                    except Exception as e:
                        logger.error(f"  Failed to set center loss weight: {e}")

        # Add adjustments to logs for tracking
        if logs is not None:
            if self._focal_loss_criterion is not None:
                logs["focal_loss_gamma"] = getattr(
                    self._focal_loss_criterion, "gamma", None
                )
            if self._center_loss_criterion is not None:
                logs["center_loss_weight"] = getattr(
                    self._center_loss_criterion, "weight", None
                )
