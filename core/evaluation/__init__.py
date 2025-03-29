"""
Evaluation and interpretability tools for plant disease classification models.
"""

from core.evaluation.evaluate import evaluate_model
from core.evaluation.interpretability import (
    GradCAM,
    explain_model_predictions,
    visualize_gradcam,
)
from core.evaluation.interpretability import (
    evaluate_model as evaluate_model_with_gradcam,
)
from core.evaluation.metrics import (
    IncrementalMetricsCalculator,
    calculate_accuracy,
    calculate_all_metrics,
    calculate_confusion_matrix,
    calculate_per_class_metrics,
    calculate_precision_recall_f1,
)

__all__ = [
    # Model evaluation
    "evaluate_model",
    # Model interpretability
    "GradCAM",
    "visualize_gradcam",
    "explain_model_predictions",
    "evaluate_model_with_gradcam",
    # Metrics
    "calculate_accuracy",
    "calculate_all_metrics",
    "calculate_confusion_matrix",
    "calculate_per_class_metrics",
    "calculate_precision_recall_f1",
    "IncrementalMetricsCalculator",
]
