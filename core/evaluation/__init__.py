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
    ClassificationMetrics,
    IncrementalMetricsCalculator,
    calculate_accuracy,
    calculate_all_metrics,
    calculate_confusion_matrix,
    calculate_per_class_metrics,
    calculate_precision_recall_f1,
    plot_confusion_matrix,
    plot_metrics_history,
)
from core.evaluation.shap_evaluation import (
    batch_explain_with_shap,
    compare_gradcam_and_shap,
    evaluate_with_shap,
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
    "ClassificationMetrics",
    "IncrementalMetricsCalculator",
    "plot_confusion_matrix",
    "plot_metrics_history",
    # SHAP
    "evaluate_with_shap",
    "batch_explain_with_shap",
    "compare_gradcam_and_shap",
    "explain_image_with_shap",
]
