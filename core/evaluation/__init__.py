"""
Evaluation and interpretability tools for PlantDoc.
"""

from plantdoc.core.evaluation.interpretability import (
    GradCAM,
    visualize_gradcam,
    explain_model_predictions,
)

__all__ = [
    "GradCAM",
    "visualize_gradcam",
    "explain_model_predictions",
]
