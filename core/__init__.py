"""
PlantDoc core modules

Contains models, data processing, training and hyperparameter tuning utilities.
"""

from core.data import PlantDiseaseDataModule, PlantDiseaseDataset, get_transforms
from core.models import (
    BaseModel,
    CBAMResNet18Model,
    get_model_class,
    list_models,
    register_model,
)
from core.training import (
    Trainer,
    WeightedCrossEntropyLoss,
    get_loss_fn,
    get_optimizer,
    get_scheduler,
    train_model,
)

__all__ = [
    # Data
    "PlantDiseaseDataset",
    "PlantDiseaseDataModule",
    "get_transforms",
    # Models
    "CBAMResNet18Model",
    "BaseModel",
    "register_model",
    "get_model_class",
    "list_models",
    # Training
    "Trainer",
    "train_model",
    "get_loss_fn",
    "get_optimizer",
    "get_scheduler",
    "WeightedCrossEntropyLoss",
]
