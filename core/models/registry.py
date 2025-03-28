# Path: PlantDoc/core/models/registry.py

"""
Registry for model classes.
"""

from typing import Dict, Type

from plantdoc.core.models.base import BaseModel
from plantdoc.utils.logging import get_logger

logger = get_logger(__name__)

# Global registry of models
MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {}


def register_model(name, model_cls=None):
    """
    Register a model in the model registry.

    Can be used as a decorator:
    @register_model("name")
    class ModelClass: ...

    Or called directly:
    register_model("name", ModelClass)

    Args:
        name: A string identifier for the model
        model_cls: Optional model class for direct registration

    Returns:
        The model class or a decorator function
    """
    if model_cls is not None:
        # Direct registration
        MODEL_REGISTRY[name] = model_cls
        return model_cls

    # Decorator usage
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model_class(name: str) -> Type[BaseModel]:
    """
    Get a model class by name.

    Args:
        name: Name of the model

    Returns:
        Model class
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")

    return MODEL_REGISTRY[name]


def list_models() -> Dict[str, Type[BaseModel]]:
    """
    Get all registered models.

    Returns:
        Dictionary of model names to model classes
    """
    return MODEL_REGISTRY.copy()
