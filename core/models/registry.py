# Path: PlantDoc/core/models/registry.py

"""
Enhanced registry for model classes with parameter metadata.
"""

from typing import Any, Dict, List, Optional, Type, Union

from core.models.base import BaseModel
from utils.logger import get_logger

logger = get_logger(__name__)

# Global registry of models with their metadata
MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {}
MODEL_METADATA: Dict[str, Dict[str, Any]] = {}


def register_model(name, model_cls=None, **metadata):
    """
    Register a model in the model registry with metadata.

    Can be used as a decorator:
    @register_model("name", param1={"type": "int", "default": 10})
    class ModelClass: ...

    Or called directly:
    register_model("name", ModelClass, param1={"type": "int", "default": 10})

    Args:
        name: A string identifier for the model
        model_cls: Optional model class for direct registration
        **metadata: Optional metadata for parameter validation and documentation

    Returns:
        The model class or a decorator function
    """
    if model_cls is not None:
        # Direct registration
        MODEL_REGISTRY[name] = model_cls
        MODEL_METADATA[name] = metadata
        return model_cls

    # Decorator usage
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        MODEL_METADATA[name] = metadata
        
        # Attach metadata to the class for introspection
        if not hasattr(cls, "model_metadata"):
            cls.model_metadata = {}
        cls.model_metadata[name] = metadata
        
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
        raise ValueError(f"Unknown model: {name}. Available models: {list(MODEL_REGISTRY.keys())}")

    return MODEL_REGISTRY[name]


def get_model_metadata(name: str) -> Dict[str, Any]:
    """
    Get metadata for a model by name.

    Args:
        name: Name of the model

    Returns:
        Model metadata dictionary
    """
    if name not in MODEL_METADATA:
        raise ValueError(f"Unknown model: {name}. Available models: {list(MODEL_METADATA.keys())}")

    return MODEL_METADATA[name]


def validate_model_params(name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate parameters against model metadata and apply defaults.

    Args:
        name: Name of the model
        params: Parameters to validate

    Returns:
        Validated parameters with defaults applied
    """
    if name not in MODEL_METADATA:
        logger.warning(f"No metadata found for model {name}. Skipping validation.")
        return params

    metadata = MODEL_METADATA[name]
    validated_params = {}

    # Add all parameters with defaults
    for param_name, param_meta in metadata.items():
        if param_name in params:
            # Perform type validation if specified
            if "type" in param_meta:
                param_type = param_meta["type"]
                param_value = params[param_name]
                
                # Basic type checking
                if param_type == "int" and not isinstance(param_value, int):
                    logger.warning(f"Parameter {param_name} should be an integer. Got {type(param_value).__name__}.")
                elif param_type == "float" and not isinstance(param_value, (int, float)):
                    logger.warning(f"Parameter {param_name} should be a float. Got {type(param_value).__name__}.")
                elif param_type == "bool" and not isinstance(param_value, bool):
                    logger.warning(f"Parameter {param_name} should be a boolean. Got {type(param_value).__name__}.")
                elif param_type == "str" and not isinstance(param_value, str):
                    logger.warning(f"Parameter {param_name} should be a string. Got {type(param_value).__name__}.")
                elif param_type == "list" and not isinstance(param_value, list):
                    logger.warning(f"Parameter {param_name} should be a list. Got {type(param_value).__name__}.")
                elif param_type == "dict" and not isinstance(param_value, dict):
                    logger.warning(f"Parameter {param_name} should be a dictionary. Got {type(param_value).__name__}.")
                
                # Range checking if specified
                if "range" in param_meta and isinstance(param_value, (int, float)):
                    min_val, max_val = param_meta["range"]
                    if param_value < min_val or param_value > max_val:
                        logger.warning(f"Parameter {param_name} should be in range [{min_val}, {max_val}]. Got {param_value}.")
                
                # Choice validation if specified
                if "choices" in param_meta and param_value not in param_meta["choices"]:
                    logger.warning(f"Parameter {param_name} should be one of {param_meta['choices']}. Got {param_value}.")
            
            validated_params[param_name] = params[param_name]
        elif "default" in param_meta:
            validated_params[param_name] = param_meta["default"]
        elif "required" in param_meta and param_meta["required"]:
            logger.error(f"Required parameter {param_name} not provided for model {name}.")
            raise ValueError(f"Required parameter {param_name} not provided for model {name}.")

    # Log any extra parameters not in metadata
    for param_name in params:
        if param_name not in metadata:
            logger.warning(f"Unknown parameter {param_name} for model {name}.")
            validated_params[param_name] = params[param_name]

    return validated_params


def list_models() -> Dict[str, Dict[str, Any]]:
    """
    Get all registered models with their metadata.

    Returns:
        Dictionary mapping model names to their metadata
    """
    return {name: {"class": MODEL_REGISTRY[name], "metadata": MODEL_METADATA.get(name, {})} 
            for name in MODEL_REGISTRY}


def get_model_param_schema(name: str) -> Dict[str, Any]:
    """
    Get a schema of parameters for a model.

    Args:
        name: Name of the model

    Returns:
        Dictionary with parameter schema
    """
    if name not in MODEL_METADATA:
        logger.warning(f"No metadata found for model {name}.")
        return {}

    metadata = MODEL_METADATA[name]
    schema = {}

    for param_name, param_meta in metadata.items():
        schema[param_name] = {
            "type": param_meta.get("type", "any"),
            "default": param_meta.get("default", None),
            "description": param_meta.get("description", ""),
            "required": param_meta.get("required", False),
        }
        
        if "range" in param_meta:
            schema[param_name]["range"] = param_meta["range"]
        
        if "choices" in param_meta:
            schema[param_name]["choices"] = param_meta["choices"]

    return schema
