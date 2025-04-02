"""
Experiment registry for tracking model versions and experiment paths.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from utils.logger import get_logger
from utils.paths import ensure_dir, get_project_root

logger = get_logger(__name__)

# Registry file path
REGISTRY_DIR = get_project_root() / "outputs" / "registry"
REGISTRY_FILE = REGISTRY_DIR / "experiment_registry.json"


def get_next_version(model_name: str) -> int:
    """
    Get the next version number for a model.

    Args:
        model_name: The name of the model

    Returns:
        Next version number (1-based)
    """
    registry = load_registry()

    # Find all experiments for this model
    model_experiments = [
        exp
        for exp in registry.get("experiments", [])
        if exp.get("model_name") == model_name
    ]

    if not model_experiments:
        return 1

    # Find the highest version number
    versions = [exp.get("version", 0) for exp in model_experiments]
    return max(versions) + 1


def get_experiment_name(model_name: str, version: Optional[int] = None) -> str:
    """
    Generate a standardized experiment name from model name and version.

    Args:
        model_name: Name of the model
        version: Version number (if None, will be auto-incremented)

    Returns:
        Formatted experiment name
    """
    # Clean model name for use in paths (remove spaces, special chars)
    clean_name = model_name.replace(" ", "_").lower()

    # Get next version if not provided
    if version is None:
        version = get_next_version(model_name)

    # Format as model_name_v{version}
    return f"{clean_name}_v{version}"


def register_experiment(
    model_name: str,
    experiment_dir: Union[str, Path],
    params: Optional[Dict] = None,
    version: Optional[int] = None,
    description: Optional[str] = None,
) -> Dict:
    """
    Register an experiment in the registry.

    Args:
        model_name: Name of the model
        experiment_dir: Path to the experiment directory
        params: Model parameters
        version: Version number (if None, will be auto-incremented)
        description: Optional description

    Returns:
        The registered experiment info
    """
    registry = load_registry()

    # Ensure experiments list exists
    if "experiments" not in registry:
        registry["experiments"] = []

    # Get version if not provided
    if version is None:
        version = get_next_version(model_name)

    # Convert experiment_dir to string if it's a Path
    experiment_dir_str = str(experiment_dir)

    # Extract more comprehensive information from params
    full_params = params or {}

    # Create experiment info with enhanced details
    experiment = {
        "model_name": model_name,
        "version": version,
        "experiment_dir": experiment_dir_str,
        "created_at": datetime.now().isoformat(),
        "params": full_params,
        "description": description or f"Training run for {model_name} v{version}",
        "status": "running",
        "training": {
            "epochs": full_params.get("epochs", None),
            "batch_size": full_params.get("batch_size", None),
            "learning_rate": full_params.get("learning_rate", None),
            "optimizer": (
                full_params.get("optimizer", {}).get("name", None)
                if isinstance(full_params.get("optimizer", {}), dict)
                else None
            ),
            "loss": (
                full_params.get("loss", {}).get("name", None)
                if isinstance(full_params.get("loss", {}), dict)
                else None
            ),
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
            "duration_seconds": None,
        },
        "performance": {
            "best_epoch": None,
            "val_loss": None,
            "val_accuracy": None,
            "val_precision": None,
            "val_recall": None,
            "val_f1": None,
        },
        "hardware": {
            "device": full_params.get("device", None),
            "precision": full_params.get("precision", None),
        },
        "dataset": {
            "name": full_params.get("dataset_name", None),
            "num_classes": full_params.get("num_classes", None),
        },
    }

    # Add to registry
    registry["experiments"].append(experiment)

    # Save registry
    save_registry(registry)

    logger.info(
        f"Registered experiment: {model_name} v{version} in {experiment_dir_str}"
    )
    return experiment


def update_experiment_info(
    experiment_dir: Union[str, Path], results: Dict, status: str = "completed"
) -> Dict:
    """
    Update experiment information with training results.

    Args:
        experiment_dir: Path to the experiment directory
        results: Dictionary containing training results
        status: Status of the experiment (completed, failed, etc.)

    Returns:
        Updated experiment info
    """
    registry = load_registry()
    experiment_dir_str = str(experiment_dir)

    # Find the experiment in the registry
    for exp in registry.get("experiments", []):
        if exp.get("experiment_dir") == experiment_dir_str:
            # Update status
            exp["status"] = status

            # Update training information
            if "training" not in exp:
                exp["training"] = {}

            exp["training"]["completed_at"] = datetime.now().isoformat()

            # Calculate duration if possible
            if "started_at" in exp["training"]:
                try:
                    start_time = datetime.fromisoformat(exp["training"]["started_at"])
                    end_time = datetime.fromisoformat(exp["training"]["completed_at"])
                    duration = (end_time - start_time).total_seconds()
                    exp["training"]["duration_seconds"] = duration
                except Exception as e:
                    logger.warning(f"Could not calculate training duration: {e}")

            # Update performance metrics
            if "performance" not in exp:
                exp["performance"] = {}

            # Extract performance metrics from results
            performance = exp["performance"]
            performance["best_epoch"] = results.get("best_epoch")
            performance["val_loss"] = results.get("best_val_loss") or results.get(
                "val_loss"
            )
            performance["val_accuracy"] = results.get("best_val_acc") or results.get(
                "val_accuracy"
            )
            performance["val_precision"] = results.get("val_precision")
            performance["val_recall"] = results.get("val_recall")
            performance["val_f1"] = results.get("val_f1")

            # Add additional metrics if provided
            for k, v in results.items():
                if k not in [
                    "best_epoch",
                    "best_val_loss",
                    "best_val_acc",
                    "val_accuracy",
                    "val_precision",
                    "val_recall",
                    "val_f1",
                ]:
                    if k.startswith("val_"):
                        performance[k] = v

            # Save registry
            save_registry(registry)

            logger.info(f"Updated experiment info for {experiment_dir_str}")
            return exp

    logger.warning(f"Experiment not found in registry: {experiment_dir_str}")
    return {}


def load_registry() -> Dict:
    """
    Load the experiment registry.

    Returns:
        Registry data or empty dict if not found
    """
    ensure_dir(REGISTRY_DIR)

    if not REGISTRY_FILE.exists():
        return {"experiments": []}

    try:
        with open(REGISTRY_FILE) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading registry: {e}")
        return {"experiments": []}


def save_registry(registry: Dict) -> None:
    """
    Save the experiment registry.

    Args:
        registry: Registry data to save
    """
    ensure_dir(REGISTRY_DIR)

    try:
        # Convert registry to a serializable format without circular references
        def convert_to_serializable(obj, depth=0):
            # Prevent excessive recursion
            if depth > 20:  # Reasonable limit to prevent stack overflow
                return "<max recursion depth reached>"

            if hasattr(obj, "__dict__"):
                return {
                    k: convert_to_serializable(v, depth + 1)
                    for k, v in obj.__dict__.items()
                    if not k.startswith("_")
                }  # Skip private attributes
            elif isinstance(obj, dict):
                return {
                    k: convert_to_serializable(v, depth + 1) for k, v in obj.items()
                }
            elif isinstance(obj, list) or str(type(obj)).endswith("ListConfig'>"):
                return [convert_to_serializable(item, depth + 1) for item in obj]
            elif str(type(obj)).endswith("DictConfig'>"):
                return {
                    k: convert_to_serializable(v, depth + 1) for k, v in obj.items()
                }
            # Handle numpy arrays and torch tensors
            elif hasattr(obj, "tolist"):
                return obj.tolist()
            elif hasattr(obj, "item") and callable(obj.item):
                try:
                    return obj.item()
                except:
                    return str(obj)
            else:
                # Try to make it JSON serializable
                try:
                    json.dumps(obj)
                    return obj
                except:
                    return str(obj)

        serializable_registry = convert_to_serializable(registry)

        with open(REGISTRY_FILE, "w") as f:
            json.dump(serializable_registry, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving registry: {e}")
        if isinstance(e, RecursionError):
            logger.error(
                "Recursion error detected, likely due to circular references in the data"
            )
            # Attempt a simpler serialization
            try:
                simplified_registry = {"experiments": []}
                for exp in registry.get("experiments", []):
                    simplified_exp = {
                        "model_name": exp.get("model_name", "unknown"),
                        "version": exp.get("version", 0),
                        "experiment_dir": str(exp.get("experiment_dir", "")),
                        "created_at": exp.get("created_at", datetime.now().isoformat()),
                        "description": exp.get("description", ""),
                    }
                    # Extract basic params
                    if "params" in exp:
                        try:
                            params = exp["params"]
                            simplified_params = {}
                            for key in [
                                "num_classes",
                                "pretrained",
                                "in_channels",
                                "input_size",
                            ]:
                                if key in params:
                                    simplified_params[key] = params[key]
                            simplified_exp["params"] = simplified_params
                        except:
                            simplified_exp["params"] = {}

                    simplified_registry["experiments"].append(simplified_exp)

                with open(REGISTRY_FILE, "w") as f:
                    json.dump(simplified_registry, f, indent=2)
                logger.info("Saved simplified registry as fallback")
            except Exception as e2:
                logger.error(f"Failed to save simplified registry: {e2}")


def get_experiment_info(experiment_name: str) -> Optional[Dict]:
    """
    Get information about an experiment.

    Args:
        experiment_name: Name of the experiment

    Returns:
        Experiment info or None if not found
    """
    registry = load_registry()

    for exp in registry.get("experiments", []):
        # Match based on the directory name or formatted name
        exp_dir = Path(exp.get("experiment_dir", ""))
        if (
            exp_dir.name == experiment_name
            or f"{exp.get('model_name')}_{exp.get('version')}" == experiment_name
        ):
            return exp

    return None


def list_experiments(model_name: Optional[str] = None) -> List[Dict]:
    """
    List all experiments or filter by model name.

    Args:
        model_name: Optional filter by model name

    Returns:
        List of experiment info
    """
    registry = load_registry()

    experiments = registry.get("experiments", [])

    if model_name:
        experiments = [
            exp for exp in experiments if exp.get("model_name") == model_name
        ]

    # Sort by model name and version
    return sorted(
        experiments, key=lambda x: (x.get("model_name", ""), x.get("version", 0))
    )
