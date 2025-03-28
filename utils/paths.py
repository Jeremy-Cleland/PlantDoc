# paths.py stub
"""
Path handling utilities for the project.
"""

import os
from pathlib import Path
from typing import Optional, Union

from plantdoc.utils.logging import get_logger

logger = get_logger(__name__)


def get_project_root() -> Path:
    """
    Get the absolute path to the project root directory.
    
    This searches for a .git directory or setup.py file to locate the project root.
    If neither is found, it uses the current working directory.
    
    Returns:
        Path to the project root directory
    """
    # Start from the directory of this file
    current_dir = Path(__file__).resolve().parent
    
    # Traverse up looking for .git directory or setup.py file
    while current_dir != current_dir.parent:
        if (current_dir / ".git").exists() or (current_dir / "setup.py").exists():
            return current_dir
        current_dir = current_dir.parent
    
    # If we can't find a project root, use the current working directory
    logger.warning("Could not determine project root. Using current working directory.")
    return Path.cwd()


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {path}")
    return path


def get_data_dir(data_type: str = "raw") -> Path:
    """
    Get the path to a data directory.
    
    Args:
        data_type: Type of data directory ("raw", "processed", "interim", etc.)
        
    Returns:
        Path to the data directory
    """
    valid_types = ["raw", "processed", "interim", "external"]
    if data_type not in valid_types:
        logger.warning(f"Unknown data type: {data_type}. Using 'raw' instead.")
        data_type = "raw"
        
    data_dir = get_project_root() / "data" / data_type
    ensure_dir(data_dir)
    return data_dir


def get_outputs_dir(experiment_name: Optional[str] = None) -> Path:
    """
    Get the path to the outputs directory.
    
    Args:
        experiment_name: Optional experiment name for subdirectory
        
    Returns:
        Path to the outputs directory
    """
    outputs_dir = get_project_root() / "outputs"
    ensure_dir(outputs_dir)
    
    if experiment_name:
        experiment_dir = outputs_dir / experiment_name
        ensure_dir(experiment_dir)
        return experiment_dir
    
    return outputs_dir


def get_checkpoints_dir(experiment_name: str) -> Path:
    """
    Get the path to the model checkpoints directory.
    
    Args:
        experiment_name: Experiment name
        
    Returns:
        Path to the checkpoints directory
    """
    checkpoints_dir = get_outputs_dir(experiment_name) / "checkpoints"
    ensure_dir(checkpoints_dir)
    return checkpoints_dir


def get_logs_dir(experiment_name: str) -> Path:
    """
    Get the path to the logs directory.
    
    Args:
        experiment_name: Experiment name
        
    Returns:
        Path to the logs directory
    """
    logs_dir = get_outputs_dir(experiment_name) / "logs"
    ensure_dir(logs_dir)
    return logs_dir


def get_reports_dir(experiment_name: str) -> Path:
    """
    Get the path to the reports directory.
    
    Args:
        experiment_name: Experiment name
        
    Returns:
        Path to the reports directory
    """
    reports_dir = get_outputs_dir(experiment_name) / "reports"
    ensure_dir(reports_dir)
    return reports_dir


def get_plots_dir(experiment_name: str) -> Path:
    """
    Get the path to the plots directory.
    
    Args:
        experiment_name: Experiment name
        
    Returns:
        Path to the plots directory
    """
    plots_dir = get_reports_dir(experiment_name) / "plots"
    ensure_dir(plots_dir)
    return plots_dir


def resolve_path(path: Union[str, Path], relative_to: Optional[Union[str, Path]] = None) -> Path:
    """
    Resolve a path, making it absolute if it's relative.
    
    Args:
        path: Path to resolve
        relative_to: Base path for relative paths
        
    Returns:
        Absolute Path object
    """
    path = Path(path)
    
    if path.is_absolute():
        return path
    
    if relative_to:
        return Path(relative_to) / path
    
    # Relative to project root by default
    return get_project_root() / path