"""
Configuration utilities for loading and managing config without Hydra.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str, cli_args: Optional[List[str]] = None) -> DictConfig:
    """
    Load configuration from YAML file using OmegaConf.

    Args:
        config_path: Path to the YAML configuration file
        cli_args: Optional list of CLI arguments to override config

    Returns:
        DictConfig: Loaded and merged configuration
    """
    # Check if config file exists
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load base config
    config = OmegaConf.load(config_path)

    # Handle CLI overrides if provided, otherwise use sys.argv
    if cli_args:
        cli_conf = OmegaConf.from_dotlist(cli_args)
    else:
        # Skip program name and other non-override arguments
        override_args = [arg for arg in sys.argv if "=" in arg]
        cli_conf = OmegaConf.from_dotlist(override_args)

    # Merge base config with CLI overrides
    config = OmegaConf.merge(config, cli_conf)

    # Handle environment variable substitution
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))

    # Configure paths
    config = _configure_experiment_paths(config)

    return config


def _configure_experiment_paths(config: DictConfig) -> DictConfig:
    """
    Configure experiment paths based on experiment name.

    Args:
        config: Configuration object

    Returns:
        Updated configuration with resolved paths
    """
    # Create experiment directory path
    experiment_name = config.paths.experiment_name
    output_dir = config.paths.output_dir

    # Create experiment directory path
    experiment_dir = Path(output_dir) / experiment_name

    # Add additional path configurations
    config.paths.experiment_dir = str(experiment_dir)
    config.paths.checkpoint_dir = str(experiment_dir / "checkpoints")
    config.paths.log_dir = str(experiment_dir / "logs")
    config.paths.plot_dir = str(experiment_dir / "plots")

    # Create directories
    Path(config.paths.experiment_dir).mkdir(parents=True, exist_ok=True)
    Path(config.paths.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.paths.log_dir).mkdir(parents=True, exist_ok=True)
    Path(config.paths.plot_dir).mkdir(parents=True, exist_ok=True)

    return config
