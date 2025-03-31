"""
Command-line interface for the CBAM Classification project.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import typer
from omegaconf import DictConfig, OmegaConf

from core.data.datamodule import PlantDiseaseDataModule
from core.data.prepare_data import run_prepare_data
from core.models.registry import get_model_class, get_model_param_schema, list_models
from core.training.train import train_model
from reports.generate_plots import generate_plots_for_report
from reports.generate_report import generate_report
from utils.config_utils import load_config
from utils.experiment_registry import (
    get_experiment_name,
    get_next_version,
    register_experiment,
)
from utils.logger import configure_logging, log_execution_params
from utils.mps_utils import set_manual_seed

app = typer.Typer(help="CBAM Classification CLI")


@app.command()
def train(
    config_path: str = typer.Option(
        "configs/config.yaml", "--config", "-c", help="Path to configuration file"
    ),
    experiment_name: str = typer.Option(
        None, "--experiment", "-e", help="Experiment name (overrides config value)"
    ),
    model_name: str = typer.Option(
        None, "--model", "-m", help="Model name (overrides config value)"
    ),
    epochs: Optional[int] = typer.Option(
        None, "--epochs", help="Number of epochs (overrides config value)"
    ),
    version: Optional[int] = typer.Option(
        None,
        "--version",
        "-v",
        help="Version number (auto-incremented if not specified)",
    ),
    description: Optional[str] = typer.Option(
        None, "--description", "-d", help="Description of the experiment"
    ),
):
    """
    Run the training pipeline.

    This command trains a model using the specified configuration.
    """
    # Prepare CLI override arguments
    cli_args = []

    # Load configuration using OmegaConf directly
    cfg = load_config(config_path, cli_args)

    # Use model name from command line or from config
    actual_model_name = model_name or cfg.model.name

    # Get version number for the experiment
    if version is None:
        version = get_next_version(actual_model_name)

    # Generate clean versioned experiment name
    if not experiment_name:
        experiment_name = f"{actual_model_name}_v{version}"

    # Override experiment name in config
    cfg["paths"]["experiment_name"] = experiment_name

    # Add other overrides
    if model_name:
        cfg["model"]["name"] = model_name
    if epochs:
        cfg["training"]["epochs"] = epochs

    # Set the command in the config for context-aware directory handling
    cfg["command"] = "train"

    # Set checkpoint directory path
    if not hasattr(cfg.callbacks.model_checkpoint, "dirpath_original"):
        # Save original relative path
        cfg.callbacks.model_checkpoint.dirpath_original = (
            cfg.callbacks.model_checkpoint.dirpath
        )

    # Set absolute path
    cfg.callbacks.model_checkpoint.dirpath = "checkpoints"

    # Configure logging
    logger = configure_logging(cfg)
    logger.info(f"Starting training with model: {cfg.model.name}")
    logger.info(f"Experiment name: {experiment_name}")
    logger.info(f"Experiment version: {version}")

    # Set seed for reproducibility
    set_manual_seed(cfg.data.random_seed, deterministic=cfg.training.deterministic)

    # Log execution parameters
    log_execution_params(logger, cfg)

    # Create data module
    data_module = PlantDiseaseDataModule(cfg)
    data_module.prepare_data()
    data_module.setup()

    # Get model class and initialize model
    model_class = get_model_class(cfg.model.name)

    # Create a copy of the model config without the 'name' field
    model_params = {k: v for k, v in cfg.model.items() if k != "name"}

    # Initialize model with the filtered parameters
    model = model_class(**model_params)

    # Register the experiment in the registry
    register_experiment(
        model_name=actual_model_name,
        experiment_dir=cfg.paths.experiment_dir,
        params=model_params,
        version=version,
        description=description,
    )

    # Train the model
    results = train_model(
        model=model,
        train_loader=data_module.train_dataloader(),
        val_loader=data_module.val_dataloader(),
        cfg=cfg,
    )

    logger.info(
        f"Training completed. Best validation accuracy: {results['best_val_acc']:.4f}"
    )

    # Always generate plots and report regardless of config settings
    try:
        logger.info("Generating plots...")
        generate_plots_for_report(cfg.paths.experiment_dir)
    except Exception as e:
        logger.error(f"Error generating plots: {e}", exc_info=True)

    try:
        logger.info("Generating training report...")
        generate_report(cfg.paths.experiment_dir)
    except Exception as e:
        logger.error(f"Error generating report: {e}", exc_info=True)

    return results


@app.command()
def eval(
    config_path: str = typer.Option(
        "configs/config.yaml", "--config", "-c", help="Path to configuration file"
    ),
    checkpoint_path: str = typer.Option(
        None, "--checkpoint", "-ckpt", help="Path to model checkpoint"
    ),
    split: str = typer.Option(
        "test", "--split", "-s", help="Dataset split to evaluate on (test, val, train)"
    ),
    output_dir: Optional[str] = typer.Option(
        None, "--output-dir", "-o", help="Output directory for evaluation results"
    ),
    interpret: bool = typer.Option(
        False, "--interpret", "-i", help="Generate model interpretation visualizations"
    ),
):
    """
    Run the evaluation pipeline.

    This command evaluates a trained model on a specific dataset split.
    """
    # Load configuration using OmegaConf
    cfg = load_config(config_path)

    # Set the command in the config for context-aware directory handling
    cfg["command"] = "eval"

    # Ensure experiment directory exists
    experiment_dir = Path(cfg.paths.experiment_dir)
    checkpoint_dir = experiment_dir / "checkpoints"

    # Resolve the checkpoint path
    if checkpoint_path is None:
        checkpoint_path = checkpoint_dir / "best_model.pth"
    else:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_absolute():
            # Try relative to checkpoint directory first
            if (checkpoint_dir / checkpoint_path).exists():
                checkpoint_path = checkpoint_dir / checkpoint_path
            # If not found, try relative to current working directory
            elif not checkpoint_path.exists():
                checkpoint_path = Path.cwd() / checkpoint_path

    # Configure logging
    logger = configure_logging(cfg)
    logger.info(f"Starting evaluation on {split} split")
    logger.info(f"Using checkpoint: {checkpoint_path}")

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Set up output directory for evaluation results
    if output_dir is None:
        eval_output_dir = experiment_dir / f"evaluation_{split}"
    else:
        eval_output_dir = Path(output_dir)

    # Create output directory
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Evaluation results will be saved to: {eval_output_dir}")

    # Set seed for reproducibility
    set_manual_seed(cfg.data.random_seed, deterministic=True)

    # Import the evaluation module
    from core.evaluation.evaluate import evaluate_model
    from core.evaluation.interpretability import evaluate_model_with_gradcam

    # Create data module
    data_module = PlantDiseaseDataModule(cfg)
    data_module.prepare_data()
    data_module.setup(stage="test")

    # Get the appropriate dataloader
    if split == "test":
        dataloader = data_module.test_dataloader()
    elif split == "val":
        dataloader = data_module.val_dataloader()
    elif split == "train":
        dataloader = data_module.train_dataloader()
    else:
        logger.error(
            f"Invalid split: {split}. Must be one of 'test', 'val', or 'train'."
        )
        raise ValueError(f"Invalid split: {split}")

    # Get model class and initialize model
    model_class = get_model_class(cfg.model.name)
    model = model_class(**cfg.model)

    # Evaluate the model
    metrics = evaluate_model(
        model=model,
        dataloader=dataloader,
        checkpoint_path=checkpoint_path,
        cfg=cfg,
        output_dir=eval_output_dir,
        device=None,  # Auto-detect device
    )

    # Print evaluation results
    logger.info(f"Evaluation completed. Results:")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1']:.4f}")

    # Generate model interpretations if requested
    if interpret:
        logger.info("Generating model interpretation visualizations...")
        class_names = data_module.get_class_names()

        interpretation_output_dir = eval_output_dir / "interpretations"
        interpretation_metrics = evaluate_model_with_gradcam(
            model=model,
            data_loader=dataloader,
            num_samples=10,  # Number of samples to visualize
            output_dir=interpretation_output_dir,
            class_names=class_names,
        )

        logger.info(
            f"Generated {interpretation_metrics['num_visualized_samples']} visualization samples"
        )
        logger.info(
            f"Visualizations saved to: {interpretation_metrics['visualization_dir']}"
        )

    return metrics


@app.command()
def tune(
    config_path: str = typer.Option(
        "configs/config.yaml", "--config", "-c", help="Path to configuration file"
    ),
    n_trials: int = typer.Option(100, "--trials", "-t", help="Number of Optuna trials"),
    study_name: str = typer.Option(
        "cbam_tuning", "--study", "-s", help="Optuna study name"
    ),
):
    """
    Run hyperparameter tuning using Optuna.

    This command tunes hyperparameters for a model using Optuna.
    """
    # Load configuration
    cfg = load_config(config_path)

    # Configure logging
    logger = configure_logging(cfg)
    logger.info(f"Starting hyperparameter tuning with {n_trials} trials")

    # Set seed for reproducibility
    set_manual_seed(
        cfg.data.random_seed, deterministic=False
    )  # Non-deterministic for tuning

    # Import the tuning module
    from core.tuning.optuna_runner import run_optuna_study
    from core.tuning.search_space import get_search_space

    # Get search space based on model
    search_space = get_search_space(cfg.model.name)

    # Run Optuna study
    best_params, study = run_optuna_study(
        cfg=cfg,
        search_space=search_space,
        n_trials=n_trials,
        study_name=study_name,
    )

    logger.info(f"Tuning completed. Best parameters: {best_params}")
    logger.info(f"Best validation score: {study.best_value:.4f}")

    # Save best parameters
    output_file = Path(cfg.paths.experiment_dir) / "best_params.yaml"
    with open(output_file, "w") as f:
        OmegaConf.save(config=OmegaConf.create(best_params), f=f)

    logger.info(f"Best parameters saved to {output_file}")

    return best_params


@app.command()
def report(
    experiment: str = typer.Option(
        ...,
        "--experiment",
        "-e",
        help="Name of the experiment or path to experiment directory",
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output directory for the report"
    ),
    no_plots: bool = typer.Option(
        False, "--no-plots", help="Skip generating plots for the report"
    ),
):
    """
    Generate a training report for an experiment.

    This command generates an HTML report with visualizations from training results.
    """
    # Resolve experiment directory
    experiment_dir = Path(experiment)
    if not experiment_dir.is_absolute():
        # Try to find in outputs directory
        output_parent = Path(__file__).parents[1] / "outputs"
        experiment_dir = output_parent / experiment_dir

    # Check if experiment directory exists
    if not experiment_dir.exists():
        typer.echo(f"Error: Experiment directory not found: {experiment_dir}")
        raise typer.Exit(code=1)

    # Generate report
    generate_report(
        experiment_dir=experiment_dir,
        output_dir=output,
        generate_plots=not no_plots,
    )

    typer.echo(f"Report generated for experiment: {experiment_dir}")


@app.command()
def plots(
    experiment: str = typer.Option(
        ...,
        "--experiment",
        "-e",
        help="Name of the experiment or path to experiment directory",
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output directory for the plots"
    ),
):
    """
    Generate plots from training results.

    This command generates visualizations from training metrics.
    """
    # Resolve experiment directory
    experiment_dir = Path(experiment)
    if not experiment_dir.is_absolute():
        # Try to find in outputs directory
        output_parent = Path(__file__).parents[1] / "outputs"
        experiment_dir = output_parent / experiment_dir

    # Check if experiment directory exists
    if not experiment_dir.exists():
        typer.echo(f"Error: Experiment directory not found: {experiment_dir}")
        raise typer.Exit(code=1)

    # Generate plots
    generate_plots_for_report(
        experiment_dir=experiment_dir,
        output_dir=output,
    )

    typer.echo(f"Plots generated for experiment: {experiment_dir}")


@app.command()
def prepare(
    config_path: str = typer.Option(
        "configs/config.yaml", "--config", "-c", help="Path to configuration file"
    ),
    raw_dir: Optional[str] = typer.Option(
        None,
        "--raw-dir",
        "-r",
        help="Path to raw data directory (overrides config value)",
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for preparation results (overrides config value)",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Run validation without making changes"
    ),
):
    """
    Run the data preparation pipeline.

    This command validates, analyzes, and prepares the dataset.
    """
    # Prepare CLI override arguments
    cli_args = []
    if raw_dir:
        cli_args.append(f"paths.raw_dir={raw_dir}")
    if output_dir:
        cli_args.append(f"prepare_data.output_dir={output_dir}")
    if dry_run:
        cli_args.append(f"prepare_data.dry_run=true")

    # Load configuration
    cfg = load_config(config_path, cli_args)

    # Set the command in the config for context-aware directory handling
    cfg["command"] = "prepare"

    # Resolve paths for preparation
    from pathlib import Path

    # Ensure raw_dir exists and is absolute
    raw_dir_path = Path(cfg.paths.raw_dir)
    if not raw_dir_path.is_absolute():
        raw_dir_path = Path.cwd() / raw_dir_path
    cfg.paths.raw_dir = str(raw_dir_path)

    # Ensure output_dir is absolute
    output_dir_path = Path(cfg.prepare_data.output_dir)
    if not output_dir_path.is_absolute():
        output_dir_path = Path.cwd() / output_dir_path
    cfg.prepare_data.output_dir = str(output_dir_path)

    # Create output directory if it doesn't exist
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logger = configure_logging(cfg)
    logger.info(f"Starting data preparation for raw data in: {cfg.paths.raw_dir}")
    logger.info(f"Results will be saved to: {cfg.prepare_data.output_dir}")

    # Set seed for reproducibility
    set_manual_seed(cfg.data.random_seed)

    # Log execution parameters
    log_execution_params(logger, cfg)

    # Run data preparation pipeline
    run_prepare_data(cfg)

    logger.info(
        f"Data preparation completed. Results saved to: {cfg.prepare_data.output_dir}"
    )


@app.command()
def models(
    list_all: bool = typer.Option(
        False, "--list", "-l", help="List all available models"
    ),
    model_name: Optional[str] = typer.Option(
        None, "--model", "-m", help="Model name to get details for"
    ),
    format: str = typer.Option(
        "human", "--format", "-f", help="Output format (human, json, yaml)"
    ),
):
    """
    List available models and their parameters.

    This command shows information about registered models and their parameters.
    """
    if list_all:
        models_dict = list_models()
        typer.echo(f"Available models ({len(models_dict)}):")

        for name, model_info in models_dict.items():
            model_class = model_info["class"]
            metadata = model_info["metadata"]

            typer.echo(f"\n{name}:")
            typer.echo(f"  Class: {model_class.__name__}")

            if metadata:
                typer.echo("  Parameters:")
                for param, param_info in metadata.items():
                    default = param_info.get("default", "None")
                    description = param_info.get("description", "")
                    typer.echo(f"    {param}: {description} (default: {default})")

    elif model_name:
        try:
            # Get model schema
            schema = get_model_param_schema(model_name)

            if format == "human":
                typer.echo(f"Model: {model_name}")
                typer.echo("Parameters:")

                for param, param_info in schema.items():
                    default = param_info.get("default", "None")
                    param_type = param_info.get("type", "any")
                    description = param_info.get("description", "")
                    required = param_info.get("required", False)

                    # Format additional constraints
                    constraints = []
                    if "range" in param_info:
                        constraints.append(f"range: {param_info['range']}")
                    if "choices" in param_info:
                        constraints.append(f"choices: {param_info['choices']}")

                    constraints_str = (
                        f" ({', '.join(constraints)})" if constraints else ""
                    )
                    required_str = " (required)" if required else ""

                    typer.echo(f"  {param}: {description}")
                    typer.echo(f"    Type: {param_type}{required_str}{constraints_str}")
                    typer.echo(f"    Default: {default}")

            elif format == "json":
                import json

                typer.echo(json.dumps(schema, indent=2))

            elif format == "yaml":
                import yaml

                typer.echo(yaml.dump(schema, default_flow_style=False))

            else:
                typer.echo(
                    f"Invalid format: {format}. Supported formats: human, json, yaml"
                )

        except ValueError as e:
            typer.echo(f"Error: {e}")
            available_models = list(list_models().keys())
            typer.echo(f"Available models: {', '.join(available_models)}")

    else:
        # Just list model names if no option provided
        available_models = list(list_models().keys())
        typer.echo(f"Available models: {', '.join(available_models)}")
        typer.echo(
            "\nUse --list for detailed information or --model NAME for specific model details."
        )


# Define options for attention command
ATTENTION_MODEL_OPTION = typer.Option(
    "cbam_only_resnet18", "--model", "-m", help="Model name to visualize"
)
ATTENTION_CONFIG_OPTION = typer.Option(
    "configs/config.yaml", "--config", "-c", help="Path to configuration file"
)
ATTENTION_CHECKPOINT_OPTION = typer.Option(
    None, "--checkpoint", "-ckpt", help="Path to model checkpoint"
)
ATTENTION_IMAGE_OPTION = typer.Option(..., "--image", "-i", help="Path to input image")
ATTENTION_OUTPUT_OPTION = typer.Option(
    "outputs/attention_viz",
    "--output",
    "-o",
    help="Output directory for visualization",
)
ATTENTION_LAYERS_OPTION = typer.Option(
    None,
    "--layers",
    "-l",
    help="Specific layers to visualize (e.g., layer1,layer2)",
)


@app.command()
def attention(
    model_name: str = ATTENTION_MODEL_OPTION,
    config_path: str = ATTENTION_CONFIG_OPTION,
    checkpoint_path: Optional[str] = ATTENTION_CHECKPOINT_OPTION,
    image_path: str = ATTENTION_IMAGE_OPTION,
    output_dir: str = ATTENTION_OUTPUT_OPTION,
    layers: Optional[str] = ATTENTION_LAYERS_OPTION,
):
    """
    Visualize attention maps for a model.

    This command generates visualizations of CBAM attention maps for a given input image.
    """
    from cli.attention_viz_command import visualize

    # Import inside function to avoid circular imports
    visualize(
        model_name=model_name,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        image_path=image_path,
        output_dir=output_dir,
        layers=layers,
    )


if __name__ == "__main__":
    app()
