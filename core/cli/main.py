"""
Command-line interface for the CBAM Classification project.
"""

import datetime
import os
import sys
from pathlib import Path
from typing import Optional

import hydra
import typer
from omegaconf import DictConfig, OmegaConf

from core.data.datamodule import PlantDiseaseDataModule
from core.data.prepare_data import run_prepare_data
from core.models.registry import get_model_class
from core.training.train import train_model
from reports.generate_plots import generate_plots_for_report
from reports.generate_report import generate_report
from utils.logger import configure_logging, log_execution_params
from utils.mps_utils import set_manual_seed

app = typer.Typer(help="CBAM Classification CLI")


@app.command()
def train(
    config_path: str = typer.Option(
        "configs/config.yaml", "--config", "-c", help="Path to configuration file"
    ),
    experiment_name: str = typer.Option(
        None, "--experiment", "-e", help="Experiment name"
    ),
    model_name: str = typer.Option(None, "--model", "-m", help="Model name"),
    epochs: Optional[int] = typer.Option(None, "--epochs", help="Number of epochs"),
):
    """Run the training pipeline."""
    # Load the config with Hydra, but set up paths manually
    with hydra.initialize_config_module(config_module="configs"):
        cfg = hydra.compose(config_name="config")

    # Project root is always the current directory
    project_root = Path.cwd()

    # Override experiment name if provided
    if experiment_name:
        cfg.paths.experiment_name = experiment_name

    # Override model name if provided
    if model_name:
        cfg.model.name = model_name

    # Override epochs if provided
    if epochs:
        cfg.training.epochs = epochs

    # Set up all paths programmatically (no interpolation)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Make paths absolute
    cfg.paths.output_dir = str(project_root / cfg.paths.output_dir)
    cfg.paths.raw_dir = str(project_root / cfg.paths.raw_dir)
    cfg.paths.preprocessed_dir = str(project_root / cfg.paths.preprocessed_dir)

    # Set up derived paths
    cfg.paths.experiment_dir = f"{cfg.paths.output_dir}/{cfg.paths.experiment_name}"
    cfg.paths.logs_dir = f"{cfg.paths.experiment_dir}/logs"
    cfg.paths.checkpoints_dir = f"{cfg.paths.experiment_dir}/checkpoints"
    cfg.paths.reports_dir = f"{cfg.paths.experiment_dir}/reports"
    cfg.paths.plots_dir = f"{cfg.paths.reports_dir}/plots"

    # Ensure all directories exist
    for path_name in [
        "output_dir",
        "experiment_dir",
        "logs_dir",
        "checkpoints_dir",
        "reports_dir",
        "plots_dir",
    ]:
        os.makedirs(getattr(cfg.paths, path_name), exist_ok=True)

    # Rest of your training code...
    logger = configure_logging(cfg)
    logger.info(f"Starting training with model: {cfg.model.name}")

    logger.info(f"Project root: {cfg.project_root}")
    logger.info(f"Experiment directory: {cfg.paths.experiment_dir}")

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
    model = model_class(**cfg.model)

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

    # Generate plots and report if enabled
    if cfg.reporting.generate_plots:
        logger.info("Generating plots...")
        generate_plots_for_report(cfg.paths.experiment_dir)

    if cfg.reporting.generate_report:
        logger.info("Generating training report...")
        generate_report(cfg.paths.experiment_dir)

    return results


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
    # Load configuration using Hydra
    with hydra.initialize_config_module(config_module="configs"):
        cfg = hydra.compose(config_name="config")

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
    # Load configuration using Hydra
    with hydra.initialize_config_module(config_module="configs"):
        cfg = hydra.compose(config_name="config")

    # Override config values if provided via command line
    if raw_dir:
        cfg.paths.raw_dir = raw_dir

    if output_dir:
        cfg.prepare_data.output_dir = output_dir

    if dry_run:
        cfg.prepare_data.dry_run = True

    # Configure logging
    logger = configure_logging(cfg)
    logger.info(f"Starting data preparation for raw data in: {cfg.paths.raw_dir}")

    # Set seed for reproducibility
    set_manual_seed(cfg.data.random_seed)

    # Log execution parameters
    log_execution_params(logger, cfg)

    # Run data preparation pipeline
    run_prepare_data(cfg)

    logger.info(
        f"Data preparation completed. Results saved to: {cfg.prepare_data.output_dir}"
    )


if __name__ == "__main__":
    app()
