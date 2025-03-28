"""
Command-line interface for the CBAM Classification project.
"""

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
from utils.seed import set_seed

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
):
    """
    Run the training pipeline.

    This command trains a model using the specified configuration.
    """
    # Load configuration using Hydra
    with hydra.initialize_config_module(config_module="configs"):
        cfg = hydra.compose(config_name="config")

    # Override config values if provided via command line
    if experiment_name:
        cfg.paths.experiment_name = experiment_name

    if model_name:
        cfg.model.name = model_name

    if epochs:
        cfg.training.epochs = epochs

    # Configure logging
    logger = configure_logging(cfg)
    logger.info(f"Starting training with model: {cfg.model.name}")

    # Set seed for reproducibility
    set_seed(cfg.data.random_seed, deterministic=cfg.training.deterministic)

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
):
    """
    Run the evaluation pipeline.

    This command evaluates a trained model on a specific dataset split.
    """
    # Load configuration using Hydra
    with hydra.initialize_config_module(config_module="configs"):
        cfg = hydra.compose(config_name="config")

    # Configure logging
    logger = configure_logging(cfg)
    logger.info(f"Starting evaluation on {split} split")

    # Resolve the checkpoint path
    if checkpoint_path is None:
        checkpoint_path = Path(cfg.paths.checkpoint_dir) / "best_model.pth"
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Set seed for reproducibility
    set_seed(cfg.data.random_seed, deterministic=True)

    # Import here to avoid circular imports
    from core.evaluation.evaluate import evaluate_model

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
    )

    logger.info(f"Evaluation completed. Accuracy: {metrics['accuracy']:.4f}")

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
    # Load configuration using Hydra
    with hydra.initialize_config_module(config_module="configs"):
        cfg = hydra.compose(config_name="config")

    # Configure logging
    logger = configure_logging(cfg)
    logger.info(f"Starting hyperparameter tuning with {n_trials} trials")

    # Set seed for reproducibility
    set_seed(cfg.data.random_seed, deterministic=False)  # Non-deterministic for tuning

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
    set_seed(cfg.data.random_seed)

    # Log execution parameters
    log_execution_params(logger, cfg)

    # Run data preparation pipeline
    run_prepare_data(cfg)

    logger.info(
        f"Data preparation completed. Results saved to: {cfg.prepare_data.output_dir}"
    )


if __name__ == "__main__":
    app()
