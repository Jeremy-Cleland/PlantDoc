# optuna_runner.py stub
# core/tuning/optuna_runner.py
import time
from pathlib import Path

import optuna
from omegaconf import DictConfig, OmegaConf

from core.data import PlantDiseaseDataModule
from core.models import get_model_class
from core.training import train_model
from utils.logger import get_logger
from utils.seed import set_seed

# Import suggestion functions
from .search_space import (  # import others...
    suggest_augmentation_params,
    suggest_model_params,
    suggest_optimizer_params,
)

logger = get_logger(__name__)

# Global variable to hold the base config, set by tune_model
BASE_CFG = None


def objective(trial: optuna.trial.Trial) -> float:
    """Optuna objective function."""
    global BASE_CFG
    if BASE_CFG is None:
        raise ValueError("Base config not set. Call tune_model first.")

    # Start with a copy of the base config
    cfg = OmegaConf.structured(OmegaConf.to_yaml(BASE_CFG))  # Deep copy

    # ---- Apply Suggested Parameters ----
    try:
        cfg = suggest_optimizer_params(trial, cfg)
        cfg = suggest_model_params(trial, cfg)
        cfg = suggest_augmentation_params(trial, cfg)
        # Add calls to other suggestion functions (scheduler, loss, etc.)
        # Example: suggest loader params
        # cfg.loader.batch_size = trial.suggest_categorical("loader.batch_size", [32, 64, 128])

        # --- Setup Trial Directory ---
        # Create a unique directory for this trial based on Optuna's trial number
        trial_num = trial.number
        # Assume base experiment dir is set in cfg.paths.experiment_dir
        # This might come from hydra sweep dir in the main tune command
        base_exp_dir = Path(cfg.paths.experiment_dir)
        trial_dir = base_exp_dir / f"trial_{trial_num:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        cfg.paths.experiment_dir = str(
            trial_dir
        )  # Update config with trial-specific path
        # Potentially update other paths based on this trial_dir if needed
        cfg.paths.checkpoint_dir = str(trial_dir / "checkpoints")
        cfg.paths.reports_dir = str(trial_dir / "reports")
        cfg.paths.logs_dir = str(trial_dir / "logs")

        # Set seed for reproducibility within the trial (can use trial number)
        set_seed(cfg.seed + trial_num)

        logger.info(f"--- Starting Optuna Trial {trial_num} ---")
        logger.info(f"Trial directory: {trial_dir}")
        # Log suggested parameters
        logger.info(f"Suggested params: {trial.params}")

        # ---- Run Training ----
        # 1. DataModule
        datamodule = PlantDiseaseDataModule(cfg=cfg)
        datamodule.prepare_data()
        datamodule.setup(stage="fit")  # Need class info before model init

        # Ensure num_classes is set in config if not already present
        if "num_classes" not in cfg.model:
            cfg.model.num_classes = datamodule.get_num_classes()

        # 2. Model
        ModelClass = get_model_class(cfg.model.name)
        model = ModelClass(**cfg.model)  # Pass model config dict

        # 3. Training
        # Use the train_model convenience function
        # It will create the Trainer internally
        results = train_model(
            model=model,
            train_loader=datamodule.train_dataloader(),
            val_loader=datamodule.val_dataloader(),
            cfg=cfg,  # Pass the modified config for this trial
            experiment_dir=trial_dir,  # Pass trial-specific dir
        )

        # ---- Get Metric to Optimize ----
        # Fetch the metric defined in the main config (e.g., 'val_accuracy' or 'val_loss')
        monitor_metric = cfg.training.get(
            "monitor_metric", "val_loss"
        )  # Default to val_loss
        metric_value = results.get("best_" + monitor_metric, None)

        if metric_value is None:
            # Try fetching from history if best_ isn't populated correctly
            history = results.get("history", {})
            if monitor_metric in history and history[monitor_metric]:
                monitor_mode = cfg.training.get("monitor_mode", "min")
                if monitor_mode == "min":
                    metric_value = min(history[monitor_metric])
                else:
                    metric_value = max(history[monitor_metric])
            else:
                logger.warning(
                    f"Monitor metric '{monitor_metric}' not found in results or history for trial {trial_num}. Returning inf/neginf."
                )
                # Return a bad value to penalize this trial
                return (
                    float("inf")
                    if cfg.training.get("monitor_mode", "min") == "min"
                    else float("-inf")
                )

        logger.info(
            f"--- Optuna Trial {trial_num} Finished --- Metric ({monitor_metric}): {metric_value:.6f}"
        )
        return metric_value

    except optuna.exceptions.TrialPruned as e:
        logger.info(f"Trial {trial_num} pruned.")
        raise e
    except Exception as e:
        logger.error(f"Error in Optuna trial {trial_num}: {e}", exc_info=True)
        # Return a bad value to indicate failure
        return (
            float("inf")
            if cfg.training.get("monitor_mode", "min") == "min"
            else float("-inf")
        )


# This function will be called by cli/main.py:tune
def tune_model(cfg: DictConfig):
    """Sets up and runs the Optuna study."""
    global BASE_CFG
    BASE_CFG = cfg  # Store the base config passed from Hydra

    # --- Optuna Study Setup ---
    study_name = cfg.get("optuna", {}).get("study_name", f"{cfg.model.name}-tuning")
    n_trials = cfg.get("optuna", {}).get("n_trials", 50)
    direction = cfg.get("optuna", {}).get(
        "direction", "maximize" if "acc" in cfg.training.monitor_metric else "minimize"
    )
    storage_name = cfg.get("optuna", {}).get(
        "storage", None
    )  # e.g., "sqlite:///optuna_studies.db"
    pruner_config = cfg.get("optuna", {}).get(
        "pruner", {"type": "median", "n_warmup_steps": 5}
    )

    # Setup Pruner
    pruner = None
    if pruner_config.get("type") == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=pruner_config.get("n_startup_trials", 5),
            n_warmup_steps=pruner_config.get("n_warmup_steps", 5),
            interval_steps=pruner_config.get("interval_steps", 1),
        )
    elif pruner_config.get("type") == "hyperband":
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=pruner_config.get("min_resource", 1),
            max_resource=pruner_config.get(
                "max_resource", cfg.training.epochs
            ),  # Use epochs as resource
            reduction_factor=pruner_config.get("reduction_factor", 3),
        )
    # Add other pruners if needed

    logger.info(f"Starting Optuna study '{study_name}' for {n_trials} trials.")
    logger.info(f"Optimization direction: {direction}")
    if storage_name:
        logger.info(f"Using storage: {storage_name}")
    if pruner:
        logger.info(f"Using pruner: {type(pruner).__name__}")

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=storage_name,
        pruner=pruner,
        load_if_exists=True,  # Allow resuming studies
    )

    # --- Run Optimization ---
    start_time = time.time()
    study.optimize(
        objective, n_trials=n_trials, timeout=cfg.get("optuna", {}).get("timeout", None)
    )
    duration = time.time() - start_time

    # --- Log Results ---
    logger.info(f"Optuna study finished in {duration:.2f} seconds.")
    logger.info(f"Number of finished trials: {len(study.trials)}")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"  Value ({cfg.training.monitor_metric}): {study.best_value:.6f}")
    logger.info("  Params: ")
    for key, value in study.best_params.items():
        logger.info(f"    {key}: {value}")

    # Save best params to a file in the main experiment dir (where tune was launched)
    best_params_path = Path(cfg.paths.experiment_dir) / "best_hyperparameters.yaml"
    try:
        OmegaConf.save(config=study.best_params, f=best_params_path)
        logger.info(f"Best hyperparameters saved to {best_params_path}")
    except Exception as e:
        logger.error(f"Could not save best hyperparameters: {e}")

    # Optional: Further analysis/plots using optuna.visualization
    if cfg.get("optuna", {}).get("create_plots", True):
        try:
            if optuna.visualization.is_available():
                viz_dir = Path(cfg.paths.experiment_dir) / "optuna_plots"
                viz_dir.mkdir(exist_ok=True)
                optuna.visualization.plot_optimization_history(study).write_html(
                    viz_dir / "optimization_history.html"
                )
                optuna.visualization.plot_param_importances(study).write_html(
                    viz_dir / "param_importances.html"
                )
                # Add more plots like plot_slice, plot_contour if needed
                logger.info(f"Optuna visualization plots saved to {viz_dir}")
            else:
                logger.warning(
                    "Optuna visualization library not available. Skipping plots."
                )
        except Exception as e:
            logger.error(f"Failed to generate Optuna plots: {e}")

    return study  # Return study object if needed
