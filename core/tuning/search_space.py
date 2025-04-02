# search_space.py stub
# core/tuning/search_space.py
import optuna
from omegaconf import DictConfig


def suggest_optimizer_params(trial: optuna.trial.Trial, cfg: DictConfig) -> DictConfig:
    """Suggests optimizer hyperparameters."""
    # Example: Suggest learning rate based on optimizer type
    optimizer_name = cfg.optimizer.name

    if optimizer_name in ["adam", "adamw"]:
        lr = trial.suggest_float("optimizer.lr", 1e-5, 1e-2, log=True)
        beta1 = trial.suggest_float("optimizer.beta1", 0.8, 0.99)
        beta2 = trial.suggest_float("optimizer.beta2", 0.9, 0.9999)
        weight_decay = trial.suggest_float(
            "optimizer.weight_decay", 1e-6, 1e-3, log=True
        )
        # Update config directly (or return a dict to merge)
        cfg.optimizer.lr = lr
        cfg.optimizer.beta1 = beta1
        cfg.optimizer.beta2 = beta2
        cfg.optimizer.weight_decay = weight_decay  # Adjust if using differential LR
    elif optimizer_name == "sgd":
        lr = trial.suggest_float("optimizer.lr", 1e-4, 1e-1, log=True)
        momentum = trial.suggest_float("optimizer.momentum", 0.7, 0.99)
        weight_decay = trial.suggest_float(
            "optimizer.weight_decay", 1e-6, 1e-3, log=True
        )
        cfg.optimizer.lr = lr
        cfg.optimizer.momentum = momentum
        cfg.optimizer.weight_decay = weight_decay

    # Example: Suggest differential LR settings
    cfg.optimizer.differential_lr = trial.suggest_categorical(
        "optimizer.differential_lr", [True, False]
    )
    if cfg.optimizer.differential_lr:
        cfg.optimizer.differential_lr_factor = trial.suggest_float(
            "optimizer.differential_lr_factor", 0.01, 0.5, log=True
        )

    return cfg


def suggest_model_params(trial: optuna.trial.Trial, cfg: DictConfig) -> DictConfig:
    """Suggests model hyperparameters."""
    cfg.model.dropout_rate = trial.suggest_float("model.dropout_rate", 0.1, 0.5)
    # Example: Tune CBAM reduction ratio if desired
    # cfg.model.reduction_ratio = trial.suggest_categorical("model.reduction_ratio", [8, 16, 32])
    # Example: Tune head hidden dim if using MLP/Residual head
    if cfg.model.head_type in ["mlp", "residual"]:
        cfg.model.hidden_dim = trial.suggest_categorical(
            "model.hidden_dim", [128, 256, 512]
        )

    return cfg


def suggest_augmentation_params(
    trial: optuna.trial.Trial, cfg: DictConfig
) -> DictConfig:
    """Suggests augmentation hyperparameters."""
    # Example: Tune cutout probability
    if hasattr(cfg.augmentation.train, "cutout"):
        cfg.augmentation.train.cutout.p = trial.suggest_float(
            "augmentation.train.cutout.p", 0.1, 0.7
        )
    # Example: Tune rotation limit
    if hasattr(cfg.augmentation.train, "random_rotate"):
        cfg.augmentation.train.random_rotate = trial.suggest_int(
            "augmentation.train.random_rotate", 10, 45
        )

    return cfg


# Add more functions as needed for scheduler, loss, data params etc.
