import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

# Dictionary to keep track of loggers that have been created
_LOGGERS: Dict[str, logging.Logger] = {}
_GLOBAL_FILE_HANDLER = None


def get_logger(
    name: str,
    log_level: Union[str, int] = "INFO",
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None,
    timestamp: bool = False,
    use_colors: bool = True,
) -> logging.Logger:
    """
    Create or retrieve a configured logger instance.

    Args:
        name: Name of the logger (typically __name__)
        log_level: Log level (e.g. INFO, DEBUG)
        log_file: Optional filename (e.g. 'train.log')
        log_dir: Directory to store log file
        timestamp: If True, append timestamp to log_file
        use_colors: Use colored log output for console

    Returns:
        logging.Logger instance
    """
    global _GLOBAL_FILE_HANDLER

    if name in _LOGGERS:
        return _LOGGERS[name]

    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Disable propagation for all loggers except the root logger
    # This is the key fix for double logging
    logger.propagate = False

    # Clear old handlers
    if logger.handlers:
        logger.handlers.clear()

    # Formatter
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(log_format, date_format)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if use_colors and sys.stdout.isatty():
        console_handler.setFormatter(ColoredFormatter(log_format, date_format))
    else:
        console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler - Handle both global and per-logger file handlers
    if log_dir and log_file:
        os.makedirs(log_dir, exist_ok=True)
        if timestamp:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"{Path(log_file).stem}_{timestamp_str}{Path(log_file).suffix}"
        file_path = os.path.join(log_dir, log_file)

        # Use global file handler for main logger and its children
        if name == "plantdoc" or name.startswith("plantdoc."):
            # Create a new file handler if not already present
            if (
                _GLOBAL_FILE_HANDLER is None
                or _GLOBAL_FILE_HANDLER.baseFilename != file_path
            ):
                # Close previous handler if it exists
                if _GLOBAL_FILE_HANDLER is not None:
                    _GLOBAL_FILE_HANDLER.close()

                _GLOBAL_FILE_HANDLER = logging.FileHandler(file_path, mode="a")
                _GLOBAL_FILE_HANDLER.setFormatter(formatter)

            # Ensure the handler is attached (it might have been removed)
            if _GLOBAL_FILE_HANDLER not in logger.handlers:
                logger.addHandler(_GLOBAL_FILE_HANDLER)
                logger.info(f"Logging to file: {file_path}")
        # For non-plantdoc loggers that need their own file handler
        else:
            # Create a new file handler for this logger
            file_handler = logging.FileHandler(file_path, mode="a")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info(f"Logging to file: {file_path}")

    _LOGGERS[name] = logger
    return logger


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[37m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[41m",
        "RESET": "\033[0m",
    }

    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
            )
        return super().format(record)


def configure_logging(cfg) -> logging.Logger:
    """
    Global logging setup using Hydra config.

    Args:
        cfg: DictConfig from Hydra

    Returns:
        Root logger
    """
    global _GLOBAL_FILE_HANDLER
    _GLOBAL_FILE_HANDLER = None

    # Pull settings from config
    command = cfg.get("command", "run")
    log_level = cfg.logging.get("level", "INFO")
    use_colors = cfg.logging.get("use_colors", True)
    log_to_file = cfg.logging.get("log_to_file", True)

    # Use log_dir from paths configuration
    if hasattr(cfg.paths, "log_dir"):
        log_dir = Path(cfg.paths.log_dir)
    else:
        # Fallback if log_dir isn't configured yet
        log_dir = Path("outputs/logs")

    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Create log filename based on command
    log_file = cfg.logging.get("log_file", "command.log")
    if log_file == "command.log":
        # If using the default log file name, use the command name
        log_file = f"{command}.log"

    # Silence Hydra and noisy libraries
    for noisy in ["hydra", "matplotlib", "PIL"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Configure the root logger
    root_root_logger = logging.getLogger()
    root_root_logger.setLevel(
        logging.WARNING
    )  # Only show warnings and above for third-party libs
    # Remove all handlers from the root logger to avoid duplicated logs
    for handler in root_root_logger.handlers[:]:
        root_root_logger.removeHandler(handler)

    # Create main app logger
    if log_to_file:
        root_logger = get_logger(
            "plantdoc",
            log_level=log_level,
            log_dir=log_dir,
            log_file=log_file,
            use_colors=use_colors,
        )
    else:
        root_logger = get_logger(
            "plantdoc",
            log_level=log_level,
            use_colors=use_colors,
        )

    # Set up module loggers
    for module in [
        "data",
        "models",
        "training",
        "evaluation",
        "prediction",
        "cli",
        "utils",
    ]:
        if log_to_file:
            get_logger(
                f"plantdoc.{module}",
                log_level=log_level,
                log_dir=log_dir,
                log_file=log_file,
                use_colors=use_colors,
            )
        else:
            get_logger(f"plantdoc.{module}", log_level=log_level, use_colors=use_colors)

    # Explicitly create a file in the log directory, even if empty, to ensure it exists
    if log_to_file:
        log_path = Path(log_dir) / log_file
        try:
            # Touch the file
            with open(log_path, "a") as f:
                pass  # Just create it if it doesn't exist
            root_logger.info(f"Initialized log file: {log_path}")
        except Exception as e:
            root_logger.error(f"Error initializing log file: {e}")

    root_logger.info(f"Logging initialized. Level: {log_level} | Log dir: {log_dir}")
    return root_logger


def log_execution_params(logger, cfg):
    """
    Log core execution metadata for reproducibility.

    Args:
        logger: Logger instance
        cfg: DictConfig from Hydra
    """
    logger.info("------ Execution Context ------")
    logger.info(f"Command        : {cfg.get('command', 'N/A')}")
    logger.info(f"Model          : {cfg.model.get('name', 'N/A')}")
    logger.info(f"Dataset        : {cfg.data.get('dataset_name', 'N/A')}")
    logger.info(f"Output Dir     : {cfg.get('output_dir', 'N/A')}")
    logger.info(f"Time           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("-------------------------------")
