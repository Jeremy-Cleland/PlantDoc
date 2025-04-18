# generate_report.py stub
"""
Generate HTML reports from training results using Jinja2 templates.
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape

from reports.generate_plots import generate_plots_for_report
from utils.logger import get_logger
from utils.paths import ensure_dir

logger = get_logger(__name__)


def format_time(seconds: float) -> str:
    """
    Format time in seconds to a human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string (e.g. "2h 30m 15s")
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"


def format_percentage(value: float) -> str:
    """
    Format a value as a percentage string.

    Args:
        value: Value to format (0-1)

    Returns:
        Formatted percentage string (e.g. "75.5%")
    """
    return f"{value * 100:.2f}%"


def load_metrics(metrics_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load metrics from a JSON file.

    Args:
        metrics_path: Path to metrics JSON file

    Returns:
        Dictionary of metrics
    """
    try:
        with open(metrics_path) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load metrics from {metrics_path}: {e}")
        return {}


def load_json(json_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load data from a JSON file.

    Args:
        json_path: Path to JSON file

    Returns:
        Dictionary of data
    """
    try:
        with open(json_path) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load data from {json_path}: {e}")
        return {}


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Dictionary of configuration
    """
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return {}


def load_history(history_path: Union[str, Path]) -> Dict[str, List[float]]:
    """
    Load training history from a JSON file.

    Args:
        history_path: Path to history JSON file

    Returns:
        Dictionary of training history
    """
    try:
        with open(history_path) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load history from {history_path}: {e}")
        return {}


def load_class_names(class_names_path: Union[str, Path]) -> List[str]:
    """
    Load class names from a text file.

    Args:
        class_names_path: Path to class names file

    Returns:
        List of class names
    """
    try:
        with open(class_names_path) as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Failed to load class names from {class_names_path}: {e}")
        return []


def get_model_info(model_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get information about a saved model.

    Args:
        model_path: Path to saved model

    Returns:
        Dictionary of model information
    """
    try:
        model_info = {}

        # Load model state dict to get size
        state_dict = torch.load(model_path, map_location="cpu")

        if isinstance(state_dict, dict):
            # Extract metadata if available
            if "epoch" in state_dict:
                model_info["epoch"] = state_dict["epoch"]

            if "model_state_dict" in state_dict:
                # Calculate model size
                param_size = 0
                num_params = 0
                num_layers = 0

                for param_name, param in state_dict["model_state_dict"].items():
                    param_size += param.nelement() * param.element_size()
                    num_params += param.nelement()
                    # Count layers (approximate based on unique param names)
                    if param_name.count(".") == 1:  # Just counting direct layers
                        num_layers += 1

                model_info["size_mb"] = param_size / (1024**2)
                model_info["num_parameters"] = num_params
                model_info["num_layers"] = num_layers

        return model_info

    except Exception as e:
        logger.error(f"Failed to get model info from {model_path}: {e}")
        return {}


def render_template(
    template_name: str,
    output_path: Union[str, Path],
    context: Dict[str, Any],
    templates_dir: Optional[Union[str, Path]] = None,
) -> bool:
    """
    Render a Jinja2 template to an output file.

    Args:
        template_name: Name of the template file
        output_path: Path to save the rendered output
        context: Dictionary of variables to pass to the template
        templates_dir: Directory containing templates

    Returns:
        True if successful, False otherwise
    """
    try:
        # Default templates directory is in the reports module
        if templates_dir is None:
            templates_dir = Path(__file__).parent / "templates"

        # Set up Jinja2 environment
        env = Environment(
            loader=FileSystemLoader(templates_dir),
            autoescape=select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Add custom filters
        env.filters["format_time"] = format_time
        env.filters["format_percentage"] = format_percentage

        # Get template
        template = env.get_template(template_name)

        # Render template
        output = template.render(**context)

        # Create output directory if needed
        output_path = Path(output_path)
        ensure_dir(output_path.parent)

        # Write output
        with open(output_path, "w") as f:
            f.write(output)

        logger.info(f"Rendered template {template_name} to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to render template {template_name}: {e}")
        return False


def copy_assets(assets_dir: Union[str, Path], output_dir: Union[str, Path]) -> bool:
    """
    Copy assets (CSS, JS, images) to the output directory.

    Args:
        assets_dir: Directory containing assets
        output_dir: Output directory

    Returns:
        True if successful, False otherwise
    """
    try:
        assets_dir = Path(assets_dir)
        output_dir = Path(output_dir)

        if not assets_dir.exists():
            logger.warning(f"Assets directory {assets_dir} does not exist")
            return False

        # Create output directory if needed
        output_assets_dir = output_dir / "assets"
        ensure_dir(output_assets_dir)

        # Copy assets
        for asset in assets_dir.glob("**/*"):
            if asset.is_file():
                relative_path = asset.relative_to(assets_dir)
                output_path = output_assets_dir / relative_path
                ensure_dir(output_path.parent)
                shutil.copy2(asset, output_path)

        logger.info(f"Copied assets from {assets_dir} to {output_assets_dir}")
        return True

    except Exception as e:
        logger.error(f"Failed to copy assets: {e}")
        return False


def get_class_performance(
    metrics: Dict[str, Any], class_names: List[str]
) -> List[Dict[str, Any]]:
    """
    Extract class-wise performance metrics.

    Args:
        metrics: Dictionary of metrics
        class_names: List of class names

    Returns:
        List of class performance dictionaries
    """
    class_performance = []

    for class_name in class_names:
        class_key = class_name.replace(" ", "_")

        # Extract metrics for this class
        class_data = {
            "name": class_name,
            "precision": metrics.get(f"class_{class_key}_precision", 0.0),
            "recall": metrics.get(f"class_{class_key}_recall", 0.0),
            "f1": metrics.get(f"class_{class_key}_f1", 0.0),
            "accuracy": metrics.get(f"class_{class_key}_accuracy", 0.0),
        }

        class_performance.append(class_data)

    # Sort by F1 score (descending)
    class_performance.sort(key=lambda x: x["f1"], reverse=True)

    return class_performance


def generate_report(
    experiment_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    template_name: str = "training_report.html",
    generate_plots: bool = True,
) -> None:
    """
    Generate a training report for an experiment.

    Args:
        experiment_dir: Path to experiment directory
        output_dir: Directory to save the report (default: experiment_dir/reports)
        template_name: Name of the template file
        generate_plots: Whether to generate plots for the report
    """
    experiment_dir = Path(experiment_dir)

    if not experiment_dir.is_absolute():
        # Use the current working directory
        experiment_dir = Path.cwd() / experiment_dir

    logger.info(f"Using experiment directory: {experiment_dir}")

    if not experiment_dir.exists():
        logger.error(f"Experiment directory {experiment_dir} does not exist")
        return

    if output_dir is None:
        output_dir = experiment_dir / "reports"

    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    logger.info(f"Generating report for experiment in {experiment_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Generate plots if requested
    plots_dir = output_dir / "plots"
    if generate_plots:
        logger.info("Generating plots for report")
        generate_plots_for_report(experiment_dir, plots_dir)

    # Get the list of available plot files for conditional rendering
    available_plots = []
    if plots_dir.exists():
        for plot_file in plots_dir.glob("*.png"):
            available_plots.append(str(plot_file.relative_to(output_dir)))

    # Check for SHAP visualizations
    shap_plots = []
    shap_dir = plots_dir / "shap"
    if shap_dir.exists():
        for shap_file in shap_dir.glob("*.png"):
            shap_plots.append(str(shap_file.relative_to(output_dir)))
        logger.info(f"Found {len(shap_plots)} SHAP visualizations")

    # Check for attention visualizations
    attention_plots = []
    attention_dir = experiment_dir / "attention_visualizations"
    if attention_dir.exists():
        for attention_file in attention_dir.glob("*.png"):
            # Copy attention files to plots directory
            attention_output_dir = plots_dir / "attention"
            ensure_dir(attention_output_dir)
            import shutil

            shutil.copy2(attention_file, attention_output_dir / attention_file.name)
            attention_plots.append(
                str(
                    (attention_output_dir / attention_file.name).relative_to(output_dir)
                )
            )
        logger.info(f"Found {len(attention_plots)} attention visualizations")

    # Check for augmentation examples
    augmentation_plots = []
    augmentation_dir = experiment_dir / "augmentation_examples"
    if augmentation_dir.exists():
        for aug_file in augmentation_dir.glob("*.png"):
            # Copy augmentation files to plots directory
            aug_output_dir = plots_dir / "augmentation"
            ensure_dir(aug_output_dir)
            import shutil

            shutil.copy2(aug_file, aug_output_dir / aug_file.name)
            augmentation_plots.append(
                str((aug_output_dir / aug_file.name).relative_to(output_dir))
            )
        logger.info(f"Found {len(augmentation_plots)} augmentation examples")

    # Load data
    metrics_path = experiment_dir / "metrics" / "metrics.json"
    training_metrics_path = experiment_dir / "metrics" / "training_metrics.json"
    training_params_path = experiment_dir / "metrics" / "training_params.json"
    config_path = experiment_dir / "config.yaml"
    history_path = experiment_dir / "metrics" / "history.json"
    class_names_path = experiment_dir / "class_names.txt"
    model_path = experiment_dir / "checkpoints" / "best_model.pth"

    metrics = load_metrics(metrics_path)
    training_metrics = load_json(training_metrics_path)
    training_params = load_json(training_params_path)
    config = load_config(config_path)
    history = load_history(history_path)
    class_names = load_class_names(class_names_path)
    model_info = get_model_info(model_path)

    # Check for training_metrics.json and merge with existing metrics if found
    if training_metrics_path.exists():
        logger.info(f"Found training metrics at {training_metrics_path}")
        training_metrics = load_json(training_metrics_path)

        # Merge detailed training metrics with main metrics
        # Training metrics may have more detailed information or per-epoch data
        metrics.update(
            {
                k: v
                for k, v in training_metrics.items()
                if k not in metrics or k.startswith("class_") or "detailed_" in k
            }
        )

    # Try to get information about the model checkpoint
    model_size = None
    checkpoint_size = None
    if model_path.exists():
        try:
            checkpoint_size = model_path.stat().st_size / (1024 * 1024)  # Convert to MB
        except Exception:
            pass

        # If model size is not available from the checkpoint, we can try to reconstruct it
        try:
            import torch

            model = torch.load(model_path, map_location=torch.device("cpu"))
            model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
        except Exception:
            pass

    # Create context for template
    context = {
        "title": f"Training Report: {experiment_dir.name}",
        "experiment_name": experiment_dir.name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "training_metrics": training_metrics,
        "config": config,
        "history": history,
        "class_names": class_names,
        "model_info": model_info,
        "class_performance": get_class_performance(metrics, class_names),
        "plots_dir": "plots",  # Relative path from output_dir
        "available_plots": available_plots,  # List of available plot files
        "shap_plots": shap_plots,  # List of SHAP visualizations
        "attention_plots": attention_plots,  # List of attention visualizations
        "augmentation_plots": augmentation_plots,  # List of augmentation examples
        "training_params": training_params,
        "model_size": model_size,
        "checkpoint_size": checkpoint_size,
    }

    # Add history data if available
    if history:
        context["history"] = history

    # Add model name from config
    if config and "model" in config and "name" in config["model"]:
        context["model_name"] = config["model"]["name"]
    else:
        context["model_name"] = "Unknown"

    # Add performance metrics
    for key in ["accuracy", "precision", "recall", "f1"]:
        if key in metrics:
            context[key] = metrics[key]

    # Add training time
    if "total_time" in metrics:
        context["total_time"] = metrics["total_time"]
        context["total_time_formatted"] = format_time(metrics["total_time"])

    # Render template
    output_path = output_dir / template_name
    templates_dir = Path(__file__).parent / "templates"

    render_template(
        template_name=template_name,
        output_path=output_path,
        context=context,
        templates_dir=templates_dir,
    )

    # Copy assets
    assets_dir = templates_dir / "assets"
    copy_assets(assets_dir, output_dir)

    logger.info(f"Report generated: {output_path}")


def main():
    """
    Main entry point for report generation.
    """
    parser = argparse.ArgumentParser(description="Generate training report")
    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        required=True,
        help="Name of the experiment or path to experiment directory",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output directory for the report"
    )
    parser.add_argument(
        "--template",
        "-t",
        type=str,
        default="training_report.html",
        help="Name of the template file",
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip generating plots for the report"
    )

    args = parser.parse_args()

    # Generate report
    generate_report(
        experiment_dir=args.experiment,
        output_dir=args.output,
        template_name=args.template,
        generate_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
