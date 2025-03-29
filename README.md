# PlantDoc: Plant Disease Classification with CBAM-Augmented ResNet18

This repository contains a complete implementation of a plant disease classification system using a CBAM (Convolutional Block Attention Module) augmented ResNet18 architecture.

## Features

- Data preprocessing and augmentation pipeline using Albumentations
- CBAM-enhanced ResNet18 model implementation
- Training pipeline with Hydra configuration
- Evaluation metrics and model interpretability with GradCAM
- Command-line interface for all operations

## Project Structure

```
plantdoc/
│
├── cli/                  # Command-line interface
├── configs/              # Configuration files
├── core/                 # Core modules
│   ├── data/             # Data handling
│   ├── evaluation/       # Model evaluation and interpretability
│   ├── models/           # Model architectures
│   └── training/         # Training utilities
├── reports/              # Reporting utilities
└── utils/                # Utility functions
```

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- CUDA-capable GPU (recommended)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/plantdoc.git
   cd plantdoc
   ```

2. Install the requirements:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Data Preparation

Prepare your data for training and evaluation:

```bash
python -m cli.main prepare --config configs/config.yaml
```

By default, this will:
1. Create train/val/test splits in your data directory
2. Generate class mappings
3. Validate image integrity

### Training

Train the model using:

```bash
python -m cli.main train --config configs/config.yaml
```

To override configuration parameters:

```bash
python -m cli.main train --config configs/config.yaml --model cbam_only_resnet18 --epochs 50
```

### Evaluation

Evaluate a trained model on the test set:

```bash
python -m cli.main eval --config configs/config.yaml --checkpoint outputs/experiment_name/checkpoints/best_model.pth
```

For evaluation with model interpretability using GradCAM:

```bash
python -m cli.main eval --config configs/config.yaml --interpret
```

### Generating Reports

Generate evaluation reports and visualizations:

```bash
python -m cli.main report --experiment experiment_name
```

To generate only plots:

```bash
python -m cli.main plots --experiment experiment_name
```

### Hyperparameter Tuning

Run hyperparameter tuning using Optuna:

```bash
python -m cli.main tune --config configs/config.yaml --trials 100
```

## Configuration

The project uses Hydra for configuration management. The main configuration file is located at `configs/config.yaml`. It includes settings for:

- Data paths and preprocessing
- Model architecture and parameters
- Training hyperparameters
- Evaluation settings
- Reporting configuration

## Model Architecture

The main model architecture is CBAM-ResNet18, which enhances the standard ResNet18 with attention mechanisms:

1. **Channel Attention Module**: Focuses on "what" is important by applying attention across channels
2. **Spatial Attention Module**: Focuses on "where" is important by applying attention across spatial locations

This dual attention mechanism improves the model's ability to focus on relevant features for plant disease classification.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@article{plantdoc2023,
  title={PlantDoc: A Plant Disease Classification System with CBAM-Augmented ResNet},
  author={Your Name},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2023}
}
```

## Acknowledgements

- CBAM paper: [Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)
- ResNet paper: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- Plant disease dataset source: [Link to dataset]




plantdoc/
├── pyproject.toml         # Modern Python packaging config
├── README.md              # Project documentation
├── .gitignore             # Git ignore file
├── plantdoc/              # Main package directory
│   ├── __init__.py        # Package initialization
│   ├── core/              # Core modules
│   │   ├── __init__.py
│   │   ├── data/          # Data processing
│   │   │   ├── __init__.py
│   │   │   ├── datamodule.py
│   │   │   ├── datasets.py
│   │   │   ├── transforms.py
│   │   │   └── preparation/  # Split prepare_data.py
│   │   │       ├── __init__.py
│   │   │       ├── validation.py
│   │   │       ├── analysis.py
│   │   │       ├── visualization.py
│   │   │       └── main.py
│   │   ├── models/        # Model definitions
│   │   │   ├── __init__.py
│   │   │   ├── attention.py
│   │   │   ├── base.py
│   │   │   ├── model_cbam18.py
│   │   │   ├── registry.py
│   │   │   ├── backbones/
│   │   │   └── heads/
│   │   ├── training/      # Training logic
│   │   │   ├── __init__.py
│   │   │   ├── loss.py
│   │   │   ├── optimizers.py
│   │   │   ├── schedulers.py
│   │   │   ├── train.py
│   │   │   └── callbacks/
│   │   └── tuning/        # Hyperparameter tuning
│   │       ├── __init__.py
│   │       ├── optuna_runner.py
│   │       └── search_space.py
│   ├── utils/             # Utility functions
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   ├── metrics.py
│   │   ├── paths.py
│   │   ├── seed.py
│   │   └── visualization.py
│   ├── cli/               # Command-line interface
│   │   ├── __init__.py
│   │   └── main.py
│   └── reports/           # Reporting functionality
│       ├── __init__.py
│       ├── generate_plots.py
│       ├── generate_report.py
│       └── templates/
├── configs/               # Configuration files
│   ├── config.yaml        # Main config
│   ├── model/
│   ├── hydra/
│   └── overrides/
├── tests/                 # Unit and integration tests
│   └── __init__.py
└── examples/              # Example usage
    └── basic_training.py





# Installation Guide

## Setting Up the Development Environment

### Option 1: Install with pip in development mode

```bash
# Clone the repository
git clone https://github.com/yourusername/git
cd plantdoc

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package in development mode
pip install -e .
```

### Option 2: Install with pip in development mode with dev dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/git
cd plantdoc

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package with development dependencies
pip install -e ".[dev]"
```

## Setting Up Pre-commit Hooks

If you installed the development dependencies, you can set up pre-commit hooks:

```bash
pre-commit install
```

## Testing Your Installation

You can verify your installation by running a simple Python script:

```python
from core.models.registry import list_models

# This should print the available models
print(list_models())
```

Or by running the CLI:

```bash
python -m cli.main --help
```

## 1. Model Registr

### Usage Example:

```bash
# List all models with their parameters
python cli/main.py models --list

# Get detailed information about a specific model
python cli/main.py models --model cbam_only_resnet18

# Get parameter schema in JSON or YAML format
python cli/main.py models --model cbam_only_resnet18 --format json
```

## 2. Attention Visualization Tools

I've implemented a complete suite of tools for visualizing CBAM attention maps:

- **Attention Map Extraction**: Enhanced the model and backbone classes to capture and expose attention maps.

- **Visualization Functions**: Created comprehensive visualization tools for channel attention, spatial attention, and overlays on input images.

- **HTML Report Generation**: Added a report generator that creates an interactive HTML report with all visualizations.

- **CLI Command**: Added a new `attention` command to the main CLI for easy visualization.

### Usage Example:

```bash
# Visualize attention maps for a model on a specific image
python cli/main.py attention --model cbam_only_resnet18 --image path/to/image.jpg

# Visualize specific layers
python cli/main.py attention --model cbam_only_resnet18 --image path/to/image.jpg --layers layer1,layer4

# Use a trained checkpoint
python cli/main.py attention --model cbam_only_resnet18 --image path/to/image.jpg --checkpoint path/to/checkpoint.pth
```
