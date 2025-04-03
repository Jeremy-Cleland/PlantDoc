# PlantDoc: Plant Disease Classification with CBAM-Augmented ResNet18

![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-orange)

This repository contains a complete implementation of a plant disease classification system using a CBAM (Convolutional Block Attention Module) augmented ResNet18 architecture. The system is designed to accurately identify various plant diseases from images, leveraging attention mechanisms to focus on the most relevant features for diagnosis.

## Overview

Plant diseases cause significant crop losses worldwide. Early and accurate detection is crucial for effective management. This project implements a state-of-the-art deep learning approach that combines ResNet18 with attention mechanisms to improve classification accuracy for plant disease diagnosis.

The CBAM architecture enhances the model's ability to focus on relevant disease features by applying:

1. **Channel attention** - Emphasizes important feature channels ("what" to focus on)
2. **Spatial attention** - Highlights important regions in the image ("where" to focus on)

## Features

- **Advanced Model Architecture**: CBAM-enhanced ResNet18 with customizable attention mechanisms
- **Comprehensive Data Pipeline**: Preprocessing, augmentation, and validation using Albumentations
- **Flexible Training System**: Configurable training with callbacks, mixed precision, and transfer learning
- **Extensive Augmentation Options**: RandAugment, CutMix, and other advanced augmentation strategies
- **Model Interpretability**: GradCAM and SHAP visualizations to explain model decisions
- **Attention Visualization**: Tools to visualize and understand attention maps
- **Performance Monitoring**: Confidence calibration, metrics tracking, and comprehensive reporting
- **Hyperparameter Optimization**: Integrated Optuna-based hyperparameter tuning
- **Command-line Interface**: Intuitive CLI for all operations with extensive configuration options
- **Hardware Optimization**: Support for CUDA, MPS (Apple Silicon), and CPU training

## Project Structure

```text
plantdoc/
├── pyproject.toml         # Modern Python packaging config
├── README.md              # Project documentation
├── .gitignore             # Git ignore file
├── cli/                   # Command-line interface
│   └── main.py            # Main CLI entry point
├── configs/               # Configuration files
│   └── config.yaml        # Main configuration file
├── core/                  # Core modules
│   ├── data/              # Data processing
│   │   ├── datamodule.py  # PyTorch data module
│   │   ├── datasets.py    # Dataset implementations
│   │   ├── transforms.py  # Data transformations
│   │   └── prepare_data.py # Data preparation utilities
│   ├── evaluation/        # Model evaluation
│   │   ├── evaluate.py    # Evaluation pipeline
│   │   ├── interpretability.py # GradCAM implementation
│   │   ├── metrics.py     # Evaluation metrics
│   │   └── shap_evaluation.py # SHAP analysis
│   ├── models/            # Model definitions
│   │   ├── attention.py   # CBAM implementation
│   │   ├── base.py        # Base model class
│   │   ├── model_cbam18.py # CBAM-ResNet18 model
│   │   ├── registry.py    # Model registry
│   │   ├── backbones/     # Model backbones
│   │   └── heads/         # Classification heads
│   ├── training/          # Training utilities
│   │   ├── callbacks/     # Training callbacks
│   │   ├── loss.py        # Loss functions
│   │   ├── optimizers.py  # Optimizer configurations
│   │   ├── schedulers.py  # LR scheduler implementations
│   │   └── train.py       # Training loop
│   ├── tuning/            # Hyperparameter tuning
│   │   ├── optuna_runner.py # Optuna integration
│   │   └── search_space.py # Hyperparameter search space
│   └── visualization/     # Visualization tools
│       ├── attention_viz.py # Attention visualization
│       └── visualization.py # General visualizations
├── reports/               # Reporting utilities
│   ├── generate_plots.py  # Plot generation
│   ├── generate_report.py # HTML report generation
│   └── templates/         # Report templates
├── utils/                 # Utility functions
│   ├── config_utils.py    # Configuration utilities
│   ├── logger.py          # Logging setup
│   ├── paths.py           # Path management
│   └── mps_utils.py       # Apple Silicon GPU utilities
├── scripts/               # Utility scripts
└── data/                  # Data directory
    └── raw/               # Raw data storage
```

## Getting Started

### Prerequisites

- Python 3.8+ (3.8, 3.9, 3.10, 3.11 supported)
- PyTorch 2.1+
- CUDA-capable GPU (recommended) or Apple Silicon (MPS support)

### Installation

#### Option 1: Install with pip

```bash
# Clone the repository
git clone https://github.com/yourusername/plantdoc.git
cd plantdoc

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install .
```

#### Option 2: Install in development mode with dev dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/plantdoc.git
cd plantdoc

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package with development dependencies
pip install -e ".[dev]"
```

### Testing Your Installation

Verify your installation by running:

```bash
python -m cli.main --help
```

Or in Python:

```python
from core.models.registry import list_models

# This should print the available models
print(list_models())
```

## Usage

### Data Preparation

Prepare your data for training and evaluation:

```bash
python -m cli.main prepare --config configs/config.yaml
```

This will:

1. Validate image integrity and fix common issues
2. Analyze class distribution and image properties
3. Generate visualizations of the dataset
4. Create train/val/test splits

Options:

```bash
# Specify custom directories
python -m cli.main prepare --raw-dir data/my_dataset --output-dir data/processed

# Run in dry-run mode (no changes)
python -m cli.main prepare --dry-run
```

### Training

Train the model using:

```bash
python -m cli.main train --config configs/config.yaml
```

Customize training:

```bash
# Override configuration parameters
python -m cli.main train --config configs/config.yaml --model cbam_only_resnet18 --epochs 50

# Specify experiment name and version
python -m cli.main train --experiment my_experiment --version 2
```

Training automatically:

1. Creates an experiment directory with versioning
2. Logs metrics and checkpoints
3. Generates visualizations of attention maps
4. Creates a training report with plots

### Evaluation

Evaluate a trained model on the test set:

```bash
python -m cli.main eval --config configs/config.yaml --checkpoint outputs/experiment_name/checkpoints/best_model.pth
```

For evaluation with model interpretability:

```bash
python -m cli.main eval --config configs/config.yaml --interpret
```

Options:

```bash
# Evaluate on a specific split
python -m cli.main eval --split val

# Specify output directory
python -m cli.main eval --output-dir my_evaluation_results
```

### Attention Visualization

Visualize attention maps for a specific image:

```bash
python -m cli.main attention --model cbam_only_resnet18 --image path/to/image.jpg
```

Customize visualization:

```bash
# Visualize specific layers
python -m cli.main attention --model cbam_only_resnet18 --image path/to/image.jpg --layers layer1,layer4

# Use a trained checkpoint
python -m cli.main attention --model cbam_only_resnet18 --image path/to/image.jpg --checkpoint path/to/checkpoint.pth
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

### Model Registry

Explore available models:

```bash
# List all models with their parameters
python -m cli.main models --list

# Get detailed information about a specific model
python -m cli.main models --model cbam_only_resnet18

# Get parameter schema in JSON or YAML format
python -m cli.main models --model cbam_only_resnet18 --format json
```

## Configuration

The project uses a YAML-based configuration system. The main configuration file is located at `configs/config.yaml` and includes settings for:

- **Data**: Dataset paths, class names, and splits
- **Model**: Architecture, attention parameters, and regularization
- **Training**: Epochs, batch size, learning rate, and precision
- **Augmentation**: Data augmentation strategies including RandAugment and CutMix
- **Optimization**: Optimizer, scheduler, and loss function
- **Callbacks**: Early stopping, checkpointing, and visualization
- **Evaluation**: Metrics, interpretability, and reporting
- **Hardware**: Device selection and optimization settings

Configuration can be overridden via command-line arguments or by creating custom config files.

## Model Architecture

The main model architecture is CBAM-ResNet18, which enhances the standard ResNet18 with attention mechanisms:

### CBAM (Convolutional Block Attention Module)

1. **Channel Attention Module**:
   - Applies both average and max pooling across spatial dimensions
   - Processes pooled features through a shared MLP
   - Combines results with element-wise addition
   - Applies sigmoid activation to generate channel attention weights

2. **Spatial Attention Module**:
   - Applies both average and max pooling across channel dimension
   - Concatenates pooled features and processes with a convolution
   - Applies sigmoid activation to generate spatial attention weights

3. **Integration with ResNet**:
   - CBAM modules are inserted after each residual block
   - Can be configured with different reduction ratios and kernel sizes
   - Includes optional stochastic depth for regularization

This dual attention mechanism significantly improves the model's ability to focus on relevant features for plant disease classification, particularly subtle lesions, spots, and discoloration patterns.

## Performance and Results

The CBAM-augmented ResNet18 model achieves superior performance compared to standard ResNet18 for plant disease classification:

- Higher accuracy, especially for visually similar diseases
- Better generalization to new images
- Improved interpretability through attention visualization
- More robust to variations in lighting and background

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Install development dependencies (`pip install -e ".[dev]"`)
4. Set up pre-commit hooks (`pre-commit install`)
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
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
- Plant disease dataset: [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
- Albumentations library: [Albumentations](https://albumentations.ai/)
- PyTorch: [PyTorch](https://pytorch.org/)
- Optuna: [Optuna](https://optuna.org/)
