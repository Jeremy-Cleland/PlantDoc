# CBAM Classification

Deep Learning project for multiclass CNN with CBAM.




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
git clone https://github.com/yourusername/plantdoc.git
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
git clone https://github.com/yourusername/plantdoc.git
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
from plantdoc.core.models.registry import list_models

# This should print the available models
print(list_models())
```

Or by running the CLI:

```bash
python -m plantdoc.cli.main --help
```

## Common Issues

### Import Errors

If you encounter import errors, make sure:
1. The virtual environment is activated
2. The package is installed in development mode
3. Your imports use the correct package structure (see examples)

### CUDA/MPS Issues

For GPU acceleration:
- CUDA: Make sure your CUDA version is compatible with the installed PyTorch version
- MPS (Mac with Apple Silicon): Use PyTorch 2.0+ for proper MPS support