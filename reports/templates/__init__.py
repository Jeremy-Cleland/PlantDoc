"""
Templates for HTML report generation.
"""

from pathlib import Path

# Get the path to the templates directory
TEMPLATES_DIR = Path(__file__).parent.absolute()

# Define template file paths
TRAINING_REPORT_TEMPLATE = TEMPLATES_DIR / "training_report.html"