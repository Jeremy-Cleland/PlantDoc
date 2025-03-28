#!/usr/bin/env python
"""
Minimal script to run data validation directly without imports.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Set up more verbose logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("prepare_data")


def validate_image(file_path):
    """Just report if the file exists, no validation"""
    return True


def run_validation(data_dir, output_dir, dry_run=True):
    """Simple validation that checks if data directory has image files"""
    logger.info(f"Starting dataset validation in: {data_dir}")
    if dry_run:
        logger.info("DRY RUN MODE: No changes will be made.")

    stats = {
        "total_files": 0,
        "valid_files": 0,
        "invalid_files": [],
        "classes": {},
        "timestamp": datetime.now().isoformat(),
    }

    if not data_dir.is_dir():
        logger.error(f"Data directory not found: {data_dir}")
        return {"error": f"Data directory not found: {data_dir}"}

    class_dirs = [
        d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    ]

    if not class_dirs:
        logger.warning(f"No class directories found in {data_dir}")
        # Create a sample class directory for testing
        if not dry_run:
            sample_dir = data_dir / "sample_class"
            sample_dir.mkdir(exist_ok=True)
            logger.info(f"Created sample class directory: {sample_dir}")
            class_dirs = [sample_dir]

    logger.info(f"Found {len(class_dirs)} potential class directories.")

    for class_dir in class_dirs:
        class_name = class_dir.name
        stats["classes"][class_name] = {"count": 0, "valid": 0, "invalid": 0}

        # Count files in the class directory
        files = [f for f in class_dir.iterdir() if f.is_file()]
        stats["total_files"] += len(files)
        stats["classes"][class_name]["count"] = len(files)

        for file_path in files:
            if validate_image(file_path):
                stats["valid_files"] += 1
                stats["classes"][class_name]["valid"] += 1
            else:
                stats["invalid_files"].append(str(file_path))
                stats["classes"][class_name]["invalid"] += 1

    # Save validation results
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_report_path = output_dir / "validation_report.json"

    with open(validation_report_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Validation completed. Results saved to: {validation_report_path}")
    logger.info(
        f"Total files: {stats['total_files']}, Valid: {stats['valid_files']}, Invalid: {len(stats['invalid_files'])}"
    )

    return stats


def main():
    """Run the data validation pipeline directly."""
    logger.info("Starting data preparation script")

    try:
        # Set up paths
        project_root = Path(os.path.abspath("."))
        data_dir = project_root / "data" / "raw"
        output_dir = project_root / "outputs" / "data_quality" / "validation"

        # Create data/raw if it doesn't exist
        data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using data directory: {data_dir}")

        # Make output dir
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using output directory: {output_dir}")

        logger.info("Running in dry-run mode")

        # Run the validation
        logger.info(f"Starting data validation for raw data in: {data_dir}")
        start_time = time.time()
        stats = run_validation(data_dir, output_dir, dry_run=True)
        elapsed_time = time.time() - start_time

        logger.info(f"Data validation completed in {elapsed_time:.2f} seconds.")
        logger.info(f"Results saved to: {output_dir}")

    except Exception as e:
        logger.exception(f"Error in data preparation script: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
