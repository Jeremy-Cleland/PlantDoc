#!/usr/bin/env python
"""
Script for running parameter sweeps, similar to Hydra's multirun functionality.
"""

import argparse
import itertools
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def run_sweep(base_command, sweep_params, output_dir=None):
    """
    Run a parameter sweep similar to Hydra's multirun.

    Args:
        base_command: Base command to run (e.g., "python cli/main.py train")
        sweep_params: Dictionary mapping parameter names to lists of values
        output_dir: Optional output directory for sweep results
    """
    # Create product of all parameter combinations
    param_names = sweep_params.keys()
    param_values = sweep_params.values()
    combinations = list(itertools.product(*param_values))

    # Set up output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if output_dir is None:
        sweep_dir = Path("outputs/multirun") / timestamp
    else:
        sweep_dir = Path(output_dir) / timestamp

    sweep_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running {len(combinations)} combinations")
    print(f"Sweep results will be saved to: {sweep_dir}")

    # Create summary file
    with open(sweep_dir / "sweep_summary.txt", "w") as f:
        f.write("Parameter Sweep Summary\n")
        f.write("======================\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Base Command: {base_command}\n\n")
        f.write("Parameters:\n")
        for name, values in sweep_params.items():
            f.write(f"  {name}: {values}\n")
        f.write(f"\nTotal Combinations: {len(combinations)}\n\n")

    # Run each combination
    for i, combination in enumerate(combinations):
        # Create parameter string
        params = " ".join(
            [f"{name}={value}" for name, value in zip(param_names, combination)]
        )

        # Create a unique run directory for this combination
        run_dir = sweep_dir / f"run_{i+1}"
        run_dir.mkdir(exist_ok=True)

        # Create run-specific override for experiment name to enable separate output directories
        experiment_name = f"sweep_{timestamp}_run_{i+1}"
        full_params = f"{params} paths.experiment_name={experiment_name}"

        # Construct full command
        full_command = f"{base_command} {full_params}"

        # Log the command
        with open(sweep_dir / "sweep_summary.txt", "a") as f:
            f.write(f"\nRun {i+1}/{len(combinations)}:\n")
            f.write(f"  Command: {full_command}\n")
            f.write(f"  Parameters: {dict(zip(param_names, combination))}\n")

        print(f"\nRun {i+1}/{len(combinations)}: {full_command}")
        print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")

        start_time = time.time()

        # Run the command and capture output
        result = subprocess.run(
            full_command, shell=True, capture_output=True, text=True
        )

        duration = time.time() - start_time

        # Save command output
        with open(run_dir / "output.log", "w") as f:
            f.write(f"Command: {full_command}\n")
            f.write(f"Return Code: {result.returncode}\n")
            f.write(f"Duration: {duration:.2f} seconds\n\n")
            f.write("STDOUT:\n")
            f.write(result.stdout)
            f.write("\nSTDERR:\n")
            f.write(result.stderr)

        # Update summary
        with open(sweep_dir / "sweep_summary.txt", "a") as f:
            f.write(f"  Duration: {duration:.2f} seconds\n")
            f.write(f"  Status: {'Success' if result.returncode == 0 else 'Failed'}\n")

        print(f"Completed in {duration:.2f} seconds")
        print(f"Status: {'Success' if result.returncode == 0 else 'Failed'}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run parameter sweeps")
    parser.add_argument(
        "command", help="Base command to run (e.g., 'train' for training pipeline)"
    )
    parser.add_argument(
        "--param",
        "-p",
        action="append",
        dest="params",
        help="Parameter in format 'name=value1,value2,value3'",
        required=True,
    )
    parser.add_argument(
        "--output-dir", "-o", help="Output directory for sweep results", default=None
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()

    # Parse parameters
    sweep_params = {}
    for param_str in args.params:
        if "=" not in param_str:
            print(
                f"Error: Parameter must be in format 'name=value1,value2,value3', got '{param_str}'"
            )
            sys.exit(1)

        name, values_str = param_str.split("=", 1)
        values = values_str.split(",")
        sweep_params[name] = values

    # Construct base command
    if args.command in ["train", "eval", "tune", "prepare"]:
        base_command = f"python cli/main.py {args.command}"
    else:
        base_command = args.command

    # Run sweep
    run_sweep(base_command, sweep_params, args.output_dir)


if __name__ == "__main__":
    main()
