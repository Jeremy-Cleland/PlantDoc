"""
Utilities for working with Apple Metal Performance Shaders (MPS) on macOS.
"""

import os
import platform
from typing import Optional, Tuple, Union

import torch

from utils.logger import get_logger

logger = get_logger(__name__)


def is_mps_available() -> bool:
    """
    Check if MPS (Metal Performance Shaders) is available.

    Returns:
        bool: True if MPS is available, False otherwise
    """
    if platform.system() != "Darwin":
        return False

    try:
        if not torch.backends.mps.is_available():
            return False
        if not torch.backends.mps.is_built():
            return False
        # Actually try to create a tensor to confirm MPS works
        torch.zeros(1).to(torch.device("mps"))
        return True
    except (AttributeError, AssertionError, RuntimeError):
        return False


def get_mps_device() -> Optional[torch.device]:
    """
    Get MPS device if available.

    Returns:
        Optional[torch.device]: MPS device if available, None otherwise
    """
    if is_mps_available():
        return torch.device("mps")
    return None


def set_mps_device(device_name: str = None) -> torch.device:
    """
    Set the device to use for training based on availability.

    Args:
        device_name: Device to use (mps, cuda, cpu). If None, it will try to use MPS if available,
                    or fall back to CPU.

    Returns:
        torch.device: The selected device
    """
    if device_name == "mps" and is_mps_available():
        device = torch.device("mps")
        logger.info("Using MPS (Metal Performance Shaders) device")
    elif device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        if device_name in ("mps", "cuda") and device_name != "cpu":
            logger.warning(
                f"Requested device '{device_name}' is not available, falling back to CPU"
            )
        device = torch.device("cpu")
        logger.info("Using CPU device")

    return device


def get_device_info(device: torch.device) -> Tuple[str, Union[str, None]]:
    """
    Get information about the device.

    Args:
        device: PyTorch device

    Returns:
        Tuple[str, Union[str, None]]: Device type and name if available
    """
    device_name = None

    if device.type == "mps":
        device_type = "MPS (Metal Performance Shaders)"
        # No direct way to get the GPU name in PyTorch for MPS
        try:
            import subprocess

            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
            )
            output = result.stdout
            for line in output.split("\n"):
                if "Chipset Model" in line:
                    device_name = line.split(":")[1].strip()
                    break
        except Exception:
            pass
    elif device.type == "cuda":
        device_type = "CUDA"
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(
                device.index if device.index else 0
            )
    else:
        device_type = "CPU"
        try:
            import platform

            device_name = platform.processor()
        except Exception:
            pass

    return device_type, device_name


def optimize_for_mps(module: torch.nn.Module) -> torch.nn.Module:
    """
    Apply MPS-specific optimizations to the module if running on MPS.

    Args:
        module: PyTorch module to optimize

    Returns:
        torch.nn.Module: Optimized module
    """
    # This function can be expanded later with MPS-specific optimizations
    # Currently just returns the module as-is
    return module


def synchronize() -> None:
    """
    Wait for all MPS or CUDA operations to complete.
    Useful before timing operations or when measuring memory usage.
    """
    if is_mps_available():
        torch.mps.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def empty_cache(force_gc: bool = False) -> None:
    """
    Empty the MPS cache to free up memory.
    This should be called between large operations or after validation epochs.

    Args:
        force_gc: Whether to also run Python's garbage collector
    """
    if is_mps_available():
        # Empty the MPS cache
        torch.mps.empty_cache()

        # Optionally run Python's garbage collector
        if force_gc:
            import gc

            gc.collect()
            # Empty cache again after GC
            torch.mps.empty_cache()


def deep_clean_memory() -> None:
    """
    Perform a deep cleaning of memory on MPS devices.
    This function is more aggressive than empty_cache and should be used
    before/after major operations that could cause memory pressure.
    """
    if not is_mps_available():
        return

    # First run Python's garbage collector
    import gc

    gc.collect()

    # Empty MPS cache
    torch.mps.empty_cache()

    # Short sleep to let system process the cleanup
    import time

    time.sleep(0.1)

    # Run GC and empty cache once more
    gc.collect()
    torch.mps.empty_cache()


def log_memory_stats(description: str = "Current", log_level: str = "debug") -> dict:
    """
    Log memory statistics for the current device.

    Args:
        description: Description for the log message
        log_level: Log level to use (debug, info, warning, error)

    Returns:
        dict: Memory statistics
    """
    import gc

    import psutil

    # Get process memory info
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    # Prepare stats dictionary
    stats = {
        "system_rss_mb": memory_info.rss / (1024 * 1024),
        "system_vms_mb": memory_info.vms / (1024 * 1024),
    }

    # Force garbage collection
    gc.collect()

    # Device-specific memory info
    if torch.cuda.is_available():
        # CUDA memory stats
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)

        stats.update(
            {
                "device": "cuda",
                "allocated_mb": allocated,
                "reserved_mb": reserved,
                "max_allocated_mb": max_allocated,
            }
        )
    elif is_mps_available():
        # MPS doesn't have built-in memory stats, but we can estimate
        # by checking before and after allocating a large tensor
        try:
            # Synchronize to ensure all operations are complete
            synchronize()

            # Get system memory before allocation
            before = process.memory_info().rss / (1024 * 1024)

            # Allocate a 100MB tensor on MPS
            temp_tensor = torch.ones((1, 100, 1024, 1024), device="mps")
            synchronize()

            # Get system memory after allocation
            after = process.memory_info().rss / (1024 * 1024)

            # Clean up
            del temp_tensor
            empty_cache(force_gc=True)

            # The difference gives us a rough idea of MPS memory usage
            stats.update(
                {
                    "device": "mps",
                    "estimated_usage_mb": after - before,
                }
            )
        except Exception as e:
            stats.update(
                {
                    "device": "mps",
                    "error": str(e),
                }
            )
    else:
        stats["device"] = "cpu"

    # Format log message
    log_msg = f"{description} memory stats: "
    for key, value in stats.items():
        if isinstance(value, float):
            log_msg += f"{key}={value:.2f}MB, "
        else:
            log_msg += f"{key}={value}, "

    # Log at appropriate level
    if log_level == "info":
        logger.info(log_msg[:-2])  # Remove trailing comma and space
    elif log_level == "warning":
        logger.warning(log_msg[:-2])
    elif log_level == "error":
        logger.error(log_msg[:-2])
    else:  # Default to debug
        logger.debug(log_msg[:-2])

    return stats


class MPSProfiler:
    """
    Context manager for MPS profiling.
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize the MPS profiler.

        Args:
            enabled: Whether to enable profiling.
        """
        self.enabled = enabled and is_mps_available()

    def __enter__(self):
        """
        Enter the profiling context.
        """
        if self.enabled:
            torch.mps.profiler.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the profiling context and save the profile.
        """
        if self.enabled:
            torch.mps.profiler.stop()
            # Force synchronization after stopping profiler
            synchronize()
            # Clear memory after profiling
            empty_cache()
            logger.info("MPS profiler stopped and resources released")


def set_manual_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set the seed for MPS and other devices.

    This is a device-specific extension of set_seed that handles
    MPS-specific seeding when available.

    Args:
        seed: Seed number to set
        deterministic: Whether to set deterministic algorithms in torch
    """
    # Set random libraries seeds
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    # Set standard torch seed
    torch.manual_seed(seed)

    # Set MPS seed if available
    if is_mps_available():
        # MPS doesn't have a separate seed setting function,
        # but we can ensure tensors are created deterministically
        # by setting the global seed and creating a test tensor
        torch.mps.manual_seed(seed)
        # Create a test tensor to ensure proper seeding
        _ = torch.randn(1, device="mps")
        synchronize()
        logger.debug(f"Set MPS random seed to {seed}")

    # Set CUDA seed if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Set deterministic behavior if requested
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            # Set deterministic algorithms for ops with non-deterministic implementations
            import os

            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True, warn_only=True)

        logger.debug(f"Set CUDA random seed to {seed} (deterministic={deterministic})")

    logger.info(f"Set random seed to {seed} (deterministic={deterministic})")
