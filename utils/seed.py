"""
Seed setting utilities for reproducibility.
"""

import os
import random

import numpy as np
import torch

from utils.logger import get_logger

logger = get_logger(__name__)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set seeds for reproducibility.

    This sets seeds for random, numpy, torch, and other libraries
    to ensure reproducible results.

    Args:
        seed: Seed number
        deterministic: Whether to set deterministic algorithms in torch
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set CUDA seeds if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Set deterministic behavior (note: this may impact performance)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Set deterministic algorithms for ops with non-deterministic implementations
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    
    logger.info(f"Set random seed to {seed} (deterministic={deterministic})")


def worker_init_fn(worker_id: int) -> None:
    """
    Initialize seeds for DataLoader workers.
    
    This should be passed to DataLoader's worker_init_fn parameter
    to ensure each worker has a different but reproducible seed.
    
    Args:
        worker_id: Worker ID from DataLoader
    """
    # Get base seed from torch
    base_seed = torch.initial_seed()
    
    # Different seed for each worker but still deterministic
    seeded_worker_id = base_seed + worker_id
    
    # Set seed for this worker
    random.seed(seeded_worker_id)
    np.random.seed(seeded_worker_id % (2**32 - 1))  # numpy only accepts 32-bit seeds
    torch.manual_seed(seeded_worker_id)