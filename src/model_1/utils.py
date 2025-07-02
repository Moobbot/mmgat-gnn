"""
Utility functions for the knee osteoarthritis classifier.

This module contains utility functions used across the model.
"""

import os
import logging
import hashlib
import json
import torch
import numpy as np
import random
from typing import Dict, List, Optional, Union, Any, Tuple
import matplotlib.pyplot as plt
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Random seed set to {seed}")

def get_device() -> torch.device:
    """Get the device to use for training/inference.

    Returns:
        PyTorch device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("GPU not available, using CPU")

    return device

def calculate_file_hash(file_path: str, algorithm: str = "sha256") -> str:
    """Calculate the hash of a file.

    Args:
        file_path: Path to the file
        algorithm: Hash algorithm to use

    Returns:
        File hash
    """
    hash_obj = hashlib.new(algorithm)

    with open(file_path, "rb") as f:
        # Read the file in chunks to avoid loading large files into memory
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)

    return hash_obj.hexdigest()

def save_json(data: Any, file_path: str) -> None:
    """Save data to a JSON file.

    Args:
        data: Data to save
        file_path: Path to save the file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Data saved to {file_path}")

def load_json(file_path: str) -> Any:
    """Load data from a JSON file.

    Args:
        file_path: Path to the file

    Returns:
        Loaded data
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    return data

def create_timestamp() -> str:
    """Create a timestamp string.

    Returns:
        Timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def create_experiment_dir(base_dir: str, experiment_name: str) -> str:
    """Create a directory for an experiment.

    Args:
        base_dir: Base directory
        experiment_name: Name of the experiment

    Returns:
        Path to the experiment directory
    """
    timestamp = create_timestamp()
    experiment_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)

    return experiment_dir

def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_summary(model: torch.nn.Module) -> None:
    """Print a summary of the model.

    Args:
        model: PyTorch model
    """
    print(f"Model: {model.__class__.__name__}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    # Print layer information
    print("\nLayer Information:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  {name}: {module.__class__.__name__} ({params:,} parameters)")

def plot_learning_rate(
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot the learning rate schedule.

    Args:
        scheduler: Learning rate scheduler
        optimizer: Optimizer
        num_epochs: Number of epochs
        save_path: Path to save the plot

    Returns:
        Matplotlib figure
    """
    # Store initial learning rate
    initial_lr = optimizer.param_groups[0]['lr']

    # Simulate learning rate schedule
    lrs = []
    for _ in range(num_epochs):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

    # Reset learning rate
    optimizer.param_groups[0]['lr'] = initial_lr

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), lrs)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)

    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    return plt.gcf()