"""
System-wide utility functions.

This module contains utility functions used across the system.
"""

import logging
import os
import hashlib
import json
import shutil
from datetime import datetime
import socket
from typing import List, Dict, Any, Optional, Union
import psutil

from config import get_system_config, FOLDERS, ERROR_MESSAGES, ALLOWED_EXTENSIONS

# Set up logging
logger = logging.getLogger(__name__)

def get_local_ip() -> str:
    """Get local IP address.

    Returns:
        str: Local IP address
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Use Google's DNS server as a reference point
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        logger.error(f"Error getting local IP: {str(e)}")
        return "127.0.0.1"

def get_system_metrics() -> Dict[str, Any]:
    """Get system metrics.

    Returns:
        Dict[str, Any]: System metrics
    """
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # Disk usage
        disk = psutil.disk_usage("/")
        disk_percent = disk.percent

        # Network I/O
        net_io = psutil.net_io_counters()

        return {
            "cpu_usage": cpu_percent,
            "memory_usage": memory_percent,
            "disk_usage": disk_percent,
            "network": {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv,
            },
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting system metrics: {str(e)}")
        return {
            "error": "Failed to get system metrics",
            "timestamp": datetime.now().isoformat(),
        }

def calculate_file_hash(file_path: str, algorithm: str = "sha256") -> str:
    """Calculate the hash of a file.

    Args:
        file_path: Path to the file
        algorithm: Hash algorithm to use

    Returns:
        str: File hash
    """
    hash_obj = hashlib.new(algorithm)

    with open(file_path, "rb") as f:
        # Read the file in chunks to avoid loading large files into memory
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)

    return hash_obj.hexdigest()

def save_file(file_content: bytes, file_path: str) -> str:
    """Save file content to disk.

    Args:
        file_content: File content
        file_path: Path to save the file

    Returns:
        str: Path to the saved file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(file_content)

    return file_path

def save_uploaded_file(
    file,
    destination_folder: str,
    filename: Optional[str] = None
) -> str:
    """Save an uploaded file.

    Args:
        file: FastAPI UploadFile
        destination_folder: Folder to save the file
        filename: Custom filename (if None, use the original filename)

    Returns:
        str: Path to the saved file
    """
    # Create destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Use original filename if no custom filename is provided
    if filename is None:
        filename = file.filename

    # Full path to save the file
    file_path = os.path.join(destination_folder, filename)

    # Save the file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return file_path

def is_valid_file_type(filename: str) -> bool:
    """Check if a file has an allowed extension.

    Args:
        filename: Filename to check

    Returns:
        bool: True if the file has an allowed extension
    """
    ext = os.path.splitext(filename.lower())[1]
    return ext in ALLOWED_EXTENSIONS

def create_timestamp() -> str:
    """Create a timestamp string.

    Returns:
        str: Timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

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
        Any: Loaded data
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    return data

def setup_gpu() -> torch.device:
    """Set up GPU for PyTorch based on system configuration.

    Returns:
        torch.device: Device to use for PyTorch operations
    """
    try:
        import torch

        config = get_system_config()

        # Set CUDA_VISIBLE_DEVICES environment variable if specified
        if config.CUDA_VISIBLE_DEVICES is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
            logger.info(f"Set CUDA_VISIBLE_DEVICES to {config.CUDA_VISIBLE_DEVICES}")

        # Check if GPU is available and should be used
        if config.USE_GPU and torch.cuda.is_available():
            # Set the device
            device = torch.device(f"cuda:{config.GPU_DEVICE}")

            # Log GPU information
            gpu_name = torch.cuda.get_device_name(config.GPU_DEVICE)
            gpu_memory = torch.cuda.get_device_properties(config.GPU_DEVICE).total_memory / (1024 ** 3)  # Convert to GB
            logger.info(f"Using GPU: {gpu_name} with {gpu_memory:.2f} GB memory")

            # Set up mixed precision if requested
            if config.MIXED_PRECISION:
                if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
                    logger.info("Mixed precision training enabled")
                else:
                    logger.warning("Mixed precision requested but not supported by PyTorch version")

            return device
        else:
            if not config.USE_GPU:
                logger.info("GPU usage disabled by configuration")
            elif not torch.cuda.is_available():
                logger.warning("GPU requested but not available")

            return torch.device("cpu")

    except ImportError:
        logger.warning("PyTorch not installed, defaulting to CPU")
        return "cpu"
    except Exception as e:
        logger.error(f"Error setting up GPU: {str(e)}")
        return torch.device("cpu")
