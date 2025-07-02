"""
System-wide configuration module.

This module contains configuration settings for the entire system.
"""

import os
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Environment
ENV = os.environ.get("ENV", "development")
IS_DEV = ENV == "development"

# Load environment variables from .env file
load_dotenv()

class SystemConfig(BaseSettings):
    """System-wide configuration settings."""

    # API settings
    API_VERSION: str = Field(default="1.0.0", env="API_VERSION")
    API_TITLE: str = Field(default="ICTA API", env="API_TITLE")
    API_DESCRIPTION: str = Field(default="API for ICTA project", env="API_DESCRIPTION")
    API_PREFIX: str = Field(default="/api", env="API_PREFIX")
    DEBUG: bool = Field(default=IS_DEV, env="DEBUG")

    # Server settings
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")

    # CORS settings
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        env="CORS_ORIGINS"
    )
    CORS_ALLOWED_METHODS: List[str] = Field(
        default=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        env="CORS_ALLOWED_METHODS"
    )
    CORS_ALLOWED_HEADERS: List[str] = Field(
        default=["*"],
        env="CORS_ALLOWED_HEADERS"
    )

    # GPU settings
    USE_GPU: bool = Field(default=True, env="USE_GPU")
    GPU_DEVICE: int = Field(default=0, env="GPU_DEVICE")
    CUDA_VISIBLE_DEVICES: Optional[str] = Field(default=None, env="CUDA_VISIBLE_DEVICES")
    MIXED_PRECISION: bool = Field(default=True, env="MIXED_PRECISION")

    # Model settings
    ACTIVE_MODELS: List[str] = Field(
        default=["model_1"],
        env="ACTIVE_MODELS"
    )

    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# Create a global instance of the config
system_config = SystemConfig()

# Constants
FILE_RETENTION = 7  # days
ALLOWED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".dcm", ".zip"]
ERROR_MESSAGES = {
    "file_not_found": "File not found",
    "invalid_file_type": "Invalid file type",
    "processing_error": "Error processing file",
    "upload_error": "Error uploading file",
}

# Folders
FOLDERS = {
    "DATASETS": os.path.join(BASE_DIR, "datasets"),
    "RESULTS": os.path.join(BASE_DIR, "results"),
    "LOGS": os.path.join(BASE_DIR, "logs"),
    "MODELS": os.path.join(BASE_DIR, "models"),
}

# Logging configuration
LOG_CONFIG = {
    "FILE": os.path.join(FOLDERS["LOGS"], f"icta_api_{ENV}.log"),
    "FORMAT": "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
    "LEVEL": "DEBUG" if IS_DEV else "INFO",
    "CONSOLE": IS_DEV,
    "MAX_BYTES": 10 * 1024 * 1024,  # 10MB
    "BACKUP_COUNT": 5,
}

# Create folders if they don't exist
for folder in FOLDERS.values():
    os.makedirs(folder, exist_ok=True)

def get_system_config() -> SystemConfig:
    """Get the system configuration.

    Returns:
        SystemConfig: The system configuration.
    """
    return system_config