"""
Configuration module for the Model 1.

This module contains configuration settings specific to the knee osteoarthritis classifier model.
"""

import os
from typing import Dict, List, Optional, Union
from pydantic import BaseSettings, Field

# Import system config to get global settings
from config import get_system_config, FOLDERS, BASE_DIR

# Get system config
system_config = get_system_config()

class ModelConfig(BaseSettings):
    """Configuration settings for the knee osteoarthritis classifier model."""

    # Model parameters
    MODEL_NAME: str = "KneeOsteoarthritisClassifier"
    MODEL_VERSION: str = "1.0.0"

    # Training parameters
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.001
    NUM_EPOCHS: int = 50
    EARLY_STOPPING_PATIENCE: int = 10

    # Data parameters
    TRAIN_DATA_PATH: str = Field(default=os.path.join(FOLDERS["DATASETS"], "train"))
    VAL_DATA_PATH: str = Field(default=os.path.join(FOLDERS["DATASETS"], "val"))
    TEST_DATA_PATH: str = Field(default=os.path.join(FOLDERS["DATASETS"], "test"))
    IMAGE_SIZE: List[int] = Field(default=[224, 224])
    NUM_CLASSES: int = 5  # KL grades 0-4

    # Model architecture
    MODEL_ARCHITECTURE: str = "resnet50"
    PRETRAINED: bool = True
    FREEZE_BACKBONE: bool = False

    # Paths - use system config paths
    MODEL_SAVE_PATH: str = os.path.join(FOLDERS["MODELS"], "model_1")
    RESULTS_DIR: str = os.path.join(FOLDERS["RESULTS"], "model_1")
    LOG_DIR: str = os.path.join(FOLDERS["LOGS"], "model_1")

    class Config:
        """Pydantic config."""
        env_prefix = "MODEL1_"  # Use MODEL1_ prefix for environment variables

# Create a global instance of the config
model_config = ModelConfig()

# Create necessary directories
os.makedirs(model_config.MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(model_config.RESULTS_DIR, exist_ok=True)
os.makedirs(model_config.LOG_DIR, exist_ok=True)
os.makedirs(model_config.TRAIN_DATA_PATH, exist_ok=True)
os.makedirs(model_config.VAL_DATA_PATH, exist_ok=True)
os.makedirs(model_config.TEST_DATA_PATH, exist_ok=True)

def get_config() -> ModelConfig:
    """Get the model configuration.

    Returns:
        ModelConfig: The model configuration.
    """
    return model_config