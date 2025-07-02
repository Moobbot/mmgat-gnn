"""
Model definition for Knee Osteoarthritis Classification.

This module contains the model architecture for classifying knee X-rays
according to the Kellgren-Lawrence grading system.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Optional, Tuple, Union, Any

from src.model_1.config import get_config
from config import get_system_config
from utils import setup_gpu

class KneeOsteoarthritisClassifier(nn.Module):
    """Knee Osteoarthritis Classifier based on the Kellgren-Lawrence grading system."""

    def __init__(
        self,
        num_classes: int = 5,
        architecture: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        """Initialize the model.

        Args:
            num_classes: Number of classes (KL grades 0-4)
            architecture: Backbone architecture (resnet18, resnet50, etc.)
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze the backbone weights
        """
        super(KneeOsteoarthritisClassifier, self).__init__()

        self.num_classes = num_classes
        self.architecture = architecture

        # Load the backbone model
        if architecture == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
        elif architecture == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
        elif architecture == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        # Modify the first layer to accept grayscale images (1 channel)
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace the final fully connected layer
        self.backbone.fc = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape [batch_size, channels, height, width]

        Returns:
            Output tensor of shape [batch_size, num_classes]
        """
        return self.backbone(x)

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions on input data.

        Args:
            x: Input tensor

        Returns:
            Tuple of (predicted_classes, probabilities)
        """
        self.eval()
        with torch.no_grad():
            outputs = self(x)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

        return predicted, probabilities

    def save(self, path: str) -> None:
        """Save the model to a file.

        Args:
            path: Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        """Load the model from a file.

        Args:
            path: Path to the saved model
        """
        self.load_state_dict(torch.load(path))

def create_model() -> KneeOsteoarthritisClassifier:
    """Create a model instance with configuration from settings.

    Returns:
        Initialized model
    """
    # Get model config
    config = get_config()

    # Get system config for GPU settings
    system_config = get_system_config()

    # Create model
    model = KneeOsteoarthritisClassifier(
        num_classes=config.NUM_CLASSES,
        architecture=config.MODEL_ARCHITECTURE,
        pretrained=config.PRETRAINED,
        freeze_backbone=config.FREEZE_BACKBONE
    )

    # Set up GPU if available and enabled
    if system_config.USE_GPU:
        device = setup_gpu()
        model = model.to(device)

        # Enable mixed precision if requested
        if system_config.MIXED_PRECISION and hasattr(torch.cuda, 'amp'):
            # This doesn't actually apply mixed precision yet,
            # but sets up the model to be ready for it
            # Mixed precision is applied during training/inference
            pass

    return model

def get_device() -> torch.device:
    """Get the device to use for the model.

    Returns:
        torch.device: Device to use
    """
    return setup_gpu()