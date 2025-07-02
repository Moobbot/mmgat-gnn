"""
Data loading and processing module.

This module handles loading and processing of knee X-ray images for the model.
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union, Callable
from PIL import Image
import torchvision.transforms as transforms

from src.model_1.config import get_config
from src.model_1.preprocessing import preprocess_image

class KneeXrayDataset(Dataset):
    """Dataset for knee X-ray images with KL grades."""

    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        split: str = "train"
    ):
        """Initialize the dataset.

        Args:
            data_dir: Directory containing the data
            transform: Transformations to apply to the images
            split: Data split (train, val, test)
        """
        self.data_dir = data_dir
        self.transform = transform
        self.split = split

        # Load the data
        self.data = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        """Load the data from the data directory.

        Returns:
            DataFrame containing image paths and labels
        """
        # In a real implementation, this would load a CSV file with image paths and labels
        # For this example, we'll simulate it by scanning the directory

        image_paths = []
        labels = []

        # Pattern: {data_dir}/{split}/grade_{grade}/*.png
        for grade in range(5):  # KL grades 0-4
            grade_dir = os.path.join(self.data_dir, self.split, f"grade_{grade}")
            if os.path.exists(grade_dir):
                for img_path in glob.glob(os.path.join(grade_dir, "*.png")):
                    image_paths.append(img_path)
                    labels.append(grade)

        # Create a DataFrame
        df = pd.DataFrame({
            "image_path": image_paths,
            "kl_grade": labels
        })

        return df

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            Number of samples
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (image, label)
        """
        # Get the image path and label
        image_path = self.data.iloc[idx]["image_path"]
        label = self.data.iloc[idx]["kl_grade"]

        # Load the image
        image = Image.open(image_path).convert("L")  # Convert to grayscale

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        return image, label

def get_data_loaders(
    batch_size: int = 32,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """Get data loaders for training, validation, and testing.

    Args:
        batch_size: Batch size for the data loaders
        num_workers: Number of workers for the data loaders

    Returns:
        Dictionary containing data loaders for each split
    """
    config = get_config()

    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Create datasets
    train_dataset = KneeXrayDataset(
        data_dir=config.TRAIN_DATA_PATH,
        transform=train_transform,
        split="train"
    )

    val_dataset = KneeXrayDataset(
        data_dir=config.VAL_DATA_PATH,
        transform=val_test_transform,
        split="val"
    )

    test_dataset = KneeXrayDataset(
        data_dir=config.TEST_DATA_PATH,
        transform=val_test_transform,
        split="test"
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }