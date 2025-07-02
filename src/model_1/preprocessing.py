"""
Preprocessing module for knee X-ray images.

This module contains functions for preprocessing knee X-ray images before feeding them to the model.
"""

import cv2
import numpy as np
import torch
from typing import Tuple, Union, Optional
from PIL import Image
import torchvision.transforms as transforms

from src.model_1.config import get_config

def preprocess_image(
    image: Union[str, np.ndarray, Image.Image],
    target_size: Optional[Tuple[int, int]] = None,
    normalize: bool = True,
    to_tensor: bool = True,
    device: Optional[torch.device] = None
) -> Union[torch.Tensor, np.ndarray]:
    """Preprocess an image for the model.

    Args:
        image: Input image as file path, numpy array, or PIL Image
        target_size: Target size (height, width) for resizing
        normalize: Whether to normalize pixel values
        to_tensor: Whether to convert to PyTorch tensor
        device: PyTorch device to place the tensor on

    Returns:
        Preprocessed image as a PyTorch tensor or numpy array
    """
    config = get_config()

    if target_size is None:
        target_size = tuple(config.IMAGE_SIZE)

    # Load image if it's a file path
    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image from {image}")
    elif isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert RGB to grayscale
            img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            img = image.copy()
    elif isinstance(image, Image.Image):
        img = np.array(image.convert("L"))
    else:
        raise TypeError("Image must be a file path, numpy array, or PIL Image")

    # Resize image
    img = cv2.resize(img, target_size)

    # Normalize pixel values
    if normalize:
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5  # Normalize to [-1, 1]

    if to_tensor:
        # Convert to PyTorch tensor
        img_tensor = torch.from_numpy(img).float()

        # Add channel dimension
        if len(img_tensor.shape) == 2:
            img_tensor = img_tensor.unsqueeze(0)

        # Move to device if specified
        if device is not None:
            img_tensor = img_tensor.to(device)

        return img_tensor

    return img

def apply_clahe(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.

    Args:
        image: Input grayscale image
        clip_limit: Threshold for contrast limiting

    Returns:
        CLAHE-enhanced image
    """
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))

    # Apply CLAHE
    return clahe.apply(image.astype(np.uint8))

def segment_knee(image: np.ndarray) -> np.ndarray:
    """Segment the knee region from an X-ray image.

    Args:
        image: Input grayscale image

    Returns:
        Segmented knee region
    """
    # This is a placeholder for a more sophisticated knee segmentation algorithm
    # In a real implementation, this would use techniques like thresholding,
    # active contours, or deep learning-based segmentation

    # Simple thresholding for demonstration
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask with the largest contour
    mask = np.zeros_like(image)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], 0, 255, -1)

        # Apply the mask to the original image
        segmented = cv2.bitwise_and(image, image, mask=mask)
        return segmented

    return image  # Return original image if no contours found

def preprocess_pipeline(
    image: Union[str, np.ndarray, Image.Image],
    apply_segmentation: bool = True,
    apply_enhancement: bool = True,
    to_tensor: bool = True
) -> Union[torch.Tensor, np.ndarray]:
    """Apply the full preprocessing pipeline to an image.

    Args:
        image: Input image
        apply_segmentation: Whether to apply knee segmentation
        apply_enhancement: Whether to apply CLAHE enhancement
        to_tensor: Whether to convert to PyTorch tensor

    Returns:
        Preprocessed image
    """
    config = get_config()

    # Load image if needed
    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    elif isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[2] == 3:
            img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            img = image.copy()
    elif isinstance(image, Image.Image):
        img = np.array(image.convert("L"))
    else:
        raise TypeError("Image must be a file path, numpy array, or PIL Image")

    # Apply segmentation if requested
    if apply_segmentation:
        img = segment_knee(img)

    # Apply CLAHE enhancement if requested
    if apply_enhancement:
        img = apply_clahe(img)

    # Apply final preprocessing
    return preprocess_image(
        img,
        target_size=tuple(config.IMAGE_SIZE),
        normalize=True,
        to_tensor=to_tensor
    )