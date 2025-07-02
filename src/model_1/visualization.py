"""
Visualization module for the knee osteoarthritis classifier.

This module contains functions for visualizing model results and knee X-ray images.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from typing import Dict, List, Tuple, Union, Optional
from PIL import Image
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.cm as cm

from src.model_1.config import get_config
from src.model_1.preprocessing import preprocess_image

def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot training history.

    Args:
        history: Dictionary containing training metrics
        save_path: Path to save the plot

    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=(12, 5))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    return plt.gcf()

def visualize_model_predictions(
    model: torch.nn.Module,
    images: List[Union[str, np.ndarray, Image.Image]],
    true_labels: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    device: Optional[torch.device] = None
) -> plt.Figure:
    """Visualize model predictions on a set of images.

    Args:
        model: Trained model
        images: List of images to visualize
        true_labels: List of true labels (optional)
        save_path: Path to save the visualization
        device: Device to run the model on

    Returns:
        Matplotlib figure
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to device
    model = model.to(device)
    model.eval()

    # Determine grid size
    n_images = len(images)
    grid_size = int(np.ceil(np.sqrt(n_images)))

    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()

    # KL grade descriptions (short version)
    kl_grades = {
        0: "Normal",
        1: "Doubtful",
        2: "Minimal",
        3: "Moderate",
        4: "Severe"
    }

    # Process each image
    for i, image_path in enumerate(images):
        if i >= len(axes):
            break

        # Load and preprocess image
        if isinstance(image_path, str):
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img_display = img.copy()
        elif isinstance(image_path, np.ndarray):
            img = image_path.copy()
            img_display = img.copy()
        elif isinstance(image_path, Image.Image):
            img = np.array(image_path.convert("L"))
            img_display = img.copy()

        # Preprocess for model
        img_tensor = preprocess_image(img, to_tensor=True).to(device)

        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            predicted_class = predicted.item()
            probs = probabilities[0].cpu().numpy()

        # Display image
        axes[i].imshow(img_display, cmap='gray')

        # Set title based on prediction and true label
        if true_labels is not None:
            title = f"Pred: {kl_grades[predicted_class]} (KL-{predicted_class})\nTrue: {kl_grades[true_labels[i]]} (KL-{true_labels[i]})"
            # Color based on correct/incorrect
            title_color = 'green' if predicted_class == true_labels[i] else 'red'
        else:
            title = f"Pred: {kl_grades[predicted_class]} (KL-{predicted_class})"
            title_color = 'black'

        axes[i].set_title(title, color=title_color)
        axes[i].axis('off')

        # Add probability bar at the bottom
        bar_height = 0.1
        for j, p in enumerate(probs):
            axes[i].add_patch(plt.Rectangle(
                (j/5, -bar_height*1.5), 1/5, bar_height*p*5,
                color=plt.cm.viridis(j/4),
                alpha=0.8
            ))
            if p > 0.2:  # Only show text for significant probabilities
                axes[i].text(
                    (j+0.5)/5, -bar_height*1.5 + bar_height*p*5/2,
                    f"{p:.2f}", ha='center', va='center',
                    color='white', fontsize=8
                )

    # Hide unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    return fig

def generate_gradcam(
    model: torch.nn.Module,
    image: Union[str, np.ndarray, Image.Image],
    target_layer_name: str = "backbone.layer4",
    target_class: Optional[int] = None,
    save_path: Optional[str] = None,
    device: Optional[torch.device] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Grad-CAM visualization for a model prediction.

    Args:
        model: Trained model
        image: Input image
        target_layer_name: Name of the target layer for Grad-CAM
        target_class: Target class index (None for predicted class)
        save_path: Path to save the visualization
        device: Device to run the model on

    Returns:
        Tuple of (original image, Grad-CAM overlay)
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to device
    model = model.to(device)
    model.eval()

    # Load and preprocess image
    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        img_display = img.copy()
    elif isinstance(image, np.ndarray):
        img = image.copy()
        img_display = img.copy()
    elif isinstance(image, Image.Image):
        img = np.array(image.convert("L"))
        img_display = img.copy()

    # Preprocess for model
    img_tensor = preprocess_image(img, to_tensor=True).to(device)

    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor.requires_grad = True

    # Get target layer
    target_layer = None
    for name, module in model.named_modules():
        if name == target_layer_name:
            target_layer = module
            break

    if target_layer is None:
        raise ValueError(f"Layer {target_layer_name} not found in model")

    # Forward pass
    activations = []
    gradients = []

    def save_activation(module, input, output):
        activations.append(output)

    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Register hooks
    handle1 = target_layer.register_forward_hook(save_activation)
    handle2 = target_layer.register_backward_hook(save_gradient)

    # Forward pass
    output = model(img_tensor)

    # Get predicted class if target_class is None
    if target_class is None:
        target_class = output.argmax(dim=1).item()

    # Zero gradients
    model.zero_grad()

    # Backward pass
    output[0, target_class].backward()

    # Remove hooks
    handle1.remove()
    handle2.remove()

    # Get activations and gradients
    activation = activations[0].detach().cpu().numpy()
    gradient = gradients[0].detach().cpu().numpy()

    # Calculate weights
    weights = np.mean(gradient, axis=(2, 3))[0, :]

    # Generate CAM
    cam = np.zeros(activation.shape[2:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * activation[0, i, :, :]

    # Apply ReLU and normalize
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (img_display.shape[1], img_display.shape[0]))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Convert grayscale image to RGB
    img_rgb = cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB)

    # Overlay heatmap on image
    overlay = heatmap * 0.4 + img_rgb
    overlay = overlay / overlay.max() * 255
    overlay = overlay.astype(np.uint8)

    # Create visualization
    if save_path:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(img_display, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(cam, cmap='jet')
        plt.title('Grad-CAM')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(overlay)
        plt.title('Overlay')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    return img_display, overlay