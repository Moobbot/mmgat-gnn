"""
Evaluation module for the knee osteoarthritis classifier.

This module contains functions for evaluating the model's performance.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Union, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.model_1.config import get_config
from src.model_1.model import KneeOsteoarthritisClassifier

def calculate_metrics(
    y_true: Union[List[int], np.ndarray, torch.Tensor],
    y_pred: Union[List[int], np.ndarray, torch.Tensor],
    average: str = 'weighted'
) -> Dict[str, float]:
    """Calculate classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: Averaging method for multi-class metrics

    Returns:
        Dictionary of metrics
    """
    # Convert to numpy arrays if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

    # Calculate mean absolute error (specific to KL grading)
    mae = np.mean(np.abs(y_true - y_pred))

    # Calculate exact agreement and agreement within 1 grade
    exact_agreement = np.mean(y_true == y_pred)
    within_one_grade = np.mean(np.abs(y_true - y_pred) <= 1)

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'mae': float(mae),
        'exact_agreement': float(exact_agreement),
        'within_one_grade': float(within_one_grade)
    }

def plot_confusion_matrix(
    y_true: Union[List[int], np.ndarray, torch.Tensor],
    y_pred: Union[List[int], np.ndarray, torch.Tensor],
    normalize: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        normalize: Normalization method ('true', 'pred', 'all', or None)
        save_path: Path to save the plot

    Returns:
        Matplotlib figure
    """
    # Convert to numpy arrays if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize if requested
    if normalize == 'true':
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        title = 'Normalized Confusion Matrix (by true label)'
    elif normalize == 'pred':
        cm = cm.astype('float') / cm.sum(axis=0, keepdims=True)
        title = 'Normalized Confusion Matrix (by predicted label)'
    elif normalize == 'all':
        cm = cm.astype('float') / cm.sum()
        title = 'Normalized Confusion Matrix (by all)'
    else:
        title = 'Confusion Matrix'

    # Create figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
                xticklabels=['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4'],
                yticklabels=['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)

    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    return plt.gcf()

def evaluate_model(
    model: KneeOsteoarthritisClassifier,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate the model on a dataset.

    Args:
        model: Model to evaluate
        data_loader: Data loader for the dataset
        criterion: Loss function
        device: Device to run the evaluation on

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()

    all_labels = []
    all_predictions = []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Get predictions
            _, predictions = torch.max(outputs, 1)

            # Accumulate data
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            total_loss += loss.item() * images.size(0)

    # Calculate average loss
    avg_loss = total_loss / len(data_loader.dataset)

    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_predictions)
    metrics['loss'] = avg_loss

    return metrics