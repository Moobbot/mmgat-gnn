"""
Prediction module for the knee osteoarthritis classifier.

This module contains functions for making predictions with the trained model.
"""

import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from PIL import Image
import json
import logging

from src.model_1.config import get_config
from src.model_1.model import KneeOsteoarthritisClassifier, create_model, get_device
from src.model_1.preprocessing import preprocess_pipeline
from config import get_system_config

# Set up logging
logger = logging.getLogger(__name__)

# KL grade descriptions
KL_DESCRIPTIONS = {
    0: "Grade 0: Normal",
    1: "Grade 1: Doubtful narrowing of joint space and possible osteophytic lipping",
    2: "Grade 2: Definite osteophytes, definite narrowing of joint space",
    3: "Grade 3: Moderate multiple osteophytes, definite narrowing of joint space, some sclerosis and possible deformity of bone contour",
    4: "Grade 4: Large osteophytes, marked narrowing of joint space, severe sclerosis and definite deformity of bone contour"
}

def load_model(model_path: Optional[str] = None) -> KneeOsteoarthritisClassifier:
    """Load the trained model.

    Args:
        model_path: Path to the trained model weights

    Returns:
        Loaded model
    """
    config = get_config()

    # Use provided path or default from config
    if model_path is None:
        model_path = os.path.join(config.MODEL_SAVE_PATH, f"{config.MODEL_NAME}.pth")

    # Create model with config settings
    model = create_model()

    # Load weights if the file exists
    if os.path.exists(model_path):
        logger.info(f"Loading model from {model_path}")
        model.load(model_path)
    else:
        logger.warning(f"Model file {model_path} not found. Using untrained model.")

    return model

def predict_single_image(
    image_path: str,
    model: Optional[KneeOsteoarthritisClassifier] = None,
    device: Optional[torch.device] = None
) -> Dict:
    """Make a prediction for a single image.

    Args:
        image_path: Path to the image
        model: Model to use for prediction (loads from disk if None)
        device: Device to run the prediction on

    Returns:
        Dictionary with prediction results
    """
    # Get system config
    system_config = get_system_config()

    # Set device
    if device is None:
        device = get_device()

    # Load model if not provided
    if model is None:
        model = load_model()

    # Move model to device
    model = model.to(device)
    model.eval()

    # Enable mixed precision if requested
    use_amp = system_config.MIXED_PRECISION and hasattr(torch.cuda, 'amp') and device.type == 'cuda'

    # Preprocess the image
    try:
        image_tensor = preprocess_pipeline(image_path, to_tensor=True)
        image_tensor = image_tensor.to(device)

        # Add batch dimension if needed
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            # Use mixed precision if enabled
            if use_amp:
                with torch.amp.autocast(device_type='cuda'):
                    predicted_class, probabilities = model.predict(image_tensor)
            else:
                predicted_class, probabilities = model.predict(image_tensor)

            # Convert to Python types for JSON serialization
            predicted_class = predicted_class.item() if isinstance(predicted_class, torch.Tensor) else int(predicted_class[0])
            probabilities = {str(i): float(prob) for i, prob in enumerate(probabilities[0])}

        # Create result dictionary
        result = {
            "image_path": image_path,
            "predicted_class": predicted_class,
            "predicted_grade": f"KL-{predicted_class}",
            "description": KL_DESCRIPTIONS[predicted_class],
            "probabilities": probabilities
        }

        return result

    except Exception as e:
        logger.error(f"Error predicting image {image_path}: {str(e)}")
        return {
            "image_path": image_path,
            "error": str(e)
        }

def predict_batch(
    image_paths: List[str],
    model: Optional[KneeOsteoarthritisClassifier] = None,
    device: Optional[torch.device] = None,
    batch_size: int = 16
) -> List[Dict]:
    """Make predictions for a batch of images.

    Args:
        image_paths: List of paths to images
        model: Model to use for prediction (loads from disk if None)
        device: Device to run the prediction on
        batch_size: Batch size for processing

    Returns:
        List of dictionaries with prediction results
    """
    # Get system config
    system_config = get_system_config()

    # Set device
    if device is None:
        device = get_device()

    # Load model if not provided
    if model is None:
        model = load_model()

    # Move model to device
    model = model.to(device)
    model.eval()

    # Enable mixed precision if requested
    use_amp = system_config.MIXED_PRECISION and hasattr(torch.cuda, 'amp') and device.type == 'cuda'

    results = []

    # Process in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_tensors = []

        # Preprocess each image
        for image_path in batch_paths:
            try:
                image_tensor = preprocess_pipeline(image_path, to_tensor=True)
                batch_tensors.append(image_tensor)
            except Exception as e:
                logger.error(f"Error preprocessing image {image_path}: {str(e)}")
                results.append({
                    "image_path": image_path,
                    "error": str(e)
                })

        if batch_tensors:
            # Stack tensors into a batch
            batch = torch.stack(batch_tensors).to(device)

            # Make predictions
            with torch.no_grad():
                # Use mixed precision if enabled
                if use_amp:
                    with torch.amp.autocast(device_type='cuda'):
                        outputs = model(batch)
                else:
                    outputs = model(batch)

                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted_classes = torch.max(outputs, 1)

            # Process results
            for j, (path, pred_class) in enumerate(zip(batch_paths, predicted_classes)):
                if j < len(batch_tensors):  # Skip images that failed preprocessing
                    probs = {str(k): float(p) for k, p in enumerate(probabilities[j])}
                    results.append({
                        "image_path": path,
                        "predicted_class": int(pred_class.item()),
                        "predicted_grade": f"KL-{pred_class.item()}",
                        "description": KL_DESCRIPTIONS[pred_class.item()],
                        "probabilities": probs
                    })

    return results

def save_predictions(predictions: Union[Dict, List[Dict]], output_path: str) -> None:
    """Save predictions to a JSON file.

    Args:
        predictions: Prediction results
        output_path: Path to save the results
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)

    logger.info(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Make predictions with the knee osteoarthritis classifier")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--dir", type=str, help="Path to a directory of images")
    parser.add_argument("--output", type=str, default="predictions.json", help="Output file path")
    parser.add_argument("--model", type=str, help="Path to model weights")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Load model
    model = load_model(args.model)

    if args.image:
        # Predict single image
        result = predict_single_image(args.image, model)
        save_predictions(result, args.output)

    elif args.dir:
        # Get all image files in directory
        image_files = []
        for ext in ['.png', '.jpg', '.jpeg']:
            image_files.extend([os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.lower().endswith(ext)])

        if not image_files:
            logger.error(f"No image files found in {args.dir}")
            exit(1)

        # Predict batch of images
        results = predict_batch(image_files, model, batch_size=args.batch_size)
        save_predictions(results, args.output)

    else:
        logger.error("Either --image or --dir must be specified")
        parser.print_help()
        exit(1)