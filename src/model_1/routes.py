"""
API routes for the knee osteoarthritis classifier.

This module defines the FastAPI routes for the model.
"""

import os
import io
import json
import logging
from typing import Dict, List, Optional, Union
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import torch
import numpy as np
from PIL import Image
import uuid
import shutil
from datetime import datetime

from src.model_1.config import get_config
from src.model_1.model import create_model
from src.model_1.predict import predict_single_image, predict_batch, load_model
from src.model_1.preprocessing import preprocess_pipeline

# Set up logging
logger = logging.getLogger(__name__)

# Create router - note that we don't include the prefix here as it will be added by the main API
router = APIRouter(
    prefix="/model_1",  # This will be appended to the API_PREFIX from the main config
    tags=["model_1"],
    responses={404: {"description": "Not found"}},
)

# Model schemas
class PredictionResult(BaseModel):
    """Schema for prediction results."""
    image_id: str
    predicted_class: int
    predicted_grade: str
    description: str
    probabilities: Dict[str, float]

class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction requests."""
    image_urls: List[str]

class ModelInfo(BaseModel):
    """Schema for model information."""
    name: str
    version: str
    description: str
    input_shape: List[int]
    num_classes: int
    classes: Dict[int, str]
    architecture: str

# KL grade descriptions
KL_DESCRIPTIONS = {
    0: "Grade 0: Normal",
    1: "Grade 1: Doubtful narrowing of joint space and possible osteophytic lipping",
    2: "Grade 2: Definite osteophytes, definite narrowing of joint space",
    3: "Grade 3: Moderate multiple osteophytes, definite narrowing of joint space, some sclerosis and possible deformity of bone contour",
    4: "Grade 4: Large osteophytes, marked narrowing of joint space, severe sclerosis and definite deformity of bone contour"
}

# Dependency for getting the model
async def get_model():
    """Get the model as a dependency."""
    config = get_config()
    model_path = os.path.join(config.MODEL_SAVE_PATH, f"{config.MODEL_NAME}.pth")

    # Load model
    model = load_model(model_path)

    # Return model
    try:
        yield model
    finally:
        # Clean up (if needed)
        pass

@router.get("/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the model."""
    config = get_config()

    return {
        "name": config.MODEL_NAME,
        "version": config.MODEL_VERSION,
        "description": "Knee Osteoarthritis Classifier using the Kellgren-Lawrence grading system",
        "input_shape": [1] + config.IMAGE_SIZE,  # [channels, height, width]
        "num_classes": config.NUM_CLASSES,
        "classes": {i: desc for i, desc in KL_DESCRIPTIONS.items()},
        "architecture": config.MODEL_ARCHITECTURE
    }

@router.post("/predict", response_model=PredictionResult)
async def predict_image(
    file: UploadFile = File(...),
    model = Depends(get_model)
):
    """Predict the KL grade for a knee X-ray image."""
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("L")  # Convert to grayscale

        # Generate a unique ID for the image
        image_id = str(uuid.uuid4())

        # Make prediction
        result = predict_single_image(image, model)

        # Format response
        return {
            "image_id": image_id,
            "predicted_class": result["predicted_class"],
            "predicted_grade": result["predicted_grade"],
            "description": result["description"],
            "probabilities": result["probabilities"]
        }

    except Exception as e:
        logger.error(f"Error predicting image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@router.post("/predict/batch", response_model=List[PredictionResult])
async def predict_batch_images(
    files: List[UploadFile] = File(...),
    model = Depends(get_model)
):
    """Predict KL grades for multiple knee X-ray images."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    results = []

    # Process each file
    for file in files:
        # Validate file type
        if not file.content_type.startswith("image/"):
            logger.warning(f"Skipping non-image file: {file.filename}")
            continue

        try:
            # Read image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("L")

            # Generate a unique ID for the image
            image_id = str(uuid.uuid4())

            # Make prediction
            result = predict_single_image(image, model)

            # Format response
            results.append({
                "image_id": image_id,
                "predicted_class": result["predicted_class"],
                "predicted_grade": result["predicted_grade"],
                "description": result["description"],
                "probabilities": result["probabilities"]
            })

        except Exception as e:
            logger.error(f"Error predicting image {file.filename}: {str(e)}")
            # Continue with other images instead of failing the whole batch

    if not results:
        raise HTTPException(status_code=500, detail="Failed to process any images")

    return results

@router.post("/upload", response_model=Dict[str, str])
async def upload_and_save(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model = Depends(get_model)
):
    """Upload, save, and process a knee X-ray image."""
    config = get_config()

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Create upload directory if it doesn't exist
        upload_dir = os.path.join(config.UPLOAD_DIR, datetime.now().strftime("%Y%m%d"))
        os.makedirs(upload_dir, exist_ok=True)

        # Generate a unique filename
        file_id = str(uuid.uuid4())
        file_ext = os.path.splitext(file.filename)[1]
        filename = f"{file_id}{file_ext}"
        file_path = os.path.join(upload_dir, filename)

        # Save the file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process the image in the background
        background_tasks.add_task(process_image, file_path, model)

        return {
            "message": "File uploaded successfully",
            "file_id": file_id,
            "filename": filename,
            "status": "processing"
        }

    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

async def process_image(file_path: str, model):
    """Process an uploaded image in the background."""
    try:
        # Make prediction
        result = predict_single_image(file_path, model)

        # Save result
        result_path = f"{file_path}.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

        logger.info(f"Processed image: {file_path}")

    except Exception as e:
        logger.error(f"Error processing image {file_path}: {str(e)}")

@router.get("/results/{file_id}", response_model=Union[PredictionResult, Dict[str, str]])
async def get_prediction_result(file_id: str):
    """Get the prediction result for a previously uploaded image."""
    config = get_config()

    # Search for the file in the upload directory
    for root, _, files in os.walk(config.UPLOAD_DIR):
        for file in files:
            if file.startswith(file_id) and file.endswith(".json"):
                result_path = os.path.join(root, file)

                # Read the result
                with open(result_path, "r") as f:
                    result = json.load(f)

                # Format response
                return {
                    "image_id": file_id,
                    "predicted_class": result["predicted_class"],
                    "predicted_grade": result["predicted_grade"],
                    "description": result["description"],
                    "probabilities": result["probabilities"]
                }

    # Check if the file exists but hasn't been processed yet
    for root, _, files in os.walk(config.UPLOAD_DIR):
        for file in files:
            if file.startswith(file_id) and not file.endswith(".json"):
                return {"status": "processing", "message": "Image is still being processed"}

    raise HTTPException(status_code=404, detail=f"No results found for file ID: {file_id}")

@router.get("/image/{file_id}")
async def get_image(file_id: str):
    """Get the original image by file ID."""
    config = get_config()

    # Search for the file in the upload directory
    for root, _, files in os.walk(config.UPLOAD_DIR):
        for file in files:
            if file.startswith(file_id) and not file.endswith(".json"):
                file_path = os.path.join(root, file)
                return FileResponse(file_path)

    raise HTTPException(status_code=404, detail=f"Image not found for file ID: {file_id}")

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model": "knee_osteoarthritis_classifier"}