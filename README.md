# ICTA 2025 API Project

This repository tests several models for knee osteoarthritis classification using deep learning and provides an API for model inference.

## Project Structure

The project is organized with the following structure:

```bash
icta_2025_test/
├── src/
│   └── model_1/                # AI/ML module for knee osteoarthritis classification
│       ├── __init__.py
│       ├── config.py           # Model-specific configuration
│       ├── data.py             # Data loading and processing
│       ├── evaluation.py       # Model evaluation utilities
│       ├── model.py            # Model architecture definition
│       ├── predict.py          # Prediction functions
│       ├── preprocessing.py    # Image preprocessing
│       ├── routes.py           # API routes for the model
│       ├── utils.py            # Model-specific utilities
│       └── visualization.py    # Result visualization
├── api.py                      # Main API entry point
├── config.py                   # System-wide configuration
├── utils.py                    # System-wide utilities
├── setup.py                    # Installation and setup script
├── requirements.txt            # Package dependencies
├── .env.example                # Environment variables template
└── README.md                   # This file
```

## Installation

### Step 1: Clone the repository

```bash
git clone <repository-url>
cd icta_2025_test
```

### Step 2: Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install the project

```bash
python setup.py
```

This will automatically:

- Detect if you have a GPU and install the appropriate PyTorch version
- Install all required dependencies from requirements.txt
- Set up the project environment

### Step 4: Configure environment variables

```bash
cp .env.example .env
```

Edit the `.env` file with your configuration settings.

## Running the API

To run the API locally:

```bash
uvicorn api:app --reload
```

The API will be available at <http://localhost:8000>

API documentation is available at:

- Swagger UI: <http://localhost:8000/docs>
- ReDoc: <http://localhost:8000/redoc>

## Features

- **GPU Acceleration**: Automatic detection and utilization of GPU for faster inference
- **Mixed Precision**: Support for mixed precision to optimize performance
- **RESTful API**: Clean API endpoints for model inference
- **Modular Design**: Easy to add new AI/ML models to the system
- **Environment Configuration**: Flexible configuration through environment variables

## Development

To install the package in development mode:

```bash
pip install -e .
```

## Testing

To run tests:

```bash
pytest
```

## GPU Support

The system automatically detects if a GPU is available and configures PyTorch accordingly. You can control GPU usage through the following environment variables:

- `USE_GPU`: Set to `True` or `False` to enable or disable GPU usage
- `GPU_DEVICE`: Specify which GPU device to use (e.g., `0` for the first GPU)
- `CUDA_VISIBLE_DEVICES`: Control which GPUs are visible to the application
- `MIXED_PRECISION`: Enable or disable mixed precision for better performance
