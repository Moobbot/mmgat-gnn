"""
Main API module.

This module sets up the FastAPI application and includes all routes from different models.
"""

import importlib
import logging
import logging.handlers
import os
import socket
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import get_system_config, FOLDERS, LOG_CONFIG, ENV, IS_DEV
from utils import get_system_metrics, get_local_ip

# Set up logging
logger = logging.getLogger("icta_api")
logger.setLevel(getattr(logging, LOG_CONFIG["LEVEL"]))

# File handler
if not os.path.exists(os.path.dirname(LOG_CONFIG["FILE"])):
    os.makedirs(os.path.dirname(LOG_CONFIG["FILE"]), exist_ok=True)

file_handler = logging.handlers.RotatingFileHandler(
    LOG_CONFIG["FILE"],
    maxBytes=LOG_CONFIG["MAX_BYTES"],
    backupCount=LOG_CONFIG["BACKUP_COUNT"],
)
file_handler.setFormatter(logging.Formatter(LOG_CONFIG["FORMAT"]))
logger.addHandler(file_handler)

# Console handler
if LOG_CONFIG["CONSOLE"]:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(LOG_CONFIG["FORMAT"]))
    logger.addHandler(console_handler)

# Get system config
config = get_system_config()

# Create FastAPI app
app = FastAPI(
    title=config.API_TITLE or "ICTA API",
    description=config.API_DESCRIPTION or "API for ICTA project",
    version=config.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=config.CORS_ALLOWED_METHODS,
    allow_headers=config.CORS_ALLOWED_HEADERS,
)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint that returns API information."""
    return {
        "name": config.API_TITLE or "ICTA API",
        "version": config.API_VERSION,
        "status": "running",
        "environment": ENV,
        "ip": get_local_ip(),
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "metrics": get_system_metrics(),
    }

# System information endpoint
@app.get("/system")
async def system_info():
    """Get system information."""
    return {
        "system": {
            "hostname": socket.gethostname(),
            "ip": get_local_ip(),
            "environment": ENV,
            "metrics": get_system_metrics(),
        },
        "api": {
            "title": config.API_TITLE,
            "version": config.API_VERSION,
            "host": config.HOST,
            "port": config.PORT,
            "active_models": config.ACTIVE_MODELS,
        },
        "folders": FOLDERS,
    }

# API prefix for all routes
api_prefix = config.API_PREFIX

# Import and include routes from active models
for model_name in config.ACTIVE_MODELS:
    try:
        # Import the routes module
        module_path = f"src.{model_name}.routes"
        routes_module = importlib.import_module(module_path)

        # Include the router
        if hasattr(routes_module, "router"):
            # Add API prefix to the router
            router = routes_module.router
            app.include_router(router, prefix=api_prefix)
            logger.info(f"Included routes from {model_name} with prefix {api_prefix}")
        else:
            logger.warning(f"No router found in {module_path}")
    except ImportError as e:
        logger.error(f"Error importing routes from {model_name}: {str(e)}")
    except Exception as e:
        logger.error(f"Error including routes from {model_name}: {str(e)}")

# Error handler for exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )

# Run the application
if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting API server on {config.HOST}:{config.PORT}")

    uvicorn.run(
        "api:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
    )