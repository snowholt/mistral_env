"""
FastAPI Application for BeautyAI Inference Framework.

This module provides a REST API interface for the BeautyAI inference framework,
including endpoints for model management, chat interactions, and system monitoring.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import logging

# Import the endpoints
from .endpoints import health, models, inference, config, system

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="BeautyAI Inference API",
    description="Professional-grade API for running inference with various language models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(models.router, prefix="/models", tags=["Models"])
app.include_router(inference.router, prefix="/inference", tags=["Inference"])
app.include_router(config.router, prefix="/config", tags=["Configuration"])
app.include_router(system.router, prefix="/system", tags=["System"])


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("🚀 BeautyAI Inference API starting up...")
    logger.info("📚 API Documentation available at: http://localhost:8000/docs")
    logger.info("🔍 Alternative docs at: http://localhost:8000/redoc")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("🛑 BeautyAI Inference API shutting down...")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "BeautyAI Inference API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "inference": "/inference",
            "config": "/config",
            "system": "/system"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
