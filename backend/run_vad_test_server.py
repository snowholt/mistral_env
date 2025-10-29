#!/usr/bin/env python3
"""
VAD Test Server - Minimal FastAPI server for WebRTC + VAD testing

This server loads the existing WebRTC infrastructure but uses mock models
instead of real Whisper/LLM/TTS models. Perfect for testing VAD behavior
without heavy model dependencies.

Usage:
    VAD_TEST_MODE=1 python backend/run_vad_test_server.py
    
    Or with custom config:
    VAD_TEST_MODE=1 python backend/run_vad_test_server.py --config config/config.vad_test.yaml

Author: BeautyAI Framework
Date: October 29, 2025
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add backend src to path
backend_root = Path(__file__).parent
src_path = backend_root / "src"
sys.path.insert(0, str(src_path))

# Set VAD test mode environment variable
os.environ["VAD_TEST_MODE"] = "1"
os.environ["BEAUTYAI_VAD_DEBUG"] = "1"

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import configuration
from beautyai_inference.core.config_manager import get_config_manager

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/vad_test_server.log', mode='w')
    ]
)

logger = logging.getLogger(__name__)


def create_vad_test_app(config_path: str = "config/config.vad_test.yaml") -> FastAPI:
    """
    Create FastAPI application for VAD testing.
    
    Loads existing WebRTC infrastructure with mock models.
    """
    logger.info("="*80)
    logger.info("🧪 VAD TEST SERVER - Starting in Mock Mode")
    logger.info("="*80)
    logger.info(f"Config: {config_path}")
    logger.info(f"VAD_TEST_MODE: {os.getenv('VAD_TEST_MODE')}")
    logger.info(f"BEAUTYAI_VAD_DEBUG: {os.getenv('BEAUTYAI_VAD_DEBUG')}")
    logger.info("="*80)
    
    # Create FastAPI app
    app = FastAPI(
        title="BeautyAI VAD Test Server",
        description="WebRTC + VAD Testing Server (Mock Models)",
        version="1.0.0-vad-test",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Load configuration
    try:
        config_manager = get_config_manager()
        config = config_manager.load_config(config_path)
        logger.info(f"✅ Configuration loaded from {config_path}")
        logger.info(f"   - WebRTC enabled: {config.get('webrtc', {}).get('enabled', False)}")
        logger.info(f"   - VAD debug: {config.get('webrtc', {}).get('debug_logging', False)}")
        logger.info(f"   - Mock models: {config.get('vad_test', {}).get('use_mock_models', False)}")
    except Exception as e:
        logger.warning(f"⚠️ Could not load config from {config_path}: {e}")
        logger.info("Continuing with environment variables...")
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "mode": "vad_test",
            "vad_test_mode": os.getenv("VAD_TEST_MODE", "0"),
            "mock_models": True
        }
    
    # VAD test info endpoint
    @app.get("/vad/info")
    async def vad_info():
        """Get VAD test configuration info."""
        return {
            "test_mode": "enabled",
            "mock_models": {
                "whisper": "MockWhisperModel",
                "llm": "MockLLMModel",
                "tts": "MockTTSModel"
            },
            "vad_config": {
                "dual_mode": os.getenv("VAD_DUAL_MODE", "true"),
                "webrtc_sensitivity": 2,
                "silero_threshold": 0.3,
                "debug_enabled": True
            }
        }
    
    # Import and register WebRTC endpoints
    try:
        from beautyai_inference.api.endpoints.webrtc_voice import webrtc_voice_router
        
        app.include_router(
            webrtc_voice_router,
            tags=["webrtc-vad-test"]
        )
        logger.info("✅ WebRTC voice endpoints registered")
        
    except Exception as e:
        logger.error(f"❌ Failed to register WebRTC endpoints: {e}")
        logger.exception(e)
    
    # Startup event
    @app.on_event("startup")
    async def startup_event():
        """Initialize services on startup."""
        logger.info("🚀 VAD Test Server startup")
        
        # Initialize WebRTC connection pool
        try:
            from beautyai_inference.core.webrtc_connection_pool import initialize_webrtc_pool
            await initialize_webrtc_pool()
            logger.info("✅ WebRTC connection pool initialized")
        except Exception as e:
            logger.warning(f"⚠️ WebRTC pool initialization warning: {e}")
        
        # Log mock model status
        from beautyai_inference.services.voice.mock_models import is_vad_test_mode
        if is_vad_test_mode():
            logger.info("✅ Mock models enabled (Whisper/LLM/TTS)")
        else:
            logger.warning("⚠️ Mock models NOT enabled - check VAD_TEST_MODE")
        
        logger.info("✅ VAD Test Server ready!")
        logger.info(f"📡 WebRTC signaling: http://localhost:8000/api/v1/webrtc/voice")
        logger.info(f"📊 API docs: http://localhost:8000/docs")
    
    # Shutdown event
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        logger.info("🛑 VAD Test Server shutdown")
        
        try:
            from beautyai_inference.core.webrtc_connection_pool import shutdown_webrtc_pool
            await shutdown_webrtc_pool()
            logger.info("✅ WebRTC connection pool shut down")
        except Exception as e:
            logger.warning(f"⚠️ WebRTC pool shutdown warning: {e}")
    
    return app


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="BeautyAI VAD Test Server"
    )
    parser.add_argument(
        "--config",
        default="config/config.vad_test.yaml",
        help="Path to VAD test configuration file"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    args = parser.parse_args()
    
    # Ensure logs directory exists
    logs_dir = Path("logs/webrtc/vad_debug")
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create app
    app = create_vad_test_app(args.config)
    
    # Run server
    logger.info(f"🚀 Starting VAD Test Server on {args.host}:{args.port}")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="debug",
        access_log=True
    )


if __name__ == "__main__":
    main()
