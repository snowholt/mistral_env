#!/usr/bin/env python3
"""
API Server launcher for BeautyAI Inference Framework.

This script starts the FastAPI server with proper configuration for development and testing.
"""
import uvicorn
import logging
import sys
import os
from pathlib import Path

# Add the beautyai_inference module to the path
sys.path.insert(0, str(Path(__file__).parent))

def start_api_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = True,
    log_level: str = "info"
):
    """Start the BeautyAI API server."""
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🚀 Starting BeautyAI Inference API Server")
        logger.info(f"📡 Server will be available at: http://{host}:{port}")
        logger.info(f"📚 API Documentation: http://{host}:{port}/docs")
        logger.info(f"🔍 Alternative docs: http://{host}:{port}/redoc")
        
        # Start the server
        uvicorn.run(
            "beautyai_inference.api.app:app",
            host=host,
            port=port,
            reload=reload,
            log_level=log_level,
            access_log=True
        )
        
    except KeyboardInterrupt:
        logger.info("⚠️  Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        sys.exit(1)


def main():
    """Main function with command line argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Start BeautyAI API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    parser.add_argument("--log-level", default="info", 
                       choices=["debug", "info", "warning", "error"],
                       help="Log level")
    
    args = parser.parse_args()
    
    start_api_server(
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
        log_level=args.log_level
    )


if __name__ == "__main__":
    main()
