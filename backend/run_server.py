#!/usr/bin/env python3
"""
BeautyAI Inference API Server with proper WebSocket configuration.

This script starts the FastAPI server with optimized settings for large WebSocket frames.
"""
import uvicorn
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Start the BeautyAI API server with optimized WebSocket settings."""
    
    # Configure uvicorn with larger WebSocket limits
    config = uvicorn.Config(
        app="beautyai_inference.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(project_root)],
        log_level="info",
        # WebSocket configuration
        ws_max_size=50 * 1024 * 1024,  # 50MB max WebSocket frame size
        ws_ping_interval=20,
        ws_ping_timeout=20,
        # HTTP configuration  
        timeout_keep_alive=30,
        limit_max_requests=10000,
        # Performance settings
        workers=1,  # Single worker for development
        loop="asyncio",
        http="auto"
    )
    
    print("🚀 Starting BeautyAI Inference API Server")
    print("="*50)
    print(f"📡 Host: {config.host}:{config.port}")
    print(f"🔧 WebSocket Max Size: {config.ws_max_size / (1024*1024):.1f} MB")
    print(f"📚 API Docs: http://localhost:{config.port}/docs")
    print(f"🎤 WebSocket Voice: ws://localhost:{config.port}/ws/voice-conversation")
    print("="*50)
    
    # Start the server
    server = uvicorn.Server(config)
    server.run()

if __name__ == "__main__":
    main()
