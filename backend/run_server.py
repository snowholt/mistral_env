#!/usr/bin/env python3
"""
BeautyAI Inference API Server with proper WebSocket configuration.

This script starts the FastAPI server with optimized settings for large WebSocket frames.
"""
import uvicorn
import os
import sys
import socket
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def _env_flag(name: str, default: str = "0") -> str:
    """Return normalized boolean-ish environment variable (string)."""
    val = os.getenv(name, default).strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return "1"
    return "0"


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def main():
    """Start the BeautyAI API server with optimized WebSocket settings.

    Changes:
      - Reload can now be disabled for systemd (was hard-coded True causing orphan workers).
      - Respect env vars: UVICORN_RELOAD, BEAUTYAI_ENV, BEAUTYAI_HOST, BEAUTYAI_PORT.
      - Detect occupied port early to produce a clearer log + non-zero exit.
      - Show streaming voice endpoint banner when feature enabled.
    """

    host = os.getenv("BEAUTYAI_HOST", "0.0.0.0")
    port = int(os.getenv("BEAUTYAI_PORT", "8000"))

    # Decide reload mode:
    # Priority: explicit UVICORN_RELOAD > production env disables by default > interactive dev enables.
    reload_env = os.getenv("UVICORN_RELOAD")
    if reload_env is None:
        is_prod = os.getenv("BEAUTYAI_ENV", "development").lower() in {"prod", "production"}
        # Disable reload when not attached to a TTY (e.g., systemd) or in production
        reload_enabled = (not is_prod) and sys.stdout.isatty()
    else:
        reload_enabled = _env_flag("UVICORN_RELOAD", "0") == "1"

    if _port_in_use(host, port):
        # Provide a clear diagnostic before uvicorn tries (helps systemd logs)
        print(f"❌ Port {port} already in use on {host}. Refusing to start (reload={reload_enabled}).", flush=True)
        sys.exit(1)

    # Configure uvicorn with larger WebSocket limits
    config = uvicorn.Config(
        app="beautyai_inference.api.app:app",
        host=host,
        port=port,
        reload=reload_enabled,
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
        workers=1,  # Single worker for development / systemd manages the process
        loop="asyncio",
        http="auto",
    )

    streaming_enabled = _env_flag("VOICE_STREAMING_ENABLED", "1") == "1"
    phase4 = _env_flag("VOICE_STREAMING_PHASE4", "0") == "1"
    streaming_path = "/api/v1/ws/streaming-voice"

    print("🚀 Starting BeautyAI Inference API Server")
    print("=" * 60)
    print(f"📡 Host: {config.host}:{config.port}")
    print(f"🔄 Reload: {'enabled' if config.reload else 'disabled'}")
    print(f"🔧 WebSocket Max Size: {config.ws_max_size / (1024 * 1024):.1f} MB")
    print(f"📚 API Docs: http://localhost:{config.port}/docs")
    print(f"📘 Redoc: http://localhost:{config.port}/redoc")
    print(f"🎤 Legacy Voice WS: ws://localhost:{config.port}/ws/voice-conversation")
    if streaming_enabled:
        print(f"🌊 Streaming Voice WS: ws://localhost:{config.port}{streaming_path} (phase={'4' if phase4 else '2'})")
    else:
        print("🌊 Streaming Voice WS: disabled (set VOICE_STREAMING_ENABLED=1 to enable)")
    print("=" * 60)

    server = uvicorn.Server(config)
    server.run()

if __name__ == "__main__":
    main()
