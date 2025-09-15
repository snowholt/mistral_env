"""
Environment-aware configuration constants for BeautyAI Frontend.

Supports dynamic configuration based on environment variables
and deployment contexts.
"""

import os

# Environment detection
ENVIRONMENT = os.getenv("BEAUTYAI_ENV", "development")
DEBUG = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
PRODUCTION = os.getenv("PRODUCTION", "false").lower() in ("true", "1", "yes")

# Backend connection settings
BACKEND_HOST = os.getenv("BACKEND_HOST", "localhost")
BACKEND_PORT = os.getenv("BACKEND_PORT", "8000")
BACKEND_PROTOCOL = os.getenv("BACKEND_PROTOCOL", "http")

# WebSocket configuration - critical for production deployment
BACKEND_WS_HOST = os.getenv("BACKEND_WS_HOST", BACKEND_HOST)
BACKEND_WS_PROTOCOL = os.getenv("BACKEND_WS_PROTOCOL", "ws")

# Production overrides for WebSocket configuration
if PRODUCTION or ENVIRONMENT == "production":
    # In production, use WSS and the API domain for reliable connection
    # This matches the logic in debug_pcm_upload.html that works correctly
    BACKEND_WS_PROTOCOL = os.getenv("BACKEND_WS_PROTOCOL", "wss")
    BACKEND_WS_HOST = os.getenv("BACKEND_WS_HOST", "api.gmai.sa")  # Use api.gmai.sa instead of dev.gmai.sa
    BACKEND_WS_PORT = ""  # No port for standard WSS/HTTPS
else:
    # Development settings
    BACKEND_WS_HOST = BACKEND_HOST
    BACKEND_WS_PORT = f":{BACKEND_PORT}"

# API configuration
API_PREFIX = os.getenv("API_PREFIX", "/api/v1")
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))

# WebSocket endpoints
WEBSOCKET_SIMPLE_VOICE_PATH = "/api/v1/ws/simple-voice-chat"
WEBSOCKET_STREAMING_VOICE_PATH = "/api/v1/ws/streaming-voice"

# Dynamic backend URLs
BACKEND_BASE_URL = f"{BACKEND_PROTOCOL}://{BACKEND_HOST}:{BACKEND_PORT}"
BACKEND_API_URL = f"{BACKEND_BASE_URL}{API_PREFIX}"

# WebSocket URLs - properly handle production vs development
if PRODUCTION or ENVIRONMENT == "production":
    # Production WebSocket URLs (no port for standard HTTPS/WSS)
    BACKEND_SIMPLE_VOICE_WS = f"{BACKEND_WS_PROTOCOL}://{BACKEND_WS_HOST}{WEBSOCKET_SIMPLE_VOICE_PATH}"
    BACKEND_STREAMING_VOICE_WS = f"{BACKEND_WS_PROTOCOL}://{BACKEND_WS_HOST}{WEBSOCKET_STREAMING_VOICE_PATH}"
else:
    # Development WebSocket URLs (with port)
    BACKEND_SIMPLE_VOICE_WS = f"{BACKEND_WS_PROTOCOL}://{BACKEND_WS_HOST}:{BACKEND_PORT}{WEBSOCKET_SIMPLE_VOICE_PATH}"
    BACKEND_STREAMING_VOICE_WS = f"{BACKEND_WS_PROTOCOL}://{BACKEND_WS_HOST}:{BACKEND_PORT}{WEBSOCKET_STREAMING_VOICE_PATH}"

# Model configuration
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen3-unsloth-q4ks")
DEFAULT_ASR_MODEL = os.getenv("DEFAULT_ASR_MODEL", "openai/whisper-large-v3")
DEFAULT_TTS_MODEL = os.getenv("DEFAULT_TTS_MODEL", "edge-tts")

# Audio configuration
DEFAULT_SAMPLE_RATE = int(os.getenv("DEFAULT_SAMPLE_RATE", "16000"))
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "320"))

# Duplex streaming configuration
DUPLEX_ENABLED = os.getenv("DUPLEX_ENABLED", "true").lower() in ("true", "1", "yes")
ECHO_CANCELLATION_ENABLED = os.getenv("ECHO_CANCELLATION", "true").lower() in ("true", "1", "yes")
BARGE_IN_ENABLED = os.getenv("BARGE_IN_ENABLED", "true").lower() in ("true", "1", "yes")

# UI configuration
MAX_TRANSCRIPT_LENGTH = int(os.getenv("MAX_TRANSCRIPT_LENGTH", "10000"))
MAX_CONVERSATION_TURNS = int(os.getenv("MAX_CONVERSATION_TURNS", "50"))
UI_REFRESH_RATE_MS = int(os.getenv("UI_REFRESH_RATE_MS", "100"))

# Feature flags
ENABLE_METRICS = os.getenv("ENABLE_METRICS", "true").lower() in ("true", "1", "yes")
ENABLE_DEVICE_SELECTION = os.getenv("ENABLE_DEVICE_SELECTION", "true").lower() in ("true", "1", "yes")
ENABLE_ECHO_TEST = os.getenv("ENABLE_ECHO_TEST", "true").lower() in ("true", "1", "yes")

# Performance configuration
CONNECTION_TIMEOUT_MS = int(os.getenv("CONNECTION_TIMEOUT_MS", "5000"))
RECONNECT_ATTEMPTS = int(os.getenv("RECONNECT_ATTEMPTS", "3"))
RECONNECT_DELAY_MS = int(os.getenv("RECONNECT_DELAY_MS", "1000"))

# Environment-specific overrides
if ENVIRONMENT == "production" or PRODUCTION:
    # Production settings
    API_TIMEOUT = int(os.getenv("API_TIMEOUT", "60"))
    CONNECTION_TIMEOUT_MS = int(os.getenv("CONNECTION_TIMEOUT_MS", "10000"))
    RECONNECT_ATTEMPTS = int(os.getenv("RECONNECT_ATTEMPTS", "5"))
    RECONNECT_DELAY_MS = int(os.getenv("RECONNECT_DELAY_MS", "2000"))
elif ENVIRONMENT == "development":
    # Development settings
    DEBUG = True
    CONNECTION_TIMEOUT_MS = int(os.getenv("CONNECTION_TIMEOUT_MS", "5000"))
    RECONNECT_ATTEMPTS = int(os.getenv("RECONNECT_ATTEMPTS", "3"))
    RECONNECT_DELAY_MS = int(os.getenv("RECONNECT_DELAY_MS", "1000"))
elif ENVIRONMENT == "testing":
    # Testing settings
    BACKEND_HOST = "localhost"
    BACKEND_PORT = "8001"  # Testing port

# Configuration validation
def validate_config():
    """Validate configuration settings."""
    errors = []
    
    if not BACKEND_HOST:
        errors.append("BACKEND_HOST is required")
    
    if not BACKEND_PORT.isdigit():
        errors.append("BACKEND_PORT must be numeric")
    
    if API_TIMEOUT <= 0:
        errors.append("API_TIMEOUT must be positive")
    
    if DEFAULT_SAMPLE_RATE not in [8000, 16000, 22050, 44100, 48000]:
        errors.append(f"Invalid DEFAULT_SAMPLE_RATE: {DEFAULT_SAMPLE_RATE}")
    
    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")

# Validate on import
validate_config()

# Export configuration dictionary for JavaScript
CONFIG_DICT = {
    "environment": ENVIRONMENT,
    "debug": DEBUG,
    "production": PRODUCTION,
    "backend": {
        "host": BACKEND_HOST,
        "port": BACKEND_PORT,
        "protocol": BACKEND_PROTOCOL,
        "base_url": BACKEND_BASE_URL,
        "api_url": BACKEND_API_URL,
        "ws_protocol": BACKEND_WS_PROTOCOL,
        "ws_host": BACKEND_WS_HOST,
        "simple_voice_ws": BACKEND_SIMPLE_VOICE_WS,
        "streaming_voice_ws": BACKEND_STREAMING_VOICE_WS,
    },
    "models": {
        "default_model": DEFAULT_MODEL,
        "default_asr_model": DEFAULT_ASR_MODEL,
        "default_tts_model": DEFAULT_TTS_MODEL,
    },
    "audio": {
        "sample_rate": DEFAULT_SAMPLE_RATE,
        "chunk_size": DEFAULT_CHUNK_SIZE,
    },
    "duplex": {
        "enabled": DUPLEX_ENABLED,
        "echo_cancellation": ECHO_CANCELLATION_ENABLED,
        "barge_in_enabled": BARGE_IN_ENABLED,
    },
    "ui": {
        "max_transcript_length": MAX_TRANSCRIPT_LENGTH,
        "max_conversation_turns": MAX_CONVERSATION_TURNS,
        "refresh_rate_ms": UI_REFRESH_RATE_MS,
    },
    "features": {
        "metrics": ENABLE_METRICS,
        "device_selection": ENABLE_DEVICE_SELECTION,
        "echo_test": ENABLE_ECHO_TEST,
    },
    "performance": {
        "api_timeout": API_TIMEOUT,
        "connection_timeout_ms": CONNECTION_TIMEOUT_MS,
        "reconnect_attempts": RECONNECT_ATTEMPTS,
        "reconnect_delay_ms": RECONNECT_DELAY_MS,
    }
}
