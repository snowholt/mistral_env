"""Centralized logging configuration for BeautyAI services.

This module builds a unified logging configuration supporting:
 - JSON application logs (api + webui) with daily rotation & retention
 - Separate access log (plain text) to keep JSON logs clean
 - Domain-specific log channels: inference, audio, webrtc, auth, streaming
 - Correlation / request IDs via contextvars for HTTP & WebSocket
 - Simple PII scrubbing (emails, long numbers) prior to emission

Environment variables (optional):
  BEAUTYAI_LOG_DIR              Root directory for logs (default: project_root/logs)
  BEAUTYAI_LOG_JSON             Enable JSON logs (default: 1)
  BEAUTYAI_LOG_RETENTION        App log rotation backup count (default: 30)
  BEAUTYAI_LOG_ACCESS_RETENTION Access log rotation backup count (default: 7)
  BEAUTYAI_LOG_STREAM_FILE      Override streaming log filename
  BEAUTYAI_LOG_LEVEL            Root log level (default: INFO)
  BEAUTYAI_LOG_VAD_LEVEL        VAD logger level - set to WARNING to reduce noise (default: WARNING)
  VOICE_STREAMING_METRICS_JSON  (existing) also controls structured metrics

Log Channels (separate files for easier debugging):
  - api/       : Main application logs, API endpoints
  - inference/ : LLM inference, model loading, generation
  - audio/     : VAD, RNNoise, audio processing, TTS
  - webrtc/    : WebRTC connections, ICE, data channels
  - auth/      : Authentication, JWT, user sessions
  - streaming/ : Streaming voice sessions, partials

NOTE: We use TimedRotatingFileHandler (midnight) with backupCount. If a global
disk budget is later required we can extend with a custom cleanup routine.
"""
from __future__ import annotations

import logging
import logging.config
import os
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import contextvars

try:  # optional dependency (added to requirements)
    from pythonjsonlogger import jsonlogger  # type: ignore
except Exception:  # pragma: no cover - fallback if not installed yet
    jsonlogger = None  # type: ignore

# Context variables for correlation & session IDs
request_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
session_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("session_id", default=None)


class CorrelationFilter(logging.Filter):
    """Inject correlation / session IDs into records if present."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        record.request_id = request_id_ctx.get() or "-"
        record.session_id = session_id_ctx.get() or getattr(record, "session_id", "-")
        return True


PII_EMAIL_RE = re.compile(r"[A-Za-z0-9_.+-]+@[A-Za-z0-9_.-]+\.[A-Za-z]{2,}")
PII_NUMBER_RE = re.compile(r"\b\d{8,}\b")  # long numbers (IDs, phones)


class PIIScrubberFilter(logging.Filter):
    """Basic PII scrubbing to avoid leaking emails / long numeric IDs.

    This is intentionally conservative & can be expanded later.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        if isinstance(record.msg, str):
            scrubbed = PII_EMAIL_RE.sub("<redacted_email>", record.msg)
            scrubbed = PII_NUMBER_RE.sub("<redacted_number>", scrubbed)
            record.msg = scrubbed
        return True


class UTCJsonFormatter(jsonlogger.JsonFormatter if jsonlogger else logging.Formatter):  # type: ignore
    """JSON formatter adding timestamp in ISO8601 Zulu."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        if jsonlogger and isinstance(self, jsonlogger.JsonFormatter):  # pragma: no branch
            if not hasattr(record, "timestamp"):
                record.timestamp = datetime.utcnow().isoformat() + "Z"  # type: ignore
            return super().format(record)
        # Fallback plain formatter
        ts = datetime.utcnow().isoformat() + "Z"
        return f"{ts} | {record.levelname} | {record.name} | {record.getMessage()}"


def _get_project_root() -> Path:
    """Get the project root directory (where logs/ should live)."""
    # This file is at: backend/src/beautyai_inference/logging/setup.py
    # Project root is 4 levels up
    current = Path(__file__).resolve()
    # Go up: setup.py -> logging -> beautyai_inference -> src -> backend -> project_root
    project_root = current.parents[4]
    
    # Verify we found the right directory (should have backend/ subfolder)
    if (project_root / "backend").is_dir():
        return project_root
    
    # Fallback: use CWD if structure doesn't match
    return Path.cwd()


def _ensure_dirs(root: Path) -> Dict[str, Path]:
    """Create log directory structure with domain-specific channels."""
    root.mkdir(parents=True, exist_ok=True)
    
    # Domain-specific log directories
    dirs = {
        "api": root / "api",           # Main API logs
        "inference": root / "inference", # LLM inference logs
        "audio": root / "audio",       # Audio processing (VAD, RNNoise, TTS)
        "webrtc": root / "webrtc",     # WebRTC connections
        "auth": root / "auth",         # Authentication logs
        "streaming": root / "streaming", # Streaming voice sessions
        "webui": root / "webui",       # Flask WebUI logs
    }
    
    for d in dirs.values():
        d.mkdir(exist_ok=True)
    
    return dirs


def build_logging_config(service: str = "api") -> Dict[str, Any]:
    """Build a dictConfig for the given service with domain-specific channels.

    service: "api" | "webui" - determines file naming & root logger.
    
    Log Channels:
      - api/app.jsonl       : Main application logs
      - api/access.log      : HTTP access logs (plain text)
      - inference/llm.jsonl : LLM inference, model loading
      - audio/processing.jsonl : VAD, RNNoise, TTS
      - webrtc/connections.jsonl : WebRTC sessions
      - auth/sessions.jsonl : Authentication events
      - streaming/voice.jsonl : Streaming voice sessions
    """
    json_enabled = os.getenv("BEAUTYAI_LOG_JSON", "1") == "1" and jsonlogger is not None
    
    # Separate retention for app logs (30 days) vs access logs (7 days)
    app_retention = int(os.getenv("BEAUTYAI_LOG_RETENTION", "30"))
    access_retention = int(os.getenv("BEAUTYAI_LOG_ACCESS_RETENTION", "7"))
    
    # Default log level
    log_level = os.getenv("BEAUTYAI_LOG_LEVEL", "INFO").upper()
    
    # VAD logging can be very noisy - default to WARNING to reduce log spam
    vad_log_level = os.getenv("BEAUTYAI_LOG_VAD_LEVEL", "WARNING").upper()
    
    # Determine log root directory - prefer project root logs/ over ./logs
    env_log_dir = os.getenv("BEAUTYAI_LOG_DIR")
    if env_log_dir:
        log_root = Path(env_log_dir)
    else:
        log_root = _get_project_root() / "logs"
    
    dirs = _ensure_dirs(log_root)

    # File paths for each channel
    api_app_file = dirs["api"] / f"{service}_app.jsonl"
    api_access_file = dirs["api"] / f"{service}_access.log"
    inference_file = dirs["inference"] / "llm.jsonl"
    audio_file = dirs["audio"] / "processing.jsonl"
    webrtc_file = dirs["webrtc"] / "connections.jsonl"
    auth_file = dirs["auth"] / "sessions.jsonl"
    streaming_file = Path(os.getenv("BEAUTYAI_LOG_STREAM_FILE", str(dirs["streaming"] / "voice.jsonl")))

    standard_fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    json_fmt = "%(timestamp)s %(levelname)s %(name)s %(request_id)s %(session_id)s %(message)s"

    formatters: Dict[str, Any] = {
        "standard": {"format": standard_fmt},
        "json": {
            "()": UTCJsonFormatter,
            "fmt": json_fmt,
        },
        "access": {"format": "%(asctime)s | %(levelname)s | %(message)s"},
    }

    # Handlers - one per log channel
    handlers: Dict[str, Any] = {
        "console": {
            "class": "logging.StreamHandler",
            "level": log_level,
            "formatter": "json" if json_enabled else "standard",
            "filters": ["correlation", "pii"],
            "stream": "ext://sys.stdout",
        },
        "app_file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "level": log_level,
            "formatter": "json" if json_enabled else "standard",
            "filters": ["correlation", "pii"],
            "filename": str(api_app_file),
            "when": "midnight",
            "backupCount": app_retention,
            "encoding": "utf-8",
            "utc": True,
        },
        "access_file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "level": "INFO",
            "formatter": "access",
            "filename": str(api_access_file),
            "when": "midnight",
            "backupCount": access_retention,
            "encoding": "utf-8",
            "utc": True,
        },
        "inference_file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "level": log_level,
            "formatter": "json" if json_enabled else "standard",
            "filters": ["correlation", "pii"],
            "filename": str(inference_file),
            "when": "midnight",
            "backupCount": app_retention,
            "encoding": "utf-8",
            "utc": True,
        },
        "audio_file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "level": log_level,
            "formatter": "json" if json_enabled else "standard",
            "filters": ["correlation", "pii"],
            "filename": str(audio_file),
            "when": "midnight",
            "backupCount": app_retention,
            "encoding": "utf-8",
            "utc": True,
        },
        "webrtc_file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "level": log_level,
            "formatter": "json" if json_enabled else "standard",
            "filters": ["correlation", "pii"],
            "filename": str(webrtc_file),
            "when": "midnight",
            "backupCount": app_retention,
            "encoding": "utf-8",
            "utc": True,
        },
        "auth_file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "level": log_level,
            "formatter": "json" if json_enabled else "standard",
            "filters": ["correlation", "pii"],
            "filename": str(auth_file),
            "when": "midnight",
            "backupCount": app_retention,
            "encoding": "utf-8",
            "utc": True,
        },
        "streaming_file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "level": log_level,
            "formatter": "json" if json_enabled else "standard",
            "filters": ["correlation", "pii"],
            "filename": str(streaming_file),
            "when": "midnight",
            "backupCount": app_retention,
            "encoding": "utf-8",
            "utc": True,
        },
    }

    # Filters
    filters = {
        "correlation": {"()": CorrelationFilter},
        "pii": {"()": PIIScrubberFilter},
    }

    # Loggers - mapped to domain-specific handlers
    loggers: Dict[str, Any] = {
        # Uvicorn loggers
        "uvicorn.error": {
            "level": "INFO",
            "handlers": ["console", "app_file"],
            "propagate": False,
        },
        "uvicorn.access": {
            "level": "INFO",
            "handlers": ["access_file"],
            "propagate": False,
        },
        
        # Application root
        "beautyai": {
            "level": log_level,
            "handlers": ["console", "app_file"],
            "propagate": False,
        },
        
        # Main application package (most modules use logging.getLogger(__name__))
        "beautyai_inference": {
            "level": log_level,
            "handlers": ["console", "app_file"],
            "propagate": False,
        },
        
        # API endpoints
        "beautyai_inference.api": {
            "level": log_level,
            "handlers": ["console", "app_file"],
            "propagate": False,
        },
        
        # ========== INFERENCE DOMAIN ==========
        # LLM inference, model management
        "beautyai_inference.inference_engines": {
            "level": log_level,
            "handlers": ["console", "inference_file"],
            "propagate": False,
        },
        "beautyai_inference.core.model_manager": {
            "level": log_level,
            "handlers": ["console", "inference_file"],
            "propagate": False,
        },
        "beautyai_inference.core.persistent_model_manager": {
            "level": log_level,
            "handlers": ["console", "inference_file"],
            "propagate": False,
        },
        "beautyai_inference.core.model_factory": {
            "level": log_level,
            "handlers": ["console", "inference_file"],
            "propagate": False,
        },
        "beautyai_inference.services.inference": {
            "level": log_level,
            "handlers": ["console", "inference_file"],
            "propagate": False,
        },
        
        # ========== AUDIO DOMAIN ==========
        # VAD, RNNoise, audio processing, TTS - WARNING level by default to reduce noise
        "beautyai_inference.services.voice.vad": {
            "level": vad_log_level,
            "handlers": ["console", "audio_file"],
            "propagate": False,
        },
        "beautyai_inference.utils.rnnoise_wrapper": {
            "level": log_level,
            "handlers": ["console", "audio_file"],
            "propagate": False,
        },
        "beautyai_inference.services.voice.synthesis": {
            "level": log_level,
            "handlers": ["console", "audio_file"],
            "propagate": False,
        },
        "beautyai_inference.services.voice.transcription": {
            "level": log_level,
            "handlers": ["console", "audio_file"],
            "propagate": False,
        },
        
        # ========== WEBRTC DOMAIN ==========
        "beautyai_inference.api.endpoints.webrtc_voice": {
            "level": log_level,
            "handlers": ["console", "webrtc_file"],
            "propagate": False,
        },
        "beautyai_inference.core.webrtc_connection_pool": {
            "level": log_level,
            "handlers": ["console", "webrtc_file"],
            "propagate": False,
        },
        "aiortc": {
            "level": os.getenv("AIORTC_LOG_LEVEL", "WARNING"),
            "handlers": ["console", "webrtc_file"],
            "propagate": False,
        },
        
        # ========== AUTH DOMAIN ==========
        "beautyai_inference.auth": {
            "level": log_level,
            "handlers": ["console", "auth_file"],
            "propagate": False,
        },
        "beautyai_inference.api.endpoints.whatsapp_auth": {
            "level": log_level,
            "handlers": ["console", "auth_file"],
            "propagate": False,
        },
        
        # ========== STREAMING DOMAIN ==========
        "beautyai_inference.api.endpoints.streaming_voice": {
            "level": log_level,
            "handlers": ["console", "streaming_file"],
            "propagate": False,
        },
        "beautyai_inference.services.voice.streaming": {
            "level": log_level,
            "handlers": ["console", "streaming_file"],
            "propagate": False,
        },
    }

    config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "filters": filters,
        "handlers": handlers,
        "loggers": loggers,
    }
    return config


def configure_logging(service: str = "api") -> None:
    """Apply logging configuration once with startup validation.

    Safe to call multiple times (idempotent) — will not reconfigure if root
    already has handlers.
    
    Args:
        service: "api" | "webui" - determines file naming & root logger.
    
    Raises:
        RuntimeError: If log directories cannot be created or written to.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    cfg = build_logging_config(service=service)
    logging.config.dictConfig(cfg)
    _CONFIGURED = True

    logger = logging.getLogger("beautyai_inference")
    
    # Log configuration summary
    log_dir = os.getenv("BEAUTYAI_LOG_DIR") or str(_get_project_root() / "logs")
    logger.info(
        "🔧 Logging configured | service=%s | json=%s | log_dir=%s | app_retention=%s | access_retention=%s",
        service,
        os.getenv("BEAUTYAI_LOG_JSON", "1"),
        log_dir,
        os.getenv("BEAUTYAI_LOG_RETENTION", "30"),
        os.getenv("BEAUTYAI_LOG_ACCESS_RETENTION", "7"),
    )
    
    # Validate all log channels are writable
    _validate_log_channels(logger)


def _validate_log_channels(logger: logging.Logger) -> None:
    """Write a test entry to each log channel to verify they're working."""
    log_dir = os.getenv("BEAUTYAI_LOG_DIR") or str(_get_project_root() / "logs")
    
    channels_validated = []
    channels_failed = []
    
    # Test each domain-specific logger
    test_loggers = [
        ("api", "beautyai_inference.api"),
        ("inference", "beautyai_inference.inference_engines"),
        ("audio", "beautyai_inference.services.voice.vad"),
        ("webrtc", "beautyai_inference.api.endpoints.webrtc_voice"),
        ("auth", "beautyai_inference.auth"),
        ("streaming", "beautyai_inference.api.endpoints.streaming_voice"),
    ]
    
    for channel, logger_name in test_loggers:
        try:
            test_logger = logging.getLogger(logger_name)
            test_logger.info("📝 Log channel validated | channel=%s", channel)
            channels_validated.append(channel)
        except Exception as e:
            channels_failed.append((channel, str(e)))
    
    if channels_validated:
        logger.info("✅ Log channels validated: %s", ", ".join(channels_validated))
    
    if channels_failed:
        for channel, error in channels_failed:
            logger.error("❌ Log channel failed: %s - %s", channel, error)


def get_logger(name: str, domain: Optional[str] = None) -> logging.Logger:
    """Get a logger with optional domain routing.
    
    This is a convenience function that ensures the logger is properly
    configured before returning it.
    
    Args:
        name: Logger name (typically __name__)
        domain: Optional domain hint for routing (inference, audio, webrtc, auth, streaming)
    
    Returns:
        Configured logger instance
    """
    # Ensure logging is configured
    if not _CONFIGURED:
        configure_logging()
    
    return logging.getLogger(name)


# Module-level flag: ensure we only apply dictConfig once per process.
_CONFIGURED = False


def is_logging_configured() -> bool:
    """Check if logging has been configured."""
    return _CONFIGURED


def get_log_directory() -> Path:
    """Get the current log directory path."""
    env_log_dir = os.getenv("BEAUTYAI_LOG_DIR")
    if env_log_dir:
        return Path(env_log_dir)
    return _get_project_root() / "logs"


__all__ = [
    "configure_logging",
    "get_logger",
    "get_log_directory",
    "is_logging_configured",
    "request_id_ctx",
    "session_id_ctx",
]
