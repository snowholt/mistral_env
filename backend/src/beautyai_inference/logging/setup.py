"""Centralized logging configuration for BeautyAI services.

This module builds a unified logging configuration supporting:
 - JSON application logs (api + webui) with daily rotation & retention
 - Separate access log (plain text) to keep JSON logs clean
 - Dedicated streaming voice performance log channel (JSON)
 - Correlation / request IDs via contextvars for HTTP & WebSocket
 - Simple PII scrubbing (emails, long numbers) prior to emission

Environment variables (optional):
  BEAUTYAI_LOG_DIR           Root directory for logs (default: ./logs)
  BEAUTYAI_LOG_JSON          Enable JSON logs (default: 1)
  BEAUTYAI_LOG_RETENTION     Timed rotation backup count (default: 7)
  BEAUTYAI_LOG_STREAM_FILE   Override streaming log filename
  VOICE_STREAMING_METRICS_JSON  (existing) also controls structured metrics

NOTE: We use TimedRotatingFileHandler (midnight) with backupCount. If a global
disk budget is later required we can extend with a custom cleanup routine.
"""
from __future__ import annotations

import logging
import logging.config
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
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


def _ensure_dirs(root: Path) -> Dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    api_dir = root / "api"
    streaming_dir = root / "streaming"
    webui_dir = root / "webui"
    for d in (api_dir, streaming_dir, webui_dir):
        d.mkdir(exist_ok=True)
    return {"api": api_dir, "streaming": streaming_dir, "webui": webui_dir}


def build_logging_config(service: str = "api") -> Dict[str, Any]:
    """Build a dictConfig for the given service.

    service: "api" | "webui" - determines file naming & root logger.
    """
    json_enabled = os.getenv("BEAUTYAI_LOG_JSON", "1") == "1" and jsonlogger is not None
    retention = int(os.getenv("BEAUTYAI_LOG_RETENTION", "7"))
    log_root = Path(os.getenv("BEAUTYAI_LOG_DIR", "./logs"))
    dirs = _ensure_dirs(log_root)

    api_app_file = dirs["api"] / f"{service}_app.jsonl"
    api_access_file = dirs["api"] / f"{service}_access.log"
    streaming_file = Path(os.getenv("BEAUTYAI_LOG_STREAM_FILE", str(dirs["streaming"] / "streaming_voice.jsonl")))

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

    # Handlers
    handlers: Dict[str, Any] = {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "json" if json_enabled else "standard",
            "filters": ["correlation", "pii"],
            "stream": "ext://sys.stdout",
        },
        "app_file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "level": "INFO",
            "formatter": "json" if json_enabled else "standard",
            "filters": ["correlation", "pii"],
            "filename": str(api_app_file),
            "when": "midnight",
            "backupCount": retention,
            "encoding": "utf-8",
            "utc": True,
        },
        "access_file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "level": "INFO",
            "formatter": "access",
            "filename": str(api_access_file),
            "when": "midnight",
            "backupCount": retention,
            "encoding": "utf-8",
            "utc": True,
        },
        "streaming_file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "level": "INFO",
            "formatter": "json" if json_enabled else "standard",
            "filters": ["correlation", "pii"],
            "filename": str(streaming_file),
            "when": "midnight",
            "backupCount": retention,
            "encoding": "utf-8",
            "utc": True,
        },
    }

    # Filters
    filters = {
        "correlation": {"()": CorrelationFilter},
        "pii": {"()": PIIScrubberFilter},
    }

    # Loggers
    loggers: Dict[str, Any] = {
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
            "level": "INFO",
            "handlers": ["console", "app_file"],
            "propagate": False,
        },
        # Dedicated streaming voice channel (module path may differ; using prefix)
        "beautyai_inference.api.endpoints.streaming_voice": {
            "level": "INFO",
            "handlers": ["console", "streaming_file"],
            "propagate": False,
        },
        "beautyai_inference.services.voice.streaming": {
            "level": "INFO",
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
    """Apply logging configuration once.

    Safe to call multiple times (idempotent) — will not reconfigure if root
    already has handlers.
    """
    root_logger = logging.getLogger()
    if root_logger.handlers:  # Already configured
        return
    cfg = build_logging_config(service=service)
    logging.config.dictConfig(cfg)
    logging.getLogger("beautyai").info("Logging configured (service=%s, json=%s)", service, os.getenv("BEAUTYAI_LOG_JSON", "1"))


__all__ = [
    "configure_logging",
    "request_id_ctx",
    "session_id_ctx",
]
