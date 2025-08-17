# BeautyAI Logging Guide

This document describes the unified logging system added in August 2025.

## Overview

Components now emit structured and plain logs into `./logs` (configurable via `BEAUTYAI_LOG_DIR`).

Directory layout:
```
logs/
  api/
    api_app.jsonl        # JSON (or plain) application logs for API service
    api_access.log       # Plain text access logs (HTTP)
  streaming/
    streaming_voice.jsonl # Streaming voice performance + events
  webui/                 # (Reserved for future web UI logs)
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| BEAUTYAI_LOG_DIR | ./logs | Root directory for all logs |
| BEAUTYAI_LOG_JSON | 1 | Enable JSON formatting (requires python-json-logger) |
| BEAUTYAI_LOG_RETENTION | 7 | Daily rotation backup count (TimedRotatingFileHandler) |
| BEAUTYAI_LOG_STREAM_FILE | (auto) | Override streaming voice log path |
| VOICE_STREAMING_METRICS_JSON | 0 | When 1, emits structured metrics snapshots |

## Handlers

1. Application (api_app.jsonl) – INFO+ application/service events.
2. Access (api_access.log) – UVicorn access log (kept separate for cleaner machine parsing of app log).
3. Streaming (streaming_voice.jsonl) – Dedicated channel for voice WS events & performance.

## Correlation & Session IDs

- Each HTTP/WebSocket request gets an `X-Request-ID` header and is added to log records as `request_id`.
- Streaming voice sessions bind `session_id` for the lifetime of the connection.

## PII Scrubbing

Basic filters replace:
- Email addresses with `<redacted_email>`
- Long numeric sequences (>=8 digits) with `<redacted_number>`

Extendable in `beautyai_inference/logging/setup.py`.

## Tail Helper

Use the helper script:
```bash
./tools/tail_logs.sh        # show recent slices
./tools/tail_logs.sh follow # follow all three files
```

## Rotation

Daily at midnight UTC with `backupCount=BEAUTYAI_LOG_RETENTION` (default 7). For stricter disk caps, integrate system logrotate or a cleanup cron.

## Adding Structured Events

Use the existing pattern (example for streaming metrics):
```python
from beautyai_inference.services.voice.streaming.metrics import maybe_log_structured
maybe_log_structured(logger, "voice_stream_perf_cycle", payload_dict)
```
Ensure `VOICE_STREAMING_METRICS_JSON=1` to enable emission.

## Future Enhancements (Suggested)
- WebUI logging bootstrap (parallel to API) pulling same setup module.
- Prometheus exporter for request / pipeline latency.
- Extended PII scrubbing (names, partial hashes) with allowlist.
- Log schema version stamping (e.g., `schema_version: 1`).
- Async queue handler for very high throughput streaming.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| No JSON formatting | python-json-logger missing | `pip install python-json-logger` or ensure in requirements (already added) |
| Empty streaming log | Feature disabled or no sessions | Confirm `VOICE_STREAMING_ENABLED=1` |
| Request ID missing | Middleware not attached | Ensure `CorrelationIdMiddleware` is in app (already configured) |
| Session ID missing in records | Outside WS lifecycle | Confirm inside `streaming_voice_endpoint` logic |

## Minimal Example Log Entry (JSON mode)
```json
{"timestamp":"2025-08-17T12:34:56.789Z","levelname":"INFO","name":"beautyai_inference.api.endpoints.streaming_voice","request_id":"a1b2c3d4e5f6","session_id":"stream_1234","message":"🌐 Streaming voice connection established: ..."}
```

---
For questions or enhancements, update this doc and extend `logging/setup.py`.
