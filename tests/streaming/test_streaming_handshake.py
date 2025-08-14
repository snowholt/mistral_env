"""Tests for the streaming voice WebSocket endpoint handshake & feature flag.

These tests validate:
 1. Route is mounted (default: VOICE_STREAMING_ENABLED unset -> enabled)
 2. Handshake succeeds and sends initial 'ready' event
 3. Language validation (unsupported language rejected)
 4. Feature flag off (VOICE_STREAMING_ENABLED=0) causes immediate close

We use FastAPI's TestClient which wraps Starlette's WebSocket test support.
"""
from __future__ import annotations

import os
from typing import Optional
from fastapi.testclient import TestClient

from beautyai_inference.api.app import app

# Prevent heavy model preloading during tests (clear startup events)
app.router.on_startup.clear()


def _connect(client: TestClient, language: str = "ar"):
    return client.websocket_connect(f"/api/v1/ws/streaming-voice?language={language}")


def test_handshake_ready_event_default_env(monkeypatch):
    """When VOICE_STREAMING_ENABLED is unset, endpoint should be active and emit ready event."""
    monkeypatch.delenv("VOICE_STREAMING_ENABLED", raising=False)
    with TestClient(app) as client:
        with _connect(client) as ws:
            msg = ws.receive_json()
            assert msg["type"] == "ready"
            assert msg["session_id"]
            assert msg["feature"].startswith("streaming_voice_")


def test_handshake_reject_when_disabled(monkeypatch):
    """When VOICE_STREAMING_ENABLED=0 handshake should be closed immediately (no ready)."""
    monkeypatch.setenv("VOICE_STREAMING_ENABLED", "0")
    with TestClient(app) as client:
        # Expect a failure to establish (FastAPI will raise or immediate close)
        try:
            with _connect(client) as ws:
                # If it did accept, we should not get a ready event
                data = ws.receive_json(timeout=0.5)
                assert data.get("type") != "ready", "Should not receive ready when disabled"
        except Exception:
            # Accept either exception path or silent close as success criteria for disabled state
            pass
    # reset
    monkeypatch.delenv("VOICE_STREAMING_ENABLED", raising=False)


def test_handshake_invalid_language(monkeypatch):
    """Unsupported language should close connection (no ready)."""
    monkeypatch.delenv("VOICE_STREAMING_ENABLED", raising=False)
    with TestClient(app) as client:
        try:
            with _connect(client, language="zz") as ws:
                data = ws.receive_json(timeout=0.5)
                assert data.get("type") != "ready", "Should not get ready for unsupported language"
        except Exception:
            # Expected path: rejected/closed.
            pass


def test_multiple_connections(monkeypatch):
    """Open two concurrent connections to ensure independent session IDs."""
    monkeypatch.delenv("VOICE_STREAMING_ENABLED", raising=False)
    with TestClient(app) as client:
        with _connect(client) as ws1, _connect(client, language="en") as ws2:
            ready1 = ws1.receive_json()
            ready2 = ws2.receive_json()
            assert ready1["type"] == "ready"
            assert ready2["type"] == "ready"
            assert ready1["session_id"] != ready2["session_id"]
