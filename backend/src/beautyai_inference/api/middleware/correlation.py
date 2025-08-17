"""Correlation & Request ID middleware for HTTP + WebSocket.

Generates a short request ID for every inbound HTTP request & WebSocket
connection and binds it to a contextvar so logging filters can attach it.
"""
from __future__ import annotations

import uuid
import time
from starlette.middleware.base import BaseHTTPMiddleware
from ...logging.setup import request_id_ctx


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):  # type: ignore
        token = request_id_ctx.set(_new_id())
        try:
            response = await call_next(request)
            # Attach header for clients (optional)
            response.headers["X-Request-ID"] = request_id_ctx.get() or "-"
            return response
        finally:
            request_id_ctx.reset(token)


class WebSocketCorrelationMiddleware:
    """ASGI middleware for WebSocket scopes only (manual insertion)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):  # type: ignore
        if scope["type"] == "websocket":
            token = request_id_ctx.set(_new_id())
            try:
                await self.app(scope, receive, send)
            finally:
                request_id_ctx.reset(token)
        else:
            await self.app(scope, receive, send)


__all__ = ["CorrelationIdMiddleware", "WebSocketCorrelationMiddleware"]
