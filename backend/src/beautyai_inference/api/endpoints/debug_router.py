"""Debug / Diagnostics Endpoints (Temporary)

Exposes session report aggregation for streaming voice sessions.
Controlled by environment flags; not for production deployment.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import os

from ...services.debug.session_reporter import SessionReporter

debug_router = APIRouter(prefix="/api/v1/debug", tags=["debug"])

@debug_router.get("/session-report")
def get_session_report(
    session_id: str = Query(..., description="Streaming session_id (e.g. stream_xxx)"),
    max_events: int = Query(800, ge=10, le=5000),
    include_journal: bool = Query(False),
    max_journal_lines: int = Query(300, ge=50, le=1200),
):
    reporter = SessionReporter()
    if not reporter.enabled:
        raise HTTPException(status_code=403, detail="Session reporting disabled")
    report = reporter.build_report(
        session_id=session_id,
        max_events=max_events,
        include_journal=include_journal,
        max_journal_lines=max_journal_lines,
    )
    return report

__all__ = ["debug_router"]
