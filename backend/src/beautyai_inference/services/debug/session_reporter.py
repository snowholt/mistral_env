"""Session Reporter Service

Provides aggregated debugging information for a streaming voice session.
Intended only for temporary internal diagnostics (exposed via feature flag).
"""
from __future__ import annotations

import os
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import subprocess
import datetime as _dt

LOG_DIR = Path(__file__).parents[3] / "logs" / "streaming"
STREAM_LOG_GLOB = "streaming_voice.jsonl"


class SessionReporter:
    def __init__(self) -> None:
        self.enabled = os.getenv("DEBUG_SESSION_REPORT_ENABLED", "1") == "1"
        self.include_journal = os.getenv("DEBUG_EXPOSE_JOURNAL", "0") == "1"

    def _find_stream_logs(self) -> List[Path]:
        if not LOG_DIR.exists():
            return []
        # Collect rotated variants
        files = []
        for p in LOG_DIR.glob(f"{STREAM_LOG_GLOB}*"):
            if p.is_file():
                files.append(p)
        # Newest last so we append chronological when reading tail-first
        return sorted(files)

    def _extract_session_events(self, session_id: str, max_events: int = 800) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        pattern = re.compile(r'"session_id"\s*:\s*"' + re.escape(session_id) + r'"')
        for log_file in reversed(self._find_stream_logs()):  # newest first
            try:
                with log_file.open("r", encoding="utf-8", errors="ignore") as f:
                    for line in reversed(f.readlines()):  # reverse scan
                        if pattern.search(line):
                            try:
                                obj = json.loads(line)
                                events.append(obj)
                                if len(events) >= max_events:
                                    return list(reversed(events))
                            except Exception:
                                continue
            except Exception:
                continue
        return list(reversed(events))

    def _journal_slice(self, since_iso: Optional[str], until_iso: Optional[str], max_lines: int) -> List[str]:
        if not self.include_journal:
            return []
        cmd = [
            "journalctl", "-u", os.getenv("API_SERVICE_UNIT", "beautyai-api.service"), "--no-pager",
        ]
        if since_iso:
            cmd.extend(["--since", since_iso])
        if until_iso:
            cmd.extend(["--until", until_iso])
        try:
            out = subprocess.check_output(cmd, text=True, timeout=6)
            lines = out.splitlines()
            if len(lines) > max_lines:
                lines = lines[-max_lines:]
            return lines
        except Exception:
            return []

    def _model_introspection(self) -> Dict[str, Any]:
        # Lightweight placeholder – real integration can pull from model manager
        info: Dict[str, Any] = {}
        try:
            from ..voice.transcription.whisper_finetuned_arabic_engine import WhisperFinetunedArabicEngine
            engine = WhisperFinetunedArabicEngine()
            info = engine.get_model_info()
            # We do not load the model here (expensive) – rely on metadata only.
        except Exception as e:
            info = {"error": f"introspection_failed: {e}"}
        return info

    def build_report(
        self,
        session_id: str,
        max_events: int = 800,
        include_journal: bool = False,
        max_journal_lines: int = 300,
        since_iso: Optional[str] = None,
        until_iso: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "message": "Session reporting disabled"}

        events = self._extract_session_events(session_id, max_events=max_events)
        first_ts = None
        last_ts = None
        for ev in events:
            try:
                ts = _dt.datetime.fromisoformat(ev.get("timestamp").replace("Z", "+00:00")) if ev.get("timestamp") else None
            except Exception:
                ts = None
            if ts:
                if first_ts is None or ts < first_ts:
                    first_ts = ts
                if last_ts is None or ts > last_ts:
                    last_ts = ts
        journal_lines: List[str] = []
        if include_journal and self.include_journal:
            journal_lines = self._journal_slice(
                since_iso or (first_ts.isoformat() if first_ts else None),
                until_iso or (last_ts.isoformat() if last_ts else None),
                max_journal_lines,
            )
        model_info = self._model_introspection()
        return {
            "enabled": True,
            "session_id": session_id,
            "events_count": len(events),
            "time_span": {
                "first": first_ts.isoformat() if first_ts else None,
                "last": last_ts.isoformat() if last_ts else None,
            },
            "events": events,
            "journal": {
                "included": include_journal and self.include_journal,
                "lines": journal_lines,
                "line_count": len(journal_lines),
            },
            "model": model_info,
            "config": {
                "include_journal_requested": include_journal,
                "feature_flags": {
                    "DEBUG_SESSION_REPORT_ENABLED": self.enabled,
                    "DEBUG_EXPOSE_JOURNAL": self.include_journal,
                },
            },
        }

__all__ = ["SessionReporter"]
