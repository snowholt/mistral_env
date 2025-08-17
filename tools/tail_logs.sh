#!/usr/bin/env bash
# Helper: Tail BeautyAI logs (app, access, streaming) with optional follow flag.
# Usage: ./tools/tail_logs.sh [follow]
set -euo pipefail
LOG_ROOT=${BEAUTYAI_LOG_DIR:-./logs}
APP=${LOG_ROOT}/api/api_app.jsonl
ACCESS=${LOG_ROOT}/api/api_access.log
STREAM=${LOG_ROOT}/streaming/streaming_voice.jsonl

if [ "${1:-}" = "follow" ]; then
  echo "Tailing (follow) app, access, streaming logs..."
  tail -n 50 -f "$APP" "$ACCESS" "$STREAM"
else
  echo "=== APP (last 60) ==="; tail -n 60 "$APP" || echo "(missing)"; echo
  echo "=== ACCESS (last 40) ==="; tail -n 40 "$ACCESS" || echo "(missing)"; echo
  echo "=== STREAMING (last 60) ==="; tail -n 60 "$STREAM" || echo "(missing)"; echo
fi
