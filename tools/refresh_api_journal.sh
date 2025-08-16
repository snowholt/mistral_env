#!/usr/bin/env bash
# Reset (vacuum) and capture fresh journal for beautyai-api.service
# Usage: ./tools/refresh_api_journal.sh [lines]
set -euo pipefail
LINES=${1:-500}
LOG_PATH="reports/logs/beautyai_api_journal.log"
mkdir -p "$(dirname "$LOG_PATH")"
# Rotate/vacuum old entries older than 1h for this unit (does not delete if other retention policies longer)
sudo journalctl -u beautyai-api.service --rotate || true
sudo journalctl -u beautyai-api.service --vacuum-time=1s || true
# Capture current fresh log (after rotation there may be none yet)
sudo journalctl -u beautyai-api.service -n "$LINES" --no-pager -o short-iso > "$LOG_PATH" || true
# Follow new entries (optional) if FOLLOW env set
if [[ "${FOLLOW:-0}" == "1" ]]; then
  echo "--- FOLLOWING NEW ENTRIES (Ctrl+C to stop) ---" | tee -a "$LOG_PATH"
  sudo journalctl -u beautyai-api.service -f -o short-iso | tee -a "$LOG_PATH"
fi
echo "Wrote latest $LINES lines to $LOG_PATH" >&2
