#!/usr/bin/env bash
# Create a systemd drop-in to write API service logs to a dedicated file via ExecStartPost + StandardOutput
set -euo pipefail
SERVICE=beautyai-api.service
DROPIN_DIR=/etc/systemd/system/${SERVICE}.d
sudo mkdir -p "$DROPIN_DIR"
cat <<'EOF' | sudo tee ${DROPIN_DIR}/10-logfile.conf
[Service]
# Ensure we still log to journal AND append to custom file
StandardOutput=journal
StandardError=journal
ExecStartPost=/bin/sh -c 'journalctl -u beautyai-api.service -f -o short-iso >> /home/lumi/beautyai/reports/logs/beautyai_api_journal.log 2>&1 & echo $! > /home/lumi/beautyai/backend/api_journal_tail.pid'
ExecStopPost=/bin/sh -c 'if [ -f /home/lumi/beautyai/backend/api_journal_tail.pid ]; then kill $(cat /home/lumi/beautyai/backend/api_journal_tail.pid) 2>/dev/null || true; fi'
EOF
sudo systemctl daemon-reload
sudo systemctl restart $SERVICE
echo "Configured persistent capture to reports/logs/beautyai_api_journal.log"