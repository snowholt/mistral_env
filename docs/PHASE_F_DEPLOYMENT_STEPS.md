# Phase F Deployment Steps - WebRTC Infrastructure

**Date:** October 15, 2025  
**Status:** Ready for Deployment  
**Estimated Time:** 15 minutes

---

## 📋 Overview

This guide shows **exactly what to change** in your existing production files to add WebRTC support.

### What We're Updating

1. **Nginx Configuration** - Add WebRTC signaling routes to existing `/etc/nginx/sites-enabled/gmai.sa`
2. **Systemd Service** - Add WebRTC environment variables to existing `/etc/systemd/system/beautyai-api.service`
3. **Monitoring** - Configure log ingestion for aiortc warnings

### What We're NOT Doing

- ❌ NOT creating new services
- ❌ NOT replacing existing files entirely
- ❌ NOT changing WebSocket functionality
- ✅ ONLY adding WebRTC support alongside existing systems

---

## Step 1: Update Nginx Configuration

### Current File
`/etc/nginx/sites-enabled/gmai.sa` (your production Nginx config)

### What to Add
Add these WebRTC routes **BEFORE** the existing `/api/` location block in **BOTH** server blocks.

#### For `dev.gmai.sa` Server Block

Find this section:
```nginx
# ================================================================
# WEBSOCKET ENDPOINTS for dev.gmai.sa (Frontend WebSocket requests)
# ================================================================
```

**Add BEFORE** the WebSocket section (around line 20):
```nginx
# ================================================================
# WEBRTC VOICE SIGNALING ENDPOINTS (Phase F - Oct 15, 2025)
# ================================================================

# WebRTC Health Check
location /api/v1/webrtc/voice/health {
    proxy_pass http://localhost:8000/api/v1/webrtc/voice/health;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    
    proxy_read_timeout 5s;
    proxy_send_timeout 5s;
    proxy_connect_timeout 5s;
    proxy_buffering off;
}

# WebRTC SDP Offer/Answer Exchange
location /api/v1/webrtc/voice/offer {
    proxy_pass http://localhost:8000/api/v1/webrtc/voice/offer;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    
    proxy_read_timeout 30s;
    proxy_send_timeout 30s;
    proxy_connect_timeout 10s;
    client_max_body_size 10M;
    proxy_buffering off;
}

# WebRTC ICE Candidate Exchange
location /api/v1/webrtc/voice/ice {
    proxy_pass http://localhost:8000/api/v1/webrtc/voice/ice;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    
    proxy_read_timeout 30s;
    proxy_send_timeout 30s;
    proxy_connect_timeout 10s;
    proxy_buffering off;
}

# WebRTC Session Status
location ~ ^/api/v1/webrtc/voice/([a-zA-Z0-9_-]+)/status$ {
    proxy_pass http://localhost:8000/api/v1/webrtc/voice/$1/status;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    
    proxy_read_timeout 10s;
    proxy_send_timeout 10s;
    proxy_buffering off;
}

# WebRTC Session Cleanup
location ~ ^/api/v1/webrtc/voice/([a-zA-Z0-9_-]+)$ {
    proxy_pass http://localhost:8000/api/v1/webrtc/voice/$1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    
    proxy_read_timeout 15s;
    proxy_send_timeout 15s;
    proxy_buffering off;
}
```

#### For `api.gmai.sa` Server Block

Find this section:
```nginx
# ================================================================
# WEBSOCKET ENDPOINTS: wss://api.gmai.sa/api/v1/ws/
# ================================================================
```

**Add BEFORE** the WebSocket section (around line 140):
```nginx
# ================================================================
# WEBRTC VOICE SIGNALING ENDPOINTS (Phase F - Oct 15, 2025)
# ================================================================

# WebRTC Health Check
location /api/v1/webrtc/voice/health {
    proxy_pass http://localhost:8000/api/v1/webrtc/voice/health;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    
    proxy_read_timeout 5s;
    proxy_send_timeout 5s;
    proxy_connect_timeout 5s;
    proxy_buffering off;
}

# WebRTC SDP Offer/Answer Exchange
location /api/v1/webrtc/voice/offer {
    proxy_pass http://localhost:8000/api/v1/webrtc/voice/offer;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    
    proxy_read_timeout 30s;
    proxy_send_timeout 30s;
    proxy_connect_timeout 10s;
    client_max_body_size 10M;
    proxy_buffering off;
}

# WebRTC ICE Candidate Exchange
location /api/v1/webrtc/voice/ice {
    proxy_pass http://localhost:8000/api/v1/webrtc/voice/ice;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    
    proxy_read_timeout 30s;
    proxy_send_timeout 30s;
    proxy_connect_timeout 10s;
    proxy_buffering off;
}

# WebRTC Session Status
location ~ ^/api/v1/webrtc/voice/([a-zA-Z0-9_-]+)/status$ {
    proxy_pass http://localhost:8000/api/v1/webrtc/voice/$1/status;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    
    proxy_read_timeout 10s;
    proxy_send_timeout 10s;
    proxy_buffering off;
}

# WebRTC Session Cleanup
location ~ ^/api/v1/webrtc/voice/([a-zA-Z0-9_-]+)$ {
    proxy_pass http://localhost:8000/api/v1/webrtc/voice/$1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    
    proxy_read_timeout 15s;
    proxy_send_timeout 15s;
    proxy_buffering off;
}
```

### Commands
```bash
# 1. Backup current config
sudo cp /etc/nginx/sites-enabled/gmai.sa /etc/nginx/sites-enabled/gmai.sa.backup-$(date +%Y%m%d)

# 2. Edit the file (use nano or vim)
sudo nano /etc/nginx/sites-enabled/gmai.sa

# 3. Test configuration
sudo nginx -t

# 4. If test passes, reload nginx
sudo systemctl reload nginx

# 5. Verify WebRTC routes are accessible
curl -X GET https://dev.gmai.sa/api/v1/webrtc/voice/health
```

---

## Step 2: Update Systemd Service

### Current File
`/etc/systemd/system/beautyai-api.service` (your production service)

### What to Add
Add these WebRTC environment variables **AFTER** the existing `VOICE_STREAMING_*` variables (around line 27).

Find this line:
```
Environment=VOICE_STREAMING_LOW_LATENCY_PRESET=1
```

**Add AFTER** that line:
```bash
# ================================================================
# WebRTC Configuration (Phase F - Added Oct 15, 2025)
# ================================================================
# Maximum utterance duration in seconds
Environment=VOICE_WEBRTC_MAX_UTTERANCE_SEC=10

# Enable aiortc debug logging (0 for production, 1 for debugging)
Environment=VOICE_WEBRTC_DEBUG=0

# aiortc log level: WARNING for production, DEBUG for troubleshooting
Environment=AIORTC_LOG_LEVEL=WARNING

# STUN server configuration (Google public STUN for MVP)
Environment=WEBRTC_STUN_SERVER=stun:stun.l.google.com:19302

# Enable WebRTC feature flag (1 = enabled, 0 = WebSocket fallback only)
Environment=WEBRTC_ENABLED=1

# WebRTC connection timeout settings
Environment=WEBRTC_ICE_GATHERING_TIMEOUT=30
Environment=WEBRTC_CONNECTION_TIMEOUT=10

# WebRTC audio processing settings
Environment=WEBRTC_AUDIO_SAMPLE_RATE=16000
Environment=WEBRTC_VAD_THRESHOLD_AR=0.45
Environment=WEBRTC_VAD_THRESHOLD_EN=0.50
# ================================================================
```

### Commands
```bash
# 1. Backup current service file
sudo cp /etc/systemd/system/beautyai-api.service /etc/systemd/system/beautyai-api.service.backup-$(date +%Y%m%d)

# 2. Edit the service file
sudo nano /etc/systemd/system/beautyai-api.service

# 3. Reload systemd daemon
sudo systemctl daemon-reload

# 4. Restart the service
sudo systemctl restart beautyai-api

# 5. Check service status
sudo systemctl status beautyai-api

# 6. Check logs for WebRTC initialization
sudo journalctl -u beautyai-api -f --since "1 minute ago" | grep -i webrtc
```

---

## Step 3: Setup Monitoring and Alerting

### Create Log Monitoring Script

```bash
# Create monitoring script
sudo tee /usr/local/bin/monitor-webrtc-logs.sh > /dev/null << 'EOF'
#!/bin/bash
# WebRTC Log Monitor - Checks for aiortc warnings and errors
# Created: Oct 15, 2025 (Phase F)

LOG_FILE="/var/log/beautyai/webrtc-monitor.log"
ALERT_THRESHOLD=10  # Alert if more than 10 errors in 5 minutes

# Create log directory if it doesn't exist
mkdir -p /var/log/beautyai

# Check for aiortc errors in the last 5 minutes
ERROR_COUNT=$(sudo journalctl -u beautyai-api --since "5 minutes ago" | grep -i "aiortc.*error" | wc -l)

if [ $ERROR_COUNT -gt $ALERT_THRESHOLD ]; then
    echo "[$(date)] ALERT: $ERROR_COUNT aiortc errors detected in last 5 minutes" >> $LOG_FILE
    # Add alerting mechanism here (email, Slack, PagerDuty, etc.)
fi

# Log WebRTC connection stats
WEBRTC_CONNECTIONS=$(sudo journalctl -u beautyai-api --since "5 minutes ago" | grep -i "webrtc.*connection.*established" | wc -l)
echo "[$(date)] WebRTC Connections: $WEBRTC_CONNECTIONS" >> $LOG_FILE
EOF

# Make executable
sudo chmod +x /usr/local/bin/monitor-webrtc-logs.sh
```

### Create Systemd Timer for Monitoring

```bash
# Create timer unit
sudo tee /etc/systemd/system/webrtc-monitor.timer > /dev/null << 'EOF'
[Unit]
Description=WebRTC Log Monitor Timer
Requires=webrtc-monitor.service

[Timer]
OnBootSec=5min
OnUnitActiveSec=5min

[Install]
WantedBy=timers.target
EOF

# Create service unit
sudo tee /etc/systemd/system/webrtc-monitor.service > /dev/null << 'EOF'
[Unit]
Description=WebRTC Log Monitor Service
After=beautyai-api.service

[Service]
Type=oneshot
ExecStart=/usr/local/bin/monitor-webrtc-logs.sh

[Install]
WantedBy=multi-user.target
EOF

# Enable and start timer
sudo systemctl daemon-reload
sudo systemctl enable webrtc-monitor.timer
sudo systemctl start webrtc-monitor.timer

# Check timer status
sudo systemctl status webrtc-monitor.timer
```

### Health Check Endpoint

Test the WebRTC health check endpoint:

```bash
# Test health endpoint
curl -X GET https://dev.gmai.sa/api/v1/webrtc/voice/health

# Expected response:
# {"status": "healthy", "webrtc_enabled": true, "timestamp": "2025-10-15T..."}
```

---

## Step 4: Verification

### 1. Check Nginx Configuration
```bash
# Verify WebRTC routes are loaded
sudo nginx -t
sudo systemctl status nginx

# Test routes
curl -X GET https://dev.gmai.sa/api/v1/webrtc/voice/health
curl -X GET https://api.gmai.sa/api/v1/webrtc/voice/health
```

### 2. Check Backend Service
```bash
# Verify environment variables are loaded
sudo systemctl show beautyai-api | grep -i webrtc

# Check service logs
sudo journalctl -u beautyai-api -n 50 | grep -i webrtc

# Verify WebRTC endpoints are registered
curl -X GET http://localhost:8000/docs | grep webrtc
```

### 3. Check Monitoring
```bash
# View monitoring logs
sudo tail -f /var/log/beautyai/webrtc-monitor.log

# Check timer status
sudo systemctl list-timers | grep webrtc
```

---

## Rollback Procedure (If Needed)

### Rollback Nginx
```bash
# Restore backup
sudo cp /etc/nginx/sites-enabled/gmai.sa.backup-YYYYMMDD /etc/nginx/sites-enabled/gmai.sa
sudo nginx -t
sudo systemctl reload nginx
```

### Rollback Systemd Service
```bash
# Restore backup
sudo cp /etc/systemd/system/beautyai-api.service.backup-YYYYMMDD /etc/systemd/system/beautyai-api.service
sudo systemctl daemon-reload
sudo systemctl restart beautyai-api
```

---

## Files Reference

### Repository Files (for reference only)
- `config/nginx-webrtc-routes.conf` - **Snippet showing WebRTC routes**
- `config/gmai.sa.nginx.conf` - **Complete example config with WebRTC**
- `beautyai-api.service.webrtc` - **Complete example service with WebRTC**

### Production Files (what you actually modify)
- `/etc/nginx/sites-enabled/gmai.sa` - **Your real Nginx config**
- `/etc/systemd/system/beautyai-api.service` - **Your real systemd service**

---

## Summary of Changes

| Component | Action | File | Lines Added |
|-----------|--------|------|-------------|
| Nginx Config | Add WebRTC routes | `/etc/nginx/sites-enabled/gmai.sa` | ~120 lines (2 server blocks) |
| Systemd Service | Add environment vars | `/etc/systemd/system/beautyai-api.service` | ~15 lines |
| Monitoring | Create monitoring script | `/usr/local/bin/monitor-webrtc-logs.sh` | New file |
| Monitoring | Create systemd timer | `/etc/systemd/system/webrtc-monitor.*` | 2 new files |

**Total Deployment Time:** ~15 minutes  
**Downtime Required:** None (reload only, no restart needed for Nginx)  
**Service Restart:** beautyai-api.service only (~5 seconds downtime)

---

## Next Steps

After deployment:
1. Monitor logs for 24 hours: `sudo journalctl -u beautyai-api -f | grep -i webrtc`
2. Test WebRTC voice from browser: https://dev.gmai.sa (with WebRTC mode enabled)
3. Verify STUN connectivity: Check browser console for ICE candidates
4. Document any issues in Phase F report

**TURN server deployment** (optional, post-MVP) is documented in `docs/DEPLOYMENT.md`
