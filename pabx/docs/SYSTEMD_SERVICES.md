# Systemd Service Management Guide

## Overview

The PABX system can be run as systemd services for production deployment, providing:
- ✅ Automatic startup on boot
- ✅ Automatic restart on failure
- ✅ Centralized logging with journald
- ✅ Process management and monitoring
- ✅ Resource limits and security settings

---

## Available Services

### 1. **pabx-backend.service**
Backend API server (FastAPI + Uvicorn)
- Listens on: http://0.0.0.0:8080
- WebSocket: ws://0.0.0.0:8080/ws
- Runs as: User `lumi`
- Auto-restart: On failure

### 2. **pabx-frontend.service** (Development)
Frontend dev server with hot-reload (Vite)
- Listens on: http://0.0.0.0:3000
- Runs: `npm run dev`
- Use for: Development and testing

### 3. **pabx-frontend-prod.service** (Production)
Frontend production server (Vite preview)
- Listens on: http://0.0.0.0:3000
- Runs: `npm run preview`
- Requires: Built files in `ui/dist/`
- Use for: Production deployment

---

## Installation

### Quick Install

Run the interactive installer:
```bash
cd /home/lumi/beautyai/pabx/systemd
sudo ./install_services.sh
```

**Installation Options:**
1. Development Mode - Frontend with hot-reload
2. Production Mode - Built frontend files
3. Backend Only - API server only
4. Uninstall Services - Remove all services

### Manual Install

**Backend:**
```bash
sudo cp systemd/pabx-backend.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable pabx-backend
sudo systemctl start pabx-backend
```

**Frontend (Dev):**
```bash
sudo cp systemd/pabx-frontend.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable pabx-frontend
sudo systemctl start pabx-frontend
```

**Frontend (Prod):**
```bash
# Build first
cd /home/lumi/beautyai/pabx/ui
npm run build

# Install service
sudo cp systemd/pabx-frontend-prod.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable pabx-frontend-prod
sudo systemctl start pabx-frontend-prod
```

---

## Service Management

### Using the Helper Script (Recommended)

```bash
cd /home/lumi/beautyai/pabx

# Start all services
./pabx-service.sh start

# Stop all services
./pabx-service.sh stop

# Restart all services
./pabx-service.sh restart

# Check status
./pabx-service.sh status

# View logs
./pabx-service.sh logs

# Backend only
./pabx-service.sh start backend
./pabx-service.sh logs backend

# Frontend only
./pabx-service.sh start frontend
./pabx-service.sh logs frontend
```

### Using systemctl Directly

**Start Services:**
```bash
sudo systemctl start pabx-backend
sudo systemctl start pabx-frontend
```

**Stop Services:**
```bash
sudo systemctl stop pabx-backend
sudo systemctl stop pabx-frontend
```

**Restart Services:**
```bash
sudo systemctl restart pabx-backend
sudo systemctl restart pabx-frontend
```

**Check Status:**
```bash
sudo systemctl status pabx-backend
sudo systemctl status pabx-frontend
```

**Enable Auto-start on Boot:**
```bash
sudo systemctl enable pabx-backend
sudo systemctl enable pabx-frontend
```

**Disable Auto-start:**
```bash
sudo systemctl disable pabx-backend
sudo systemctl disable pabx-frontend
```

---

## Logging

### View Logs

**Follow live logs:**
```bash
# All services
sudo journalctl -u pabx-backend -u pabx-frontend -f

# Backend only
sudo journalctl -u pabx-backend -f

# Frontend only
sudo journalctl -u pabx-frontend -f
```

**View recent logs:**
```bash
# Last 100 lines
sudo journalctl -u pabx-backend -n 100

# Last hour
sudo journalctl -u pabx-backend --since "1 hour ago"

# Today's logs
sudo journalctl -u pabx-backend --since today
```

**Search logs:**
```bash
# Search for errors
sudo journalctl -u pabx-backend | grep ERROR

# Search for specific call
sudo journalctl -u pabx-backend | grep "call_id"
```

### Log Locations

**Systemd Journal:**
- Location: `/var/log/journal/`
- Access: `journalctl -u pabx-backend`

**Application Logs:**
- Location: `/home/lumi/beautyai/pabx/logs/`
- Files:
  - `api_server.log`
  - `sip_server.log`
  - `rtp_handler.log`

---

## Troubleshooting

### Service Won't Start

**Check status:**
```bash
sudo systemctl status pabx-backend
```

**Check logs:**
```bash
sudo journalctl -u pabx-backend -n 50
```

**Common issues:**

1. **Port already in use:**
```bash
# Check what's using port 8080
sudo lsof -i :8080

# Kill the process
sudo kill -9 <PID>
```

2. **Virtual environment not found:**
```bash
# Verify venv exists
ls -la /home/lumi/beautyai/pabx/venv/bin/python3

# Recreate if needed
cd /home/lumi/beautyai/pabx
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **Permission denied:**
```bash
# Fix ownership
sudo chown -R lumi:lumi /home/lumi/beautyai/pabx

# Fix permissions
chmod +x /home/lumi/beautyai/pabx/run_server.py
```

### Service Crashes/Restarts

**View crash logs:**
```bash
sudo journalctl -u pabx-backend -p err
```

**Check restart count:**
```bash
systemctl show pabx-backend -p NRestarts
```

**Increase restart delay:**
Edit service file and change `RestartSec=10` to higher value.

### Frontend Build Issues

**Rebuild frontend:**
```bash
cd /home/lumi/beautyai/pabx/ui
rm -rf node_modules dist
npm install
npm run build
```

**Switch to dev mode:**
```bash
sudo systemctl stop pabx-frontend-prod
sudo systemctl start pabx-frontend
```

---

## Service Configuration

### Modify Service Files

**Edit service:**
```bash
sudo systemctl edit pabx-backend --full
```

**Reload after changes:**
```bash
sudo systemctl daemon-reload
sudo systemctl restart pabx-backend
```

### Common Modifications

**Change port:**
```ini
# In pabx-backend.service
ExecStart=/home/lumi/beautyai/pabx/venv/bin/python3 /home/lumi/beautyai/pabx/run_server.py --mode api --api-port 8081
```

**Add environment variables:**
```ini
Environment="DEBUG=true"
Environment="LOG_LEVEL=DEBUG"
```

**Change user:**
```ini
User=youruser
Group=yourgroup
```

---

## Security Settings

Services include security hardening:
- `NoNewPrivileges=true` - Prevent privilege escalation
- `PrivateTmp=true` - Isolated /tmp directory
- `AmbientCapabilities=CAP_NET_BIND_SERVICE` - Allow port 80/443

### Allow privileged ports:

```bash
# Allow binding to port 80
sudo setcap 'cap_net_bind_service=+ep' /home/lumi/beautyai/pabx/venv/bin/python3
```

---

## Performance Tuning

### Resource Limits

Add to service file:
```ini
[Service]
LimitNOFILE=65535          # Max open files
LimitNPROC=4096            # Max processes
CPUQuota=200%              # Max CPU (200% = 2 cores)
MemoryMax=2G               # Max memory
```

### Restart Policy

```ini
Restart=on-failure         # Restart on crash
Restart=always            # Always restart
Restart=on-abnormal       # Restart on abnormal exit

RestartSec=10             # Wait 10s before restart
StartLimitBurst=5         # Max 5 restarts
StartLimitIntervalSec=60  # In 60 seconds
```

---

## Monitoring

### Service Status Check

```bash
# Check if running
systemctl is-active pabx-backend

# Check if enabled
systemctl is-enabled pabx-backend

# Quick status
systemctl list-units "pabx-*"
```

### Automated Monitoring Script

```bash
#!/bin/bash
# monitor-pabx.sh

while true; do
    if ! systemctl is-active --quiet pabx-backend; then
        echo "Backend is down! Restarting..."
        systemctl restart pabx-backend
    fi
    
    if ! systemctl is-active --quiet pabx-frontend; then
        echo "Frontend is down! Restarting..."
        systemctl restart pabx-frontend
    fi
    
    sleep 60
done
```

---

## Uninstallation

### Remove Services

**Using installer:**
```bash
cd /home/lumi/beautyai/pabx/systemd
sudo ./install_services.sh
# Select option 4 (Uninstall)
```

**Manual removal:**
```bash
# Stop services
sudo systemctl stop pabx-backend pabx-frontend

# Disable auto-start
sudo systemctl disable pabx-backend pabx-frontend

# Remove service files
sudo rm /etc/systemd/system/pabx-backend.service
sudo rm /etc/systemd/system/pabx-frontend.service
sudo rm /etc/systemd/system/pabx-frontend-prod.service

# Reload systemd
sudo systemctl daemon-reload
```

---

## Quick Reference

### Common Commands

| Action | Command |
|--------|---------|
| Start all | `./pabx-service.sh start` |
| Stop all | `./pabx-service.sh stop` |
| Restart all | `./pabx-service.sh restart` |
| Status | `./pabx-service.sh status` |
| Logs | `./pabx-service.sh logs` |
| Enable boot | `./pabx-service.sh enable` |
| Disable boot | `./pabx-service.sh disable` |

### Service Files Location

- `/etc/systemd/system/pabx-backend.service`
- `/etc/systemd/system/pabx-frontend.service`
- `/etc/systemd/system/pabx-frontend-prod.service`

### Application Location

- Working directory: `/home/lumi/beautyai/pabx`
- Virtual env: `/home/lumi/beautyai/pabx/venv`
- Frontend: `/home/lumi/beautyai/pabx/ui`

---

## Best Practices

1. **Always use systemd in production** - Better than tmux/screen
2. **Enable auto-start** - Services restart on reboot
3. **Monitor logs** - Set up log rotation and monitoring
4. **Test before deployment** - Verify manually first
5. **Use production mode** - Build frontend for production
6. **Set resource limits** - Prevent resource exhaustion
7. **Regular updates** - Keep dependencies updated

---

## Migration from tmux/manual

**Stop manual processes:**
```bash
# Kill tmux session
tmux kill-session -t pabx

# Or kill processes
pkill -f "run_server.py"
pkill -f "npm run dev"
```

**Install services:**
```bash
cd /home/lumi/beautyai/pabx/systemd
sudo ./install_services.sh
```

**Verify:**
```bash
./pabx-service.sh status
```

---

**For more help:**
- Backend docs: `/home/lumi/beautyai/pabx/README.md`
- Frontend docs: `/home/lumi/beautyai/pabx/ui/README.md`
- Systemd manual: `man systemd.service`
