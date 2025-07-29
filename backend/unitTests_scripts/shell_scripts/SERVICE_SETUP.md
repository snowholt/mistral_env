# BeautyAI API Service Setup

This document explains how to set up the BeautyAI API as a systemd service for development purposes.

## Quick Setup

1. **Install the service:**
   ```bash
   cd /home/lumi/beautyai
   ./manage-api-service.sh install
   ```

2. **Start the API service:**
   ```bash
   ./manage-api-service.sh start
   ```

3. **Check status:**
   ```bash
   ./manage-api-service.sh status
   ```

## Available Commands

### Service Management
```bash
# Install/uninstall service
./manage-api-service.sh install      # Install systemd service
./manage-api-service.sh uninstall    # Remove systemd service

# Start/stop/restart
./manage-api-service.sh start        # Start API server
./manage-api-service.sh stop         # Stop API server  
./manage-api-service.sh restart      # Restart API server

# Status and monitoring
./manage-api-service.sh status       # Show service status & API health
./manage-api-service.sh logs         # Show recent logs

# Auto-start configuration
./manage-api-service.sh enable       # Enable auto-start on boot
./manage-api-service.sh disable      # Disable auto-start on boot
```

### Standard Systemctl Commands
After installation, you can also use standard Ubuntu service commands:

```bash
# Service control
sudo systemctl start beautyai-api
sudo systemctl stop beautyai-api
sudo systemctl restart beautyai-api
sudo systemctl status beautyai-api

# Enable/disable auto-start
sudo systemctl enable beautyai-api
sudo systemctl disable beautyai-api

# View logs
sudo journalctl -u beautyai-api -f         # Follow logs in real-time
sudo journalctl -u beautyai-api -n 100     # Show last 100 lines
```

## Service Features

- **Development Mode**: Runs with `--reload` flag for automatic code reloading
- **Auto-restart**: Automatically restarts if the service crashes
- **Security**: Runs with restricted permissions and security hardening
- **Logging**: All output goes to systemd journal for easy monitoring
- **Resource Limits**: Configured with appropriate memory and file limits
- **CUDA Support**: Properly configured environment for GPU access

## Configuration

The service is configured to:
- Run on port 8000 (accessible at http://localhost:8000)
- Use the virtual environment at `/home/lumi/beautyai/venv`
- Run as user `lumi` with working directory `/home/lumi/beautyai`
- Automatically restart on failure
- Include CUDA environment variables

## Development Workflow

```bash
# 1. Install service (one-time setup)
./manage-api-service.sh install

# 2. Start development server
./manage-api-service.sh start

# 3. Make code changes (server auto-reloads)
# Edit your code...

# 4. Check status/logs if needed
./manage-api-service.sh status
./manage-api-service.sh logs

# 5. Stop when done
./manage-api-service.sh stop
```

## Testing the API

Once the service is running, test the API:

```bash
# Health check
curl http://localhost:8000/

# Load a model
curl -X POST "http://localhost:8000/models/qwen3-model/load"

# Chat with the model
curl -X POST "http://localhost:8000/inference/chat" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "qwen3-model", "message": "Hello!"}'
```

## Troubleshooting

### Service won't start
```bash
# Check detailed status
./manage-api-service.sh status

# View error logs
./manage-api-service.sh logs

# Check if port is in use
sudo netstat -tlnp | grep :8000
```

### Permission issues
```bash
# Ensure virtual environment is accessible
ls -la /home/lumi/beautyai/venv/bin/uvicorn

# Check service file permissions
ls -la /etc/systemd/system/beautyai-api.service
```

### CUDA/GPU issues
```bash
# Verify GPU access
nvidia-smi

# Check CUDA environment in service
sudo systemctl show beautyai-api -p Environment
```

## Log Locations

- **Service logs**: `sudo journalctl -u beautyai-api`
- **Application logs**: Included in service logs via stdout/stderr
- **System logs**: `/var/log/syslog` (for systemd-related messages)

## Uninstalling

To completely remove the service:

```bash
./manage-api-service.sh uninstall
```

This will:
- Stop the service if running
- Disable auto-start
- Remove the systemd service file
- Clean up systemd configuration
