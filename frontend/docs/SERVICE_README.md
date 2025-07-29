# BeautyAI Web UI Service

This directory contains scripts and configuration files to run the BeautyAI Web UI as a systemd service on port 5000.

## Files

- `beautyai-webui.service` - Systemd service configuration file
- `beautyai-webui-service.py` - Service wrapper script that runs the web UI
- `setup-service.sh` - Installation script for setting up the service
- `manage-service.sh` - Management script for controlling the service

## Quick Start

### 1. Install the Service

Run the setup script to install and configure the service:

```bash
./setup-service.sh
```

This script will:
- Install required Python dependencies
- Create log files with proper permissions
- Install the systemd service
- Enable the service to start on boot

### 2. Start the Service

Start the service using the management script:

```bash
./manage-service.sh start
```

Or use systemctl directly:

```bash
sudo systemctl start beautyai-webui
```

### 3. Access the Web UI

Once the service is running, you can access the BeautyAI Web UI at:

**http://localhost:5000**

## Service Management

Use the `manage-service.sh` script for easy service management:

```bash
# Start the service
./manage-service.sh start

# Stop the service
./manage-service.sh stop

# Restart the service
./manage-service.sh restart

# Check service status
./manage-service.sh status

# View live logs
./manage-service.sh logs

# Enable service on boot
./manage-service.sh enable

# Disable service on boot
./manage-service.sh disable
```

## Manual Service Commands

You can also use systemctl commands directly:

```bash
# Start service
sudo systemctl start beautyai-webui

# Stop service
sudo systemctl stop beautyai-webui

# Restart service
sudo systemctl restart beautyai-webui

# Check status
sudo systemctl status beautyai-webui

# View logs
sudo journalctl -u beautyai-webui -f

# Enable on boot
sudo systemctl enable beautyai-webui

# Disable on boot
sudo systemctl disable beautyai-webui
```

## Logs

Service logs are stored in:
- **System logs**: `sudo journalctl -u beautyai-webui`
- **Application logs**: `/var/log/beautyai-webui.log`

## Configuration

The service runs with the following configuration:
- **Port**: 5000
- **Host**: 0.0.0.0 (accessible from all interfaces)
- **User**: lumi
- **Working Directory**: `/home/lumi/benchmark_and_test`
- **Auto-restart**: Yes (with 10-second delay)

## Troubleshooting

### Service Won't Start

1. Check the service status:
   ```bash
   sudo systemctl status beautyai-webui
   ```

2. View the logs:
   ```bash
   sudo journalctl -u beautyai-webui -n 50
   ```

3. Ensure dependencies are installed:
   ```bash
   python3 -m pip install flask aiohttp flask-cors requests
   ```

### Port Already in Use

If port 5000 is already in use, you can:

1. Check what's using the port:
   ```bash
   sudo netstat -tlnp | grep :5000
   ```

2. Stop the conflicting service or modify the port in:
   - `src/web_ui/app.py` (line 438)
   - `beautyai-webui-service.py` (line 42)

### Permission Issues

Ensure the service user (lumi) has:
- Read access to the project directory
- Write access to `/var/log/beautyai-webui.log`

## Security Notes

The service runs with security-hardened settings:
- No new privileges
- Private temporary directory
- Protected system directories
- Read-only home directory access

## Uninstalling

To remove the service:

```bash
# Stop and disable the service
sudo systemctl stop beautyai-webui
sudo systemctl disable beautyai-webui

# Remove service file
sudo rm /etc/systemd/system/beautyai-webui.service

# Reload systemd
sudo systemctl daemon-reload

# Remove log file (optional)
sudo rm /var/log/beautyai-webui.log
```
