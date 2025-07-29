# BeautyAI Web UI Service with SSL Support

This enhanced service configuration supports both HTTP and HTTPS modes for the BeautyAI Web UI.

## Features

- **Automatic SSL Certificate Generation**: Self-signed certificates for development
- **Flexible SSL Configuration**: Auto, enabled, or disabled modes
- **Port Configuration**: Customizable port settings
- **Service Management**: Easy start/stop/status commands with SSL info
- **Production Ready**: Systemd service integration

## Quick Start

### 1. Setup Service with SSL Support

Use the enhanced setup script that includes SSL configuration:

```bash
./setup-service-ssl.sh
```

This will prompt you to choose:
- SSL mode (auto/enable/disable)
- Port number (default: 5000)

### 2. Alternative: Use Original Setup

For HTTP-only setup (legacy):

```bash
./setup-service.sh
```

## SSL Configuration Options

### Auto Mode (Recommended)
- **Environment**: `BEAUTYAI_ENABLE_SSL=auto`
- **Behavior**: Automatically generates self-signed certificates if they don't exist
- **Fallback**: Uses HTTP if certificate generation fails
- **Best for**: Development and testing

### Enabled Mode
- **Environment**: `BEAUTYAI_ENABLE_SSL=true`
- **Behavior**: Requires existing SSL certificates
- **Fallback**: Service fails if certificates are missing
- **Best for**: Production with proper certificates

### Disabled Mode
- **Environment**: `BEAUTYAI_ENABLE_SSL=false`
- **Behavior**: Always uses HTTP, ignores certificates
- **Best for**: Local development or when HTTPS isn't needed

## Service Management

### Enhanced Management (Recommended)

```bash
# Show status with SSL configuration
./manage-service-ssl.sh status

# Configure SSL settings
./manage-service-ssl.sh ssl

# Start/stop/restart service
./manage-service-ssl.sh start
./manage-service-ssl.sh stop
./manage-service-ssl.sh restart

# View logs
./manage-service-ssl.sh logs
```

### Basic Management (Legacy)

```bash
./manage-service.sh start|stop|restart|status|logs
```

## Manual SSL Configuration

### Generate SSL Certificates

```bash
cd src/web_ui
python generate_ssl_cert.py
```

### Update Service Configuration

Edit `/etc/systemd/system/beautyai-webui.service` and modify:

```ini
Environment=BEAUTYAI_ENABLE_SSL=auto
Environment=BEAUTYAI_PORT=5000
```

Then reload and restart:

```bash
sudo systemctl daemon-reload
sudo systemctl restart beautyai-webui
```

## Environment Variables

The service supports these environment variables:

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `BEAUTYAI_ENABLE_SSL` | `auto`, `true`, `false` | `auto` | SSL configuration mode |
| `BEAUTYAI_PORT` | `1024-65535` | `5000` | Port number |
| `BEAUTYAI_API_URL` | URL | `http://localhost:8000` | Backend API URL |

## Accessing the Web UI

### With SSL (HTTPS)
- **URL**: `https://localhost:5000` (or your configured port)
- **Certificate Warning**: Browsers will show warnings for self-signed certificates
- **Security**: Click "Advanced" → "Proceed to localhost" to access

### Without SSL (HTTP)
- **URL**: `http://localhost:5000` (or your configured port)
- **Microphone**: Only works on localhost without HTTPS

## Files and Structure

```
├── beautyai-webui-service.py     # Enhanced service wrapper with SSL support
├── beautyai-webui.service        # Systemd service file with SSL environment vars
├── setup-service-ssl.sh          # Enhanced setup script with SSL configuration
├── manage-service-ssl.sh         # Enhanced management script with SSL controls
├── setup-service.sh              # Legacy setup script (HTTP only)
├── manage-service.sh             # Legacy management script
└── src/web_ui/
    ├── app.py                    # Main Flask application with SSL support
    ├── generate_ssl_cert.py      # SSL certificate generator
    ├── cert.pem                  # SSL certificate (generated)
    └── key.pem                   # SSL private key (generated)
```

## Troubleshooting

### SSL Certificate Issues

1. **Regenerate certificates**:
   ```bash
   cd src/web_ui
   rm -f cert.pem key.pem
   python generate_ssl_cert.py
   ```

2. **Check OpenSSL availability**:
   ```bash
   which openssl
   # If not found: sudo apt-get install openssl
   ```

### Service Issues

1. **Check service status**:
   ```bash
   ./manage-service-ssl.sh status
   ```

2. **View detailed logs**:
   ```bash
   ./manage-service-ssl.sh logs
   ```

3. **Check configuration**:
   ```bash
   cat /etc/systemd/system/beautyai-webui.service | grep Environment
   ```

### Port Conflicts

1. **Check if port is in use**:
   ```bash
   sudo netstat -tlnp | grep :5000
   ```

2. **Change port**:
   ```bash
   ./manage-service-ssl.sh ssl
   # Select new port when prompted
   ```

## Security Notes

- Self-signed certificates are for development only
- For production, use proper SSL certificates from a trusted CA
- Consider using Let's Encrypt for free trusted certificates
- The service runs as a non-root user for security

## Migration from HTTP-only Service

If you have an existing HTTP-only service:

1. **Stop the current service**:
   ```bash
   sudo systemctl stop beautyai-webui
   ```

2. **Run the enhanced setup**:
   ```bash
   ./setup-service-ssl.sh
   ```

3. **Choose your SSL preference when prompted**

The enhanced setup will update your existing service configuration while preserving your data and settings.
