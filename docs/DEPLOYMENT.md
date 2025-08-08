# Deployment Guide

Production deployment guide for the BeautyAI Inference Framework.

## 🚀 Production Deployment

### Prerequisites

#### System Requirements
```bash
# Ubuntu 20.04+ / RHEL 8+ / CentOS Stream 8+
# Python 3.11+
# NVIDIA GPU with CUDA 11.8+ (recommended)
# 16GB+ RAM (32GB+ recommended for large models)
# 100GB+ storage for models and logs
```

#### Required Services
```bash
# Install required system packages
sudo apt update
sudo apt install -y python3-pip python3-venv git nginx supervisor
sudo systemctl enable nginx supervisor
```

### 🔧 Server Setup

#### 1. User and Directory Setup
```bash
# Create deployment user
sudo useradd -m -s /bin/bash beautyai
sudo usermod -aG sudo beautyai

# Create application directories
sudo mkdir -p /opt/beautyai
sudo chown beautyai:beautyai /opt/beautyai

# Switch to deployment user
sudo su - beautyai
```

#### 2. Application Deployment
```bash
# Clone the repository
cd /opt
git clone https://github.com/your-org/beautyai.git
cd beautyai

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install backend dependencies
cd backend
pip install -r requirements.txt
pip install -e .

# Install frontend dependencies
cd ../frontend
pip install -r requirements.txt

# Download initial models
cd ../tools
python download_models.py --model qwen3-14b-instruct
```

#### 3. Configuration Setup
```bash
# Create production configuration
sudo mkdir -p /etc/beautyai
sudo cp backend/src/model_registry.json /etc/beautyai/
sudo cp frontend/config.json /etc/beautyai/frontend_config.json

# Set secure permissions
sudo chown -R beautyai:beautyai /etc/beautyai
sudo chmod 640 /etc/beautyai/*.json
```

### 🔄 Systemd Services

#### Backend Service
```bash
# Create backend service file
sudo tee /etc/systemd/system/beautyai-api.service > /dev/null << 'EOF'
[Unit]
Description=BeautyAI Backend API Server
After=network.target

[Service]
Type=simple
User=beautyai
Group=beautyai
WorkingDirectory=/opt/beautyai/backend
Environment=PATH=/opt/beautyai/venv/bin
Environment=PYTHONPATH=/opt/beautyai/backend/src
Environment=BEAUTYAI_CONFIG_PATH=/etc/beautyai
ExecStart=/opt/beautyai/venv/bin/python run_server.py
Restart=always
RestartSec=10
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=beautyai-api

# Security settings
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/beautyai /etc/beautyai /tmp
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF
```

#### Frontend Service
```bash
# Create frontend service file
sudo tee /etc/systemd/system/beautyai-webui.service > /dev/null << 'EOF'
[Unit]
Description=BeautyAI Frontend Web UI
After=network.target beautyai-api.service

[Service]
Type=simple
User=beautyai
Group=beautyai
WorkingDirectory=/opt/beautyai/frontend
Environment=PATH=/opt/beautyai/venv/bin
Environment=PYTHONPATH=/opt/beautyai/frontend
Environment=BEAUTYAI_CONFIG_PATH=/etc/beautyai
ExecStart=/opt/beautyai/venv/bin/python src/app.py
Restart=always
RestartSec=10
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=beautyai-webui

# Security settings
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/beautyai /etc/beautyai /tmp
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF
```

#### Enable and Start Services
```bash
# Reload systemd and enable services
sudo systemctl daemon-reload
sudo systemctl enable beautyai-api beautyai-webui

# Start services
sudo systemctl start beautyai-api
sudo systemctl start beautyai-webui

# Check status
sudo systemctl status beautyai-api beautyai-webui
```

### 🌐 Nginx Configuration

#### SSL Certificate Setup
```bash
# Install Certbot for Let's Encrypt
sudo apt install -y certbot python3-certbot-nginx

# Generate SSL certificate (replace your-domain.com)
sudo certbot --nginx -d your-domain.com
```

#### Nginx Configuration
```bash
# Create main configuration
sudo tee /etc/nginx/sites-available/beautyai > /dev/null << 'EOF'
# Upstream servers
upstream beautyai_api {
    server 127.0.0.1:8000 fail_timeout=0;
}

upstream beautyai_webui {
    server 127.0.0.1:5001 fail_timeout=0;
}

# HTTP to HTTPS redirect
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

# Main HTTPS server
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    # SSL configuration
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload";
    
    # General settings
    client_max_body_size 100M;
    proxy_read_timeout 300s;
    proxy_connect_timeout 75s;
    
    # API endpoints
    location /api/ {
        proxy_pass http://beautyai_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # WebSocket endpoints
    location /ws/ {
        proxy_pass http://beautyai_api;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
    
    # Frontend web UI
    location / {
        proxy_pass http://beautyai_webui;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Static files (if served directly by Nginx)
    location /static/ {
        alias /opt/beautyai/frontend/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
EOF

# Enable the site
sudo ln -sf /etc/nginx/sites-available/beautyai /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test and reload Nginx
sudo nginx -t
sudo systemctl reload nginx
```

### 🔐 Security Configuration

#### Firewall Setup
```bash
# Configure UFW firewall
sudo ufw --force reset
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH, HTTP, HTTPS
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Enable firewall
sudo ufw --force enable
sudo ufw status
```

#### Log Rotation
```bash
# Create log rotation configuration
sudo tee /etc/logrotate.d/beautyai > /dev/null << 'EOF'
/var/log/beautyai/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 beautyai beautyai
    postrotate
        systemctl reload beautyai-api beautyai-webui
    endscript
}
EOF

# Create log directory
sudo mkdir -p /var/log/beautyai
sudo chown beautyai:beautyai /var/log/beautyai
```

### 📊 Monitoring Setup

#### Health Check Script
```bash
# Create health check script
sudo tee /opt/beautyai/scripts/health_check.sh > /dev/null << 'EOF'
#!/bin/bash

# Health check for BeautyAI services
API_URL="http://localhost:8000/api/health"
WEBUI_URL="http://localhost:5001/health"

# Check API service
if curl -f -s "$API_URL" > /dev/null; then
    echo "API: OK"
else
    echo "API: FAILED"
    systemctl restart beautyai-api
fi

# Check WebUI service
if curl -f -s "$WEBUI_URL" > /dev/null; then
    echo "WebUI: OK"
else
    echo "WebUI: FAILED"
    systemctl restart beautyai-webui
fi
EOF

sudo chmod +x /opt/beautyai/scripts/health_check.sh
sudo chown beautyai:beautyai /opt/beautyai/scripts/health_check.sh
```

#### Cron Job for Health Checks
```bash
# Add to crontab for beautyai user
sudo -u beautyai crontab -e

# Add this line (check every 5 minutes):
*/5 * * * * /opt/beautyai/scripts/health_check.sh >> /var/log/beautyai/health_check.log 2>&1
```

## 🐳 Docker Deployment

### Dockerfile
```dockerfile
# Create Dockerfile
tee Dockerfile > /dev/null << 'EOF'
FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN useradd -m -u 1000 beautyai

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Create virtual environment and install dependencies
RUN python3 -m venv venv && \
    . venv/bin/activate && \
    pip install --upgrade pip && \
    cd backend && pip install -r requirements.txt && pip install -e . && \
    cd ../frontend && pip install -r requirements.txt

# Set ownership
RUN chown -R beautyai:beautyai /app

# Switch to application user
USER beautyai

# Expose ports
EXPOSE 8000 5001

# Start script
COPY docker-entrypoint.sh /usr/local/bin/
ENTRYPOINT ["docker-entrypoint.sh"]
EOF
```

### Docker Compose
```yaml
# Create docker-compose.yml
tee docker-compose.yml > /dev/null << 'EOF'
version: '3.8'

services:
  beautyai-api:
    build: .
    command: ["api"]
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./config:/app/config
    environment:
      - BEAUTYAI_CONFIG_PATH=/app/config
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  beautyai-webui:
    build: .
    command: ["webui"]
    ports:
      - "5001:5001"
    volumes:
      - ./config:/app/config
    environment:
      - BEAUTYAI_CONFIG_PATH=/app/config
      - BEAUTYAI_API_URL=http://beautyai-api:8000
    depends_on:
      - beautyai-api
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl/certs
    depends_on:
      - beautyai-api
      - beautyai-webui
    restart: unless-stopped
EOF
```

### Docker Entry Point
```bash
# Create docker-entrypoint.sh
tee docker-entrypoint.sh > /dev/null << 'EOF'
#!/bin/bash
set -e

# Activate virtual environment
source /app/venv/bin/activate

case "$1" in
    api)
        echo "Starting BeautyAI API server..."
        cd /app/backend
        exec python run_server.py
        ;;
    webui)
        echo "Starting BeautyAI Web UI..."
        cd /app/frontend
        exec python src/app.py
        ;;
    *)
        echo "Usage: $0 {api|webui}"
        exit 1
        ;;
esac
EOF

chmod +x docker-entrypoint.sh
```

## 🔄 Maintenance

### Update Procedure
```bash
# 1. Create backup
sudo systemctl stop beautyai-api beautyai-webui
sudo tar -czf /opt/backups/beautyai-$(date +%Y%m%d).tar.gz /opt/beautyai

# 2. Update code
cd /opt/beautyai
sudo -u beautyai git pull origin main

# 3. Update dependencies
sudo -u beautyai bash -c "source venv/bin/activate && pip install -r backend/requirements.txt && pip install -r frontend/requirements.txt"

# 4. Restart services
sudo systemctl start beautyai-api beautyai-webui
sudo systemctl status beautyai-api beautyai-webui
```

### Log Management
```bash
# View service logs
sudo journalctl -u beautyai-api -f
sudo journalctl -u beautyai-webui -f

# View application logs
sudo tail -f /var/log/beautyai/*.log

# Check system resources
sudo systemctl status beautyai-api beautyai-webui
sudo ps aux | grep python
sudo nvidia-smi
```

### Troubleshooting Commands
```bash
# Check service status
sudo systemctl status beautyai-api beautyai-webui nginx

# Check port bindings
sudo netstat -tlnp | grep -E ':(8000|5001|80|443)'

# Check GPU usage
nvidia-smi
watch -n 1 nvidia-smi

# Test API endpoints
curl -X GET http://localhost:8000/api/health
curl -X GET http://localhost:5001/health

# Check configuration
sudo -u beautyai python -c "import json; print(json.load(open('/etc/beautyai/model_registry.json')))"
```

---

**Next**: [Monitoring Guide](MONITORING.md) | [Backup Guide](BACKUP.md)
