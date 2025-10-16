#!/bin/bash
# ================================================================
# BeautyAI Alibaba Cloud Deployment Script
# ================================================================
# This script helps deploy BeautyAI services on Alibaba Cloud
# Domains: web.lumidev.ca, api.lumidev.ca
# Date: October 16, 2025
# ================================================================

set -e  # Exit on error

echo "🚀 BeautyAI Alibaba Cloud Deployment Script"
echo "============================================="
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Base directory
BASE_DIR="/home/geniusai/geniusAI/mistral_env"

# Function to print status
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    print_error "Please run as root (use sudo)"
    exit 1
fi

print_status "Running as root"

# ================================================================
# Step 1: Install Certbot for Let's Encrypt SSL
# ================================================================
echo ""
echo "Step 1: Installing Certbot for SSL certificates"
echo "------------------------------------------------"

if ! command -v certbot &> /dev/null; then
    print_warning "Certbot not found. Installing..."
    apt-get update
    apt-get install -y certbot python3-certbot-nginx
    print_status "Certbot installed"
else
    print_status "Certbot already installed"
fi

# ================================================================
# Step 2: Generate DH parameters for SSL
# ================================================================
echo ""
echo "Step 2: Generating DH parameters (this may take a while)"
echo "---------------------------------------------------------"

if [ ! -f /etc/nginx/dhparam.pem ]; then
    print_warning "Generating DH parameters..."
    openssl dhparam -out /etc/nginx/dhparam.pem 2048
    print_status "DH parameters generated"
else
    print_status "DH parameters already exist"
fi

# ================================================================
# Step 3: Create certbot webroot directory
# ================================================================
echo ""
echo "Step 3: Creating certbot webroot directory"
echo "-------------------------------------------"

mkdir -p /var/www/certbot
print_status "Certbot webroot created"

# ================================================================
# Step 4: Obtain SSL certificates
# ================================================================
echo ""
echo "Step 4: SSL Certificate Setup"
echo "------------------------------"
print_warning "You need to obtain SSL certificates for:"
print_warning "  - web.lumidev.ca"
print_warning "  - api.lumidev.ca"
echo ""
print_warning "Make sure DNS is pointing to this server before continuing!"
echo ""
read -p "Do you want to obtain SSL certificates now? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Obtain certificate for web.lumidev.ca
    print_status "Obtaining certificate for web.lumidev.ca..."
    certbot certonly --nginx \
        -d web.lumidev.ca \
        --non-interactive \
        --agree-tos \
        --email admin@lumidev.ca || print_error "Failed to obtain certificate for web.lumidev.ca"
    
    # Obtain certificate for api.lumidev.ca
    print_status "Obtaining certificate for api.lumidev.ca..."
    certbot certonly --nginx \
        -d api.lumidev.ca \
        --non-interactive \
        --agree-tos \
        --email admin@lumidev.ca || print_error "Failed to obtain certificate for api.lumidev.ca"
    
    print_status "SSL certificates obtained successfully"
else
    print_warning "Skipping SSL certificate generation"
    print_warning "You can run certbot manually later:"
    echo "  certbot certonly --nginx -d web.lumidev.ca"
    echo "  certbot certonly --nginx -d api.lumidev.ca"
fi

# ================================================================
# Step 5: Install systemd service files
# ================================================================
echo ""
echo "Step 5: Installing systemd service files"
echo "-----------------------------------------"

# Copy service files
cp ${BASE_DIR}/beautyai-api.service.alibaba /etc/systemd/system/beautyai-api.service
cp ${BASE_DIR}/beautyai-webui.service.alibaba /etc/systemd/system/beautyai-webui.service

print_status "Service files installed"

# Reload systemd
systemctl daemon-reload
print_status "Systemd daemon reloaded"

# ================================================================
# Step 6: Install nginx configuration
# ================================================================
echo ""
echo "Step 6: Installing nginx configuration"
echo "---------------------------------------"

# Backup existing nginx config if it exists
if [ -f /etc/nginx/sites-available/beautyai ]; then
    print_warning "Backing up existing nginx config"
    cp /etc/nginx/sites-available/beautyai /etc/nginx/sites-available/beautyai.backup.$(date +%Y%m%d_%H%M%S)
fi

# Copy nginx config
cp ${BASE_DIR}/nginx-beautyai-config.alibaba.conf /etc/nginx/sites-available/beautyai

# Create symlink if it doesn't exist
if [ ! -L /etc/nginx/sites-enabled/beautyai ]; then
    ln -s /etc/nginx/sites-available/beautyai /etc/nginx/sites-enabled/beautyai
    print_status "Nginx config symlink created"
else
    print_status "Nginx config symlink already exists"
fi

# Test nginx configuration
if nginx -t; then
    print_status "Nginx configuration is valid"
else
    print_error "Nginx configuration test failed"
    exit 1
fi

# ================================================================
# Step 7: Enable and start services
# ================================================================
echo ""
echo "Step 7: Service Management"
echo "--------------------------"

read -p "Do you want to enable and start services now? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Enable services
    systemctl enable beautyai-api.service
    systemctl enable beautyai-webui.service
    print_status "Services enabled"
    
    # Start services
    systemctl start beautyai-api.service
    systemctl start beautyai-webui.service
    print_status "Services started"
    
    # Reload nginx
    systemctl reload nginx
    print_status "Nginx reloaded"
    
    # Show status
    echo ""
    echo "Service Status:"
    echo "---------------"
    systemctl status beautyai-api.service --no-pager -l
    echo ""
    systemctl status beautyai-webui.service --no-pager -l
else
    print_warning "Skipping service start"
    print_warning "You can start services manually:"
    echo "  systemctl start beautyai-api.service"
    echo "  systemctl start beautyai-webui.service"
    echo "  systemctl reload nginx"
fi

# ================================================================
# Step 8: Setup automatic certificate renewal
# ================================================================
echo ""
echo "Step 8: Setting up automatic SSL certificate renewal"
echo "-----------------------------------------------------"

# Certbot should automatically set up renewal, but let's verify
if systemctl is-enabled certbot.timer &> /dev/null; then
    print_status "Certbot auto-renewal is already enabled"
else
    systemctl enable certbot.timer
    systemctl start certbot.timer
    print_status "Certbot auto-renewal enabled"
fi

# ================================================================
# Deployment Complete
# ================================================================
echo ""
echo "============================================="
echo "🎉 Deployment Complete!"
echo "============================================="
echo ""
echo "Your BeautyAI services are configured for:"
echo "  Frontend: https://web.lumidev.ca"
echo "  API:      https://api.lumidev.ca"
echo ""
echo "Useful commands:"
echo "  - View API logs:    journalctl -u beautyai-api.service -f"
echo "  - View WebUI logs:  journalctl -u beautyai-webui.service -f"
echo "  - Restart API:      systemctl restart beautyai-api.service"
echo "  - Restart WebUI:    systemctl restart beautyai-webui.service"
echo "  - Test nginx:       nginx -t"
echo "  - Reload nginx:     systemctl reload nginx"
echo ""
print_warning "Make sure to configure your .env.production file if needed"
print_warning "Location: ${BASE_DIR}/.env.production"
echo ""
