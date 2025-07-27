#!/bin/bash
# ================================================================
# Nginx WebSocket Configuration Fix Deployment Script
# ================================================================

set -e  # Exit on any error

echo "🔧 BeautyAI Nginx WebSocket Configuration Fix"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration paths
LOCAL_CONFIG="/home/lumi/beautyai/nginx-clean-config.conf"
DEPLOYED_CONFIG="/etc/nginx/sites-enabled/gmai.sa"
BACKUP_DIR="/etc/nginx/backups"

echo -e "${BLUE}Step 1: Validating local configuration...${NC}"
if [ ! -f "$LOCAL_CONFIG" ]; then
    echo -e "${RED}❌ Error: Local configuration file not found: $LOCAL_CONFIG${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Local configuration found${NC}"

echo -e "${BLUE}Step 2: Creating backup directory...${NC}"
sudo mkdir -p "$BACKUP_DIR"
echo -e "${GREEN}✅ Backup directory ready${NC}"

echo -e "${BLUE}Step 3: Backing up current configuration...${NC}"
BACKUP_FILE="$BACKUP_DIR/gmai.sa.backup.$(date +%Y%m%d_%H%M%S)"
sudo cp "$DEPLOYED_CONFIG" "$BACKUP_FILE"
echo -e "${GREEN}✅ Current configuration backed up to: $BACKUP_FILE${NC}"

echo -e "${BLUE}Step 4: Showing configuration differences...${NC}"
echo -e "${YELLOW}Changes to be applied:${NC}"
diff -u "$DEPLOYED_CONFIG" "$LOCAL_CONFIG" || true

echo -e "${BLUE}Step 5: Applying the fixed configuration...${NC}"
sudo cp "$LOCAL_CONFIG" "$DEPLOYED_CONFIG"
echo -e "${GREEN}✅ Configuration updated${NC}"

echo -e "${BLUE}Step 6: Testing nginx configuration syntax...${NC}"
if sudo nginx -t; then
    echo -e "${GREEN}✅ Nginx configuration syntax is valid${NC}"
else
    echo -e "${RED}❌ Error: Nginx configuration has syntax errors${NC}"
    echo -e "${YELLOW}Restoring backup...${NC}"
    sudo cp "$BACKUP_FILE" "$DEPLOYED_CONFIG"
    echo -e "${GREEN}✅ Backup restored${NC}"
    exit 1
fi

echo -e "${BLUE}Step 7: Reloading nginx configuration...${NC}"
if sudo systemctl reload nginx; then
    echo -e "${GREEN}✅ Nginx configuration reloaded successfully${NC}"
else
    echo -e "${RED}❌ Error: Failed to reload nginx${NC}"
    echo -e "${YELLOW}Restoring backup...${NC}"
    sudo cp "$BACKUP_FILE" "$DEPLOYED_CONFIG"
    sudo systemctl reload nginx
    echo -e "${GREEN}✅ Backup restored and nginx reloaded${NC}"
    exit 1
fi

echo -e "${BLUE}Step 8: Validating nginx service status...${NC}"
if systemctl is-active --quiet nginx; then
    echo -e "${GREEN}✅ Nginx service is running${NC}"
else
    echo -e "${RED}❌ Warning: Nginx service is not running${NC}"
fi

echo ""
echo -e "${GREEN}🎉 Nginx WebSocket Configuration Fix Applied Successfully!${NC}"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "1. Test WebSocket connectivity:"
echo "   python test_websocket_nginx_fix.py"
echo ""
echo "2. Ensure backend FastAPI service is running on localhost:8000"
echo ""
echo "3. If issues persist, check the backup at:"
echo "   $BACKUP_FILE"
echo ""
echo -e "${BLUE}Expected results after fix:${NC}"
echo "• WebSocket connections should work (no more HTTP 502 errors)"
echo "• /api/v1/ws/simple-voice-chat should be accessible"
echo "• Legacy /ws/simple-voice-chat should work for backward compatibility"
