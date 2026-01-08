#!/bin/bash
# Install PABX systemd services
# Run with: sudo ./install_services.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PABX_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  BeautyAI PABX Service Installer      ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}❌ Please run as root: sudo ./install_services.sh${NC}"
    exit 1
fi

echo -e "${YELLOW}📋 Installation Options:${NC}"
echo ""
echo "  1) Development Mode (npm run dev)"
echo "  2) Production Mode (npm run preview - requires build)"
echo "  3) Backend Only"
echo "  4) Uninstall Services"
echo ""
read -p "Select option [1-4]: " option

case $option in
    1)
        MODE="dev"
        FRONTEND_SERVICE="pabx-frontend.service"
        echo -e "${GREEN}✓ Installing Development Mode${NC}"
        ;;
    2)
        MODE="prod"
        FRONTEND_SERVICE="pabx-frontend-prod.service"
        echo -e "${GREEN}✓ Installing Production Mode${NC}"
        
        # Check if build exists
        if [ ! -d "$PABX_DIR/ui/dist" ]; then
            echo -e "${YELLOW}⚠️  Production build not found. Building now...${NC}"
            cd "$PABX_DIR/ui"
            sudo -u lumi npm run build
            echo -e "${GREEN}✓ Build complete${NC}"
        fi
        ;;
    3)
        MODE="backend"
        echo -e "${GREEN}✓ Installing Backend Only${NC}"
        ;;
    4)
        echo -e "${YELLOW}🗑️  Uninstalling services...${NC}"
        systemctl stop pabx-backend.service 2>/dev/null || true
        systemctl stop pabx-frontend.service 2>/dev/null || true
        systemctl stop pabx-frontend-prod.service 2>/dev/null || true
        systemctl disable pabx-backend.service 2>/dev/null || true
        systemctl disable pabx-frontend.service 2>/dev/null || true
        systemctl disable pabx-frontend-prod.service 2>/dev/null || true
        rm -f /etc/systemd/system/pabx-backend.service
        rm -f /etc/systemd/system/pabx-frontend.service
        rm -f /etc/systemd/system/pabx-frontend-prod.service
        systemctl daemon-reload
        echo -e "${GREEN}✅ Services uninstalled${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}❌ Invalid option${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${BLUE}Installing services...${NC}"
echo ""

# Install backend service
echo -e "${YELLOW}1/3${NC} Installing backend service..."
cp "$SCRIPT_DIR/pabx-backend.service" /etc/systemd/system/
chmod 644 /etc/systemd/system/pabx-backend.service
echo -e "${GREEN}✓ Backend service installed${NC}"

# Install frontend service if not backend-only
if [ "$MODE" != "backend" ]; then
    echo -e "${YELLOW}2/3${NC} Installing frontend service..."
    cp "$SCRIPT_DIR/$FRONTEND_SERVICE" /etc/systemd/system/
    chmod 644 /etc/systemd/system/$FRONTEND_SERVICE
    echo -e "${GREEN}✓ Frontend service installed${NC}"
fi

# Reload systemd
echo -e "${YELLOW}3/3${NC} Reloading systemd daemon..."
systemctl daemon-reload
echo -e "${GREEN}✓ Systemd reloaded${NC}"

echo ""
echo -e "${BLUE}═══════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Installation Complete!${NC}"
echo -e "${BLUE}═══════════════════════════════════════${NC}"
echo ""

# Enable and start services
read -p "Enable services to start on boot? [Y/n]: " enable_boot
if [[ $enable_boot != "n" && $enable_boot != "N" ]]; then
    echo ""
    echo -e "${YELLOW}Enabling services...${NC}"
    systemctl enable pabx-backend.service
    if [ "$MODE" != "backend" ]; then
        systemctl enable $FRONTEND_SERVICE
    fi
    echo -e "${GREEN}✓ Services enabled${NC}"
fi

read -p "Start services now? [Y/n]: " start_now
if [[ $start_now != "n" && $start_now != "N" ]]; then
    echo ""
    echo -e "${YELLOW}Starting services...${NC}"
    systemctl start pabx-backend.service
    sleep 2
    
    if [ "$MODE" != "backend" ]; then
        systemctl start $FRONTEND_SERVICE
        sleep 2
    fi
    
    echo -e "${GREEN}✓ Services started${NC}"
fi

echo ""
echo -e "${BLUE}═══════════════════════════════════════${NC}"
echo -e "${GREEN}🎉 PABX System Ready!${NC}"
echo -e "${BLUE}═══════════════════════════════════════${NC}"
echo ""

# Show status
echo -e "${YELLOW}📊 Service Status:${NC}"
echo ""
systemctl status pabx-backend.service --no-pager -l || true
echo ""
if [ "$MODE" != "backend" ]; then
    systemctl status $FRONTEND_SERVICE --no-pager -l || true
fi

echo ""
echo -e "${BLUE}═══════════════════════════════════════${NC}"
echo -e "${YELLOW}📋 Service Management Commands:${NC}"
echo -e "${BLUE}═══════════════════════════════════════${NC}"
echo ""
echo -e "  ${GREEN}Start:${NC}"
echo "    sudo systemctl start pabx-backend"
if [ "$MODE" != "backend" ]; then
    echo "    sudo systemctl start $FRONTEND_SERVICE"
fi
echo ""
echo -e "  ${YELLOW}Stop:${NC}"
echo "    sudo systemctl stop pabx-backend"
if [ "$MODE" != "backend" ]; then
    echo "    sudo systemctl stop $FRONTEND_SERVICE"
fi
echo ""
echo -e "  ${BLUE}Restart:${NC}"
echo "    sudo systemctl restart pabx-backend"
if [ "$MODE" != "backend" ]; then
    echo "    sudo systemctl restart $FRONTEND_SERVICE"
fi
echo ""
echo -e "  ${GREEN}Status:${NC}"
echo "    sudo systemctl status pabx-backend"
if [ "$MODE" != "backend" ]; then
    echo "    sudo systemctl status $FRONTEND_SERVICE"
fi
echo ""
echo -e "  ${YELLOW}Logs:${NC}"
echo "    sudo journalctl -u pabx-backend -f"
if [ "$MODE" != "backend" ]; then
    echo "    sudo journalctl -u $FRONTEND_SERVICE -f"
fi
echo ""
echo -e "  ${RED}Disable (stop auto-start):${NC}"
echo "    sudo systemctl disable pabx-backend"
if [ "$MODE" != "backend" ]; then
    echo "    sudo systemctl disable $FRONTEND_SERVICE"
fi
echo ""

echo -e "${BLUE}═══════════════════════════════════════${NC}"
echo -e "${GREEN}🌐 Access Points:${NC}"
echo -e "${BLUE}═══════════════════════════════════════${NC}"
echo ""
if [ "$MODE" != "backend" ]; then
    echo "  Frontend:  http://localhost:3000"
fi
echo "  Backend:   http://localhost:8080"
echo "  API Docs:  http://localhost:8080/docs"
echo "  WebSocket: ws://localhost:8080/ws"
echo ""
