#!/bin/bash
# PABX Service Manager - Quick control of systemd services
# Usage: ./pabx-service.sh [start|stop|restart|status|logs|enable|disable]

BACKEND_SERVICE="pabx-backend.service"
FRONTEND_DEV_SERVICE="pabx-frontend.service"
FRONTEND_PROD_SERVICE="pabx-frontend-prod.service"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

show_usage() {
    echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║     PABX Service Manager               ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
    echo ""
    echo "Usage: $0 [command] [service]"
    echo ""
    echo "Commands:"
    echo "  start      - Start services"
    echo "  stop       - Stop services"
    echo "  restart    - Restart services"
    echo "  status     - Show service status"
    echo "  logs       - Follow service logs"
    echo "  enable     - Enable auto-start on boot"
    echo "  disable    - Disable auto-start"
    echo ""
    echo "Services (optional):"
    echo "  backend    - Backend API only"
    echo "  frontend   - Frontend only (dev or prod)"
    echo "  all        - Both services (default)"
    echo ""
    echo "Examples:"
    echo "  $0 start           # Start all services"
    echo "  $0 stop backend    # Stop backend only"
    echo "  $0 logs frontend   # Follow frontend logs"
    echo "  $0 restart         # Restart all services"
    exit 1
}

# Detect which frontend service is installed
detect_frontend_service() {
    if systemctl list-unit-files | grep -q "$FRONTEND_PROD_SERVICE"; then
        echo "$FRONTEND_PROD_SERVICE"
    elif systemctl list-unit-files | grep -q "$FRONTEND_DEV_SERVICE"; then
        echo "$FRONTEND_DEV_SERVICE"
    else
        echo ""
    fi
}

FRONTEND_SERVICE=$(detect_frontend_service)

# Check if services are installed
check_installed() {
    if ! systemctl list-unit-files | grep -q "$BACKEND_SERVICE"; then
        echo -e "${RED}❌ Services not installed!${NC}"
        echo -e "${YELLOW}Run: sudo ./systemd/install_services.sh${NC}"
        exit 1
    fi
}

# Parse arguments
COMMAND=${1:-status}
TARGET=${2:-all}

case $COMMAND in
    start|stop|restart|enable|disable)
        check_installed
        
        case $TARGET in
            backend)
                echo -e "${YELLOW}${COMMAND^}ing backend...${NC}"
                sudo systemctl $COMMAND $BACKEND_SERVICE
                echo -e "${GREEN}✓ Done${NC}"
                ;;
            frontend)
                if [ -z "$FRONTEND_SERVICE" ]; then
                    echo -e "${RED}❌ Frontend service not installed${NC}"
                    exit 1
                fi
                echo -e "${YELLOW}${COMMAND^}ing frontend...${NC}"
                sudo systemctl $COMMAND $FRONTEND_SERVICE
                echo -e "${GREEN}✓ Done${NC}"
                ;;
            all|*)
                echo -e "${YELLOW}${COMMAND^}ing all services...${NC}"
                sudo systemctl $COMMAND $BACKEND_SERVICE
                if [ -n "$FRONTEND_SERVICE" ]; then
                    sudo systemctl $COMMAND $FRONTEND_SERVICE
                fi
                echo -e "${GREEN}✓ Done${NC}"
                ;;
        esac
        ;;
        
    status)
        check_installed
        
        case $TARGET in
            backend)
                sudo systemctl status $BACKEND_SERVICE --no-pager -l
                ;;
            frontend)
                if [ -z "$FRONTEND_SERVICE" ]; then
                    echo -e "${RED}❌ Frontend service not installed${NC}"
                    exit 1
                fi
                sudo systemctl status $FRONTEND_SERVICE --no-pager -l
                ;;
            all|*)
                echo -e "${BLUE}═══════════════════════════════════════${NC}"
                echo -e "${GREEN}Backend Service:${NC}"
                echo -e "${BLUE}═══════════════════════════════════════${NC}"
                sudo systemctl status $BACKEND_SERVICE --no-pager -l
                echo ""
                if [ -n "$FRONTEND_SERVICE" ]; then
                    echo -e "${BLUE}═══════════════════════════════════════${NC}"
                    echo -e "${GREEN}Frontend Service:${NC}"
                    echo -e "${BLUE}═══════════════════════════════════════${NC}"
                    sudo systemctl status $FRONTEND_SERVICE --no-pager -l
                fi
                ;;
        esac
        ;;
        
    logs)
        check_installed
        
        case $TARGET in
            backend)
                echo -e "${YELLOW}Following backend logs (Ctrl+C to exit)...${NC}"
                sudo journalctl -u $BACKEND_SERVICE -f
                ;;
            frontend)
                if [ -z "$FRONTEND_SERVICE" ]; then
                    echo -e "${RED}❌ Frontend service not installed${NC}"
                    exit 1
                fi
                echo -e "${YELLOW}Following frontend logs (Ctrl+C to exit)...${NC}"
                sudo journalctl -u $FRONTEND_SERVICE -f
                ;;
            all|*)
                echo -e "${YELLOW}Following all logs (Ctrl+C to exit)...${NC}"
                if [ -n "$FRONTEND_SERVICE" ]; then
                    sudo journalctl -u $BACKEND_SERVICE -u $FRONTEND_SERVICE -f
                else
                    sudo journalctl -u $BACKEND_SERVICE -f
                fi
                ;;
        esac
        ;;
        
    help|--help|-h)
        show_usage
        ;;
        
    *)
        echo -e "${RED}❌ Unknown command: $COMMAND${NC}"
        echo ""
        show_usage
        ;;
esac
