#!/bin/bash
#
# BeautyAI API Service Management Script
# Usage: ./manage-api-service.sh [install|uninstall|start|stop|restart|status|logs|enable|disable]
#

set -e

SERVICE_NAME="beautyai-api"
SERVICE_FILE="/home/lumi/beautyai/beautyai-api.service"
SYSTEMD_PATH="/etc/systemd/system/${SERVICE_NAME}.service"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_service_file() {
    if [[ ! -f "$SERVICE_FILE" ]]; then
        print_error "Service file not found at $SERVICE_FILE"
        exit 1
    fi
}

install_service() {
    print_status "Installing BeautyAI API service..."
    
    check_service_file
    
    # Copy service file to systemd directory
    sudo cp "$SERVICE_FILE" "$SYSTEMD_PATH"
    print_success "Service file copied to $SYSTEMD_PATH"
    
    # Reload systemd daemon
    sudo systemctl daemon-reload
    print_success "Systemd daemon reloaded"
    
    # Set permissions
    sudo chmod 644 "$SYSTEMD_PATH"
    
    print_success "BeautyAI API service installed successfully!"
    print_status "You can now use: sudo systemctl [start|stop|restart|status] $SERVICE_NAME"
    print_status "To enable auto-start on boot: sudo systemctl enable $SERVICE_NAME"
}

uninstall_service() {
    print_status "Uninstalling BeautyAI API service..."
    
    # Stop service if running
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        sudo systemctl stop "$SERVICE_NAME"
        print_status "Service stopped"
    fi
    
    # Disable service if enabled
    if systemctl is-enabled --quiet "$SERVICE_NAME"; then
        sudo systemctl disable "$SERVICE_NAME"
        print_status "Service disabled"
    fi
    
    # Remove service file
    if [[ -f "$SYSTEMD_PATH" ]]; then
        sudo rm "$SYSTEMD_PATH"
        print_success "Service file removed"
    fi
    
    # Reload systemd daemon
    sudo systemctl daemon-reload
    print_success "BeautyAI API service uninstalled successfully!"
}

start_service() {
    print_status "Starting BeautyAI API service..."
    sudo systemctl start "$SERVICE_NAME"
    print_success "Service started!"
    
    # Show status
    sleep 2
    show_status
}

stop_service() {
    print_status "Stopping BeautyAI API service..."
    sudo systemctl stop "$SERVICE_NAME"
    print_success "Service stopped!"
}

restart_service() {
    print_status "Restarting BeautyAI API service..."
    sudo systemctl restart "$SERVICE_NAME"
    print_success "Service restarted!"
    
    # Show status
    sleep 2
    show_status
}

show_status() {
    print_status "BeautyAI API Service Status:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Service status
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        echo -e "${GREEN}● Active${NC} - BeautyAI API is running"
    else
        echo -e "${RED}● Inactive${NC} - BeautyAI API is not running"
    fi
    
    # Enabled status
    if systemctl is-enabled --quiet "$SERVICE_NAME" 2>/dev/null; then
        echo -e "${GREEN}● Enabled${NC} - Will start automatically on boot"
    else
        echo -e "${YELLOW}● Disabled${NC} - Will not start automatically on boot"
    fi
    
    echo ""
    
    # Detailed status
    sudo systemctl status "$SERVICE_NAME" --no-pager --lines=5 || true
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Check if API is responding
    if curl -s http://localhost:8000/ > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} API is responding on http://localhost:8000"
    else
        echo -e "${RED}✗${NC} API is not responding on http://localhost:8000"
    fi
}

show_logs() {
    print_status "Showing BeautyAI API service logs (last 50 lines)..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Show recent logs
    sudo journalctl -u "$SERVICE_NAME" -n 50 --no-pager
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    print_status "To follow logs in real-time: sudo journalctl -u $SERVICE_NAME -f"
}

enable_service() {
    print_status "Enabling BeautyAI API service to start on boot..."
    sudo systemctl enable "$SERVICE_NAME"
    print_success "Service enabled! It will start automatically on boot."
}

disable_service() {
    print_status "Disabling BeautyAI API service auto-start..."
    sudo systemctl disable "$SERVICE_NAME"
    print_success "Service disabled! It will not start automatically on boot."
}

show_help() {
    echo "BeautyAI API Service Management Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  install    Install the systemd service"
    echo "  uninstall  Remove the systemd service"
    echo "  start      Start the API service"
    echo "  stop       Stop the API service"
    echo "  restart    Restart the API service"
    echo "  status     Show service status and API health"
    echo "  logs       Show recent service logs"
    echo "  enable     Enable auto-start on boot"
    echo "  disable    Disable auto-start on boot"
    echo "  help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 install    # Install the service"
    echo "  $0 start      # Start the API server"
    echo "  $0 status     # Check if API is running"
    echo "  $0 logs       # View recent logs"
    echo ""
    echo "After installation, you can also use standard systemctl commands:"
    echo "  sudo systemctl start $SERVICE_NAME"
    echo "  sudo systemctl stop $SERVICE_NAME"
    echo "  sudo systemctl status $SERVICE_NAME"
    echo "  sudo journalctl -u $SERVICE_NAME -f"
}

# Main command handling
case "${1:-help}" in
    install)
        install_service
        ;;
    uninstall)
        uninstall_service
        ;;
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        restart_service
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    enable)
        enable_service
        ;;
    disable)
        disable_service
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
