#!/bin/bash
# Installation script for BeautyAI PABX system

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="$SCRIPT_DIR/venv"

echo "================================="
echo "BeautyAI PABX Installation"
echo "================================="
echo ""

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "This script should NOT be run as root"
   exit 1
fi

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || [ "$PYTHON_MINOR" -lt 8 ]; then
    echo "Error: Python 3.8 or higher is required"
    echo "Current version: $PYTHON_VERSION"
    exit 1
fi

echo "Python version: $PYTHON_VERSION ✓"
echo ""

# Check system dependencies
echo "Checking system dependencies..."
MISSING_DEPS=()

# Check for required packages
for pkg in portaudio19-dev python3-dev python3-venv; do
    if ! dpkg -l | grep -q "^ii  $pkg"; then
        MISSING_DEPS+=($pkg)
    fi
done

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo "Missing system dependencies: ${MISSING_DEPS[@]}"
    echo ""
    echo "Install with:"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install ${MISSING_DEPS[@]}"
    echo ""
    read -p "Do you want to install them now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo apt-get update
        sudo apt-get install -y ${MISSING_DEPS[@]}
    else
        echo "Please install dependencies manually and run this script again"
        exit 1
    fi
fi

echo "System dependencies ✓"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created ✓"
else
    echo "Virtual environment already exists ✓"
fi
echo ""

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo ""

# Install dependencies
echo "Installing Python dependencies..."
pip install -r "$SCRIPT_DIR/requirements.txt"
echo "Dependencies installed ✓"
echo ""

# Create necessary directories
echo "Creating directories..."
mkdir -p "$SCRIPT_DIR/logs/system"
mkdir -p "$SCRIPT_DIR/logs/sessions"
mkdir -p "$SCRIPT_DIR/recordings"
mkdir -p "$SCRIPT_DIR/captures"
echo "Directories created ✓"
echo ""

# Set up systemd services
echo "Setting up systemd services..."
read -p "Do you want to install systemd services? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Update service files with actual paths
    sed -i "s|/home/lumi|$HOME|g" "$SCRIPT_DIR/pabx-server.service"
    sed -i "s|User=lumi|User=$USER|g" "$SCRIPT_DIR/pabx-server.service"
    
    sed -i "s|/home/lumi|$HOME|g" "$SCRIPT_DIR/pabx-sniffer.service"
    sed -i "s|User=lumi|User=$USER|g" "$SCRIPT_DIR/pabx-sniffer.service"
    
    # Copy service files
    sudo cp "$SCRIPT_DIR/pabx-server.service" /etc/systemd/system/
    sudo cp "$SCRIPT_DIR/pabx-sniffer.service" /etc/systemd/system/
    
    # Set capabilities for Python interpreter (for packet capture)
    sudo setcap cap_net_raw,cap_net_admin+eip "$VENV_DIR/bin/python3"
    
    # Reload systemd
    sudo systemctl daemon-reload
    
    echo "Systemd services installed ✓"
    echo ""
    echo "To enable and start services:"
    echo "  sudo systemctl enable pabx-server"
    echo "  sudo systemctl start pabx-server"
    echo "  sudo systemctl enable pabx-sniffer"
    echo "  sudo systemctl start pabx-sniffer"
    echo ""
fi

# Configuration
echo "Configuration:"
echo "  Edit config/settings.yaml to customize settings"
echo "  Edit config/devices.json to add HT813 devices"
echo ""

# Usage
echo "================================="
echo "Installation Complete!"
echo "================================="
echo ""
echo "To run the server manually:"
echo "  source venv/bin/activate"
echo "  ./run_server.py --mode api"
echo ""
echo "To run SIP server only:"
echo "  ./run_server.py --mode sip"
echo ""
echo "API will be available at: http://localhost:8080"
echo "API documentation: http://localhost:8080/docs"
echo ""
echo "Check logs:"
echo "  tail -f logs/system/app.json"
echo "  sudo journalctl -u pabx-server -f"
echo ""
