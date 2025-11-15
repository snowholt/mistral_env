#!/bin/bash
# BeautyAI PABX System Test Script
# Quick verification that everything is installed and working

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🧪 BeautyAI PABX System Test"
echo "=============================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

info() {
    echo -e "ℹ️  $1"
}

# Test 1: Python version
echo "1️⃣  Checking Python version..."
if python3 --version &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    success "Python $PYTHON_VERSION installed"
else
    error "Python 3 not found"
    exit 1
fi

# Test 2: Node.js version
echo ""
echo "2️⃣  Checking Node.js version..."
if node --version &> /dev/null; then
    NODE_VERSION=$(node --version)
    success "Node.js $NODE_VERSION installed"
else
    warning "Node.js not found - frontend won't work"
    warning "Install with: curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash - && sudo apt install -y nodejs"
fi

# Test 3: Virtual environment
echo ""
echo "3️⃣  Checking Python virtual environment..."
if [ -d "venv" ]; then
    success "Virtual environment exists"
    
    # Check if packages are installed
    if [ -f "venv/bin/python3" ]; then
        source venv/bin/activate
        
        echo "   Checking packages..."
        MISSING_PACKAGES=()
        
        python3 -c "import fastapi" 2>/dev/null || MISSING_PACKAGES+=("fastapi")
        python3 -c "import uvicorn" 2>/dev/null || MISSING_PACKAGES+=("uvicorn")
        python3 -c "import pyaudio" 2>/dev/null || MISSING_PACKAGES+=("pyaudio")
        python3 -c "import scapy" 2>/dev/null || MISSING_PACKAGES+=("scapy")
        
        if [ ${#MISSING_PACKAGES[@]} -eq 0 ]; then
            success "All required packages installed"
        else
            error "Missing packages: ${MISSING_PACKAGES[*]}"
            info "Install with: source venv/bin/activate && pip install -r requirements.txt"
        fi
        
        deactivate
    fi
else
    error "Virtual environment not found"
    info "Create with: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
fi

# Test 4: Frontend dependencies
echo ""
echo "4️⃣  Checking frontend dependencies..."
if [ -d "ui/node_modules" ]; then
    success "Frontend dependencies installed"
else
    warning "Frontend dependencies not installed"
    info "Install with: cd ui && npm install"
fi

# Test 5: Configuration file
echo ""
echo "5️⃣  Checking configuration..."
if [ -f "config/settings.yaml" ]; then
    success "Configuration file exists"
    
    # Check HT813 IP
    if grep -q "192.168.100.96" config/settings.yaml; then
        info "HT813 IP: 192.168.100.96"
        info "Update in config/settings.yaml if different"
    fi
else
    error "Configuration file not found"
    info "Should be at: config/settings.yaml"
fi

# Test 6: Backend executable
echo ""
echo "6️⃣  Checking backend entry point..."
if [ -f "run_server.py" ] && [ -x "run_server.py" ]; then
    success "run_server.py is executable"
else
    if [ -f "run_server.py" ]; then
        warning "run_server.py exists but not executable"
        info "Fix with: chmod +x run_server.py"
    else
        error "run_server.py not found"
    fi
fi

# Test 7: Port availability
echo ""
echo "7️⃣  Checking port availability..."

check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        return 1
    else
        return 0
    fi
}

if check_port 8080; then
    success "Port 8080 (API) is available"
else
    warning "Port 8080 is already in use"
    info "Process: $(lsof -i :8080 -sTCP:LISTEN | tail -1)"
fi

if check_port 3000; then
    success "Port 3000 (Frontend) is available"
else
    warning "Port 3000 is already in use"
    info "Use different port: npm run dev -- --port 3001"
fi

# Test 8: HT813 connectivity
echo ""
echo "8️⃣  Checking HT813 connectivity..."
HT813_IP="192.168.100.96"

if ping -c 1 -W 2 $HT813_IP &> /dev/null; then
    success "HT813 at $HT813_IP is reachable"
    
    # Try to access web interface
    if curl -s -m 2 http://$HT813_IP &> /dev/null; then
        success "HT813 web interface is accessible"
    else
        warning "HT813 web interface not responding"
    fi
else
    warning "Cannot reach HT813 at $HT813_IP"
    info "Update IP in config/settings.yaml if different"
    info "Or check network connection"
fi

# Test 9: Directory structure
echo ""
echo "9️⃣  Checking directory structure..."
REQUIRED_DIRS=("src" "config" "logs" "ui")
MISSING_DIRS=()

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        MISSING_DIRS+=("$dir")
    fi
done

if [ ${#MISSING_DIRS[@]} -eq 0 ]; then
    success "All required directories exist"
else
    error "Missing directories: ${MISSING_DIRS[*]}"
fi

# Test 10: Permissions
echo ""
echo "🔟 Checking permissions..."
if [ -w "logs" ]; then
    success "Logs directory is writable"
else
    error "Cannot write to logs directory"
    info "Fix with: chmod 755 logs"
fi

# Summary
echo ""
echo "=============================="
echo "📊 Test Summary"
echo "=============================="
echo ""

# Check if backend can run
CAN_RUN_BACKEND=true
[ ! -d "venv" ] && CAN_RUN_BACKEND=false
[ ! -f "run_server.py" ] && CAN_RUN_BACKEND=false
[ ! -f "config/settings.yaml" ] && CAN_RUN_BACKEND=false

if [ "$CAN_RUN_BACKEND" = true ]; then
    success "✅ Backend is ready to run!"
    info "Start with: source venv/bin/activate && ./run_server.py --mode api"
else
    error "❌ Backend cannot run - fix issues above"
fi

# Check if frontend can run
CAN_RUN_FRONTEND=true
[ ! -d "ui/node_modules" ] && CAN_RUN_FRONTEND=false

if [ "$CAN_RUN_FRONTEND" = true ]; then
    success "✅ Frontend is ready to run!"
    info "Start with: cd ui && npm run dev"
else
    warning "⚠️  Frontend cannot run - install dependencies: cd ui && npm install"
fi

echo ""
echo "=============================="
echo "🚀 Quick Start Commands"
echo "=============================="
echo ""
echo "Terminal 1 (Backend):"
echo "  cd $SCRIPT_DIR"
echo "  source venv/bin/activate"
echo "  ./run_server.py --mode api"
echo ""
echo "Terminal 2 (Frontend):"
echo "  cd $SCRIPT_DIR/ui"
echo "  npm run dev"
echo ""
echo "Then open: http://localhost:3000"
echo ""
echo "=============================="
echo ""

# Exit with appropriate code
if [ "$CAN_RUN_BACKEND" = true ] && [ "$CAN_RUN_FRONTEND" = true ]; then
    success "🎉 System is ready for testing!"
    exit 0
else
    warning "⚠️  System needs setup - see issues above"
    exit 1
fi
