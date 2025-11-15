#!/bin/bash
# Quick test script for HT813 connectivity and capture

echo "🔍 HT813 Connectivity & Audio Capture Test"
echo "=========================================="
echo

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# HT813 IP
HT813_IP="192.168.100.96"
SERVER_IP="192.168.100.39"

# Test 1: Network Connectivity
echo "Test 1: Network Connectivity"
echo "----------------------------"
if timeout 2 nc -zv $HT813_IP 80 &>/dev/null; then
    echo -e "${GREEN}✓${NC} HT813 is reachable on port 80"
else
    echo -e "${RED}✗${NC} Cannot reach HT813 on port 80"
fi

# Test 2: Web Interface
echo
echo "Test 2: Web Interface Access"
echo "----------------------------"
if timeout 3 curl -s http://$HT813_IP | grep -q "Grandstream"; then
    echo -e "${GREEN}✓${NC} Web interface is accessible"
else
    echo -e "${YELLOW}⚠${NC} Web interface may not be accessible"
fi

# Test 3: Device Status
echo
echo "Test 3: Device Status"
echo "--------------------"
STATUS=$(timeout 3 curl -s http://$HT813_IP 2>/dev/null)
if echo "$STATUS" | grep -q "FXS"; then
    echo -e "${GREEN}✓${NC} Device is responding"
    
    # Try to extract registration status (this is simplified)
    if echo "$STATUS" | grep -q "Registered"; then
        echo -e "${GREEN}✓${NC} At least one port is registered"
    else
        echo -e "${YELLOW}⚠${NC} No ports appear to be registered"
    fi
else
    echo -e "${YELLOW}⚠${NC} Could not retrieve device status"
fi

# Test 4: Network Interface
echo
echo "Test 4: Local Network Configuration"
echo "-----------------------------------"
if ip addr show | grep -q "$SERVER_IP"; then
    echo -e "${GREEN}✓${NC} Server is on the same network (192.168.100.0/24)"
else
    echo -e "${YELLOW}⚠${NC} Server may not be on the same network"
fi

# Test 5: Firewall Rules
echo
echo "Test 5: Firewall Configuration"
echo "------------------------------"
if sudo iptables -t nat -L POSTROUTING -n | grep -q "10.8.0.0/24"; then
    echo -e "${GREEN}✓${NC} VPN NAT rule is configured"
else
    echo -e "${YELLOW}⚠${NC} VPN NAT rule may be missing"
fi

if sudo ufw status | grep -q "192.168.100.0/24.*ALLOW.*10.8.0"; then
    echo -e "${GREEN}✓${NC} UFW routing rules are configured"
else
    echo -e "${YELLOW}⚠${NC} UFW routing rules may be missing"
fi

# Test 6: Python Environment
echo
echo "Test 6: Python Environment"
echo "-------------------------"
if [ -d "venv" ]; then
    echo -e "${GREEN}✓${NC} Virtual environment exists"
    
    if [ -f "venv/bin/python3" ]; then
        echo -e "${GREEN}✓${NC} Python is available in venv"
        
        # Check if scapy is installed
        if venv/bin/python3 -c "import scapy" 2>/dev/null; then
            echo -e "${GREEN}✓${NC} Scapy is installed"
        else
            echo -e "${RED}✗${NC} Scapy is not installed"
        fi
    fi
else
    echo -e "${RED}✗${NC} Virtual environment not found"
fi

# Summary
echo
echo "=========================================="
echo "📋 Summary"
echo "=========================================="
echo
echo "HT813 Device:"
echo "  IP Address: $HT813_IP"
echo "  Web Interface: http://$HT813_IP"
echo "  Default Login: admin/admin"
echo
echo "Server:"
echo "  IP Address: $SERVER_IP"
echo
echo "To start capturing audio:"
echo "  sudo venv/bin/python3 ht813_audio_capture.py -d 60"
echo
echo "To access HT813 from VPN:"
echo "  http://$HT813_IP (use HTTP, not HTTPS)"
echo
echo "For more information:"
echo "  cat README.md"
echo
