#!/bin/bash
# Diagnose call disconnection issue

echo "========================================"
echo "HT813 Call Disconnection Diagnostic"
echo "========================================"
echo ""

echo "1. Checking if SIP server is running..."
ps aux | grep simple_sip_server | grep -v grep
if [ $? -eq 0 ]; then
    echo "   ✅ SIP server is running"
else
    echo "   ❌ SIP server is NOT running!"
fi
echo ""

echo "2. Checking port 5060..."
sudo ss -ulnp | grep 5060
if [ $? -eq 0 ]; then
    echo "   ✅ Port 5060 is in use"
else
    echo "   ❌ Port 5060 is not listening!"
fi
echo ""

echo "3. Checking recent SIP traffic..."
echo "   Capturing for 10 seconds (make a call now!)..."
timeout 10 sudo tcpdump -i enp12s0 -c 20 "udp port 5060" 2>&1 | grep -E "INVITE|BYE|ACK|200|180|100" || echo "   No SIP traffic detected"
echo ""

echo "4. Checking if HT813 is reachable..."
ping -c 2 192.168.100.96 > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✅ HT813 (192.168.100.96) is reachable"
else
    echo "   ❌ Cannot reach HT813!"
fi
echo ""

echo "5. Testing HTTP access to HT813..."
curl -s -m 3 http://192.168.100.96 > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✅ HT813 web interface is accessible"
else
    echo "   ⚠️  Cannot access HT813 web interface"
fi
echo ""

echo "========================================"
echo "Diagnostic complete!"
echo "========================================"
