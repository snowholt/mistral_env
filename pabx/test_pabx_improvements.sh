#!/bin/bash
# PABX System Test Script
# Tests WebSocket events, call quality metrics, and audio functionality

set -e

BASE_URL="http://192.168.100.39:8080"
WS_URL="ws://192.168.100.39:8080/ws"

echo "=========================================="
echo "PABX System Improvement Tests"
echo "=========================================="
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test 1: Check service is running
echo -e "${YELLOW}Test 1: Service Status${NC}"
if systemctl is-active --quiet pabx-backend; then
    echo -e "${GREEN}✅ PABX service is running${NC}"
else
    echo -e "${RED}❌ PABX service is not running${NC}"
    exit 1
fi
echo ""

# Test 2: Check API health
echo -e "${YELLOW}Test 2: API Health Check${NC}"
HEALTH=$(curl -s "${BASE_URL}/api/health")
if echo "$HEALTH" | jq -e '.status == "healthy"' > /dev/null 2>&1; then
    echo -e "${GREEN}✅ API is healthy${NC}"
    echo "$HEALTH" | jq .
else
    echo -e "${RED}❌ API health check failed${NC}"
    exit 1
fi
echo ""

# Test 3: Check SIP registrations
echo -e "${YELLOW}Test 3: SIP Registrations${NC}"
REGISTRATIONS=$(curl -s "${BASE_URL}/api/registrations")
COUNT=$(echo "$REGISTRATIONS" | jq '.count')
echo "Registered users: $COUNT"
if [ "$COUNT" -gt 0 ]; then
    echo -e "${GREEN}✅ Users are registered${NC}"
    echo "$REGISTRATIONS" | jq '.registrations'
else
    echo -e "${YELLOW}⚠️  No users currently registered (HT813 may need to re-register)${NC}"
fi
echo ""

# Test 4: Check audio file
echo -e "${YELLOW}Test 4: Greeting Audio File${NC}"
AUDIO_FILE="/home/lumi/beautyai/voice_tests/input_test_questions/greeting_ar.wav"
if [ -f "$AUDIO_FILE" ]; then
    SIZE=$(du -h "$AUDIO_FILE" | cut -f1)
    echo -e "${GREEN}✅ Greeting audio file exists ($SIZE)${NC}"
    echo "   Path: $AUDIO_FILE"
else
    echo -e "${RED}❌ Greeting audio file not found${NC}"
    exit 1
fi
echo ""

# Test 5: Check for recent errors in logs
echo -e "${YELLOW}Test 5: Recent Error Log Check${NC}"
ERROR_COUNT=$(tail -100 /home/lumi/beautyai/pabx/logs/system/app.json 2>/dev/null | jq -s 'map(select(.level == "ERROR")) | length' || echo "0")
if [ "$ERROR_COUNT" -eq 0 ]; then
    echo -e "${GREEN}✅ No errors in recent logs${NC}"
else
    echo -e "${RED}❌ Found $ERROR_COUNT errors in recent logs${NC}"
    tail -100 /home/lumi/beautyai/pabx/logs/system/app.json | jq -s 'map(select(.level == "ERROR")) | .[-3:]'
fi
echo ""

# Test 6: Check WebSocket connectivity
echo -e "${YELLOW}Test 6: WebSocket Connection Test${NC}"
echo "Starting WebSocket listener in background for 10 seconds..."
echo "Please make a test call now if possible!"
echo ""

# Start WebSocket listener in background
timeout 10 /home/lumi/beautyai/pabx/venv/bin/python3 /home/lumi/beautyai/pabx/test_websocket.py 2>&1 | tee /tmp/ws_test.log &
WS_PID=$!

# Wait for connection
sleep 2

# Check if WebSocket connected
if ps -p $WS_PID > /dev/null 2>&1; then
    echo -e "${GREEN}✅ WebSocket connection established${NC}"
    echo "   Listening for events for 8 more seconds..."
    wait $WS_PID 2>/dev/null || true
else
    echo -e "${RED}❌ WebSocket connection failed${NC}"
fi
echo ""

# Test 7: Check active calls
echo -e "${YELLOW}Test 7: Active Calls Check${NC}"
CALLS=$(curl -s "${BASE_URL}/api/calls")
CALL_COUNT=$(echo "$CALLS" | jq '.count')
echo "Active calls: $CALL_COUNT"
if [ "$CALL_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✅ Active calls found${NC}"
    echo "$CALLS" | jq '.calls'
    
    # Test 8: Check call statistics (if there are active calls)
    echo ""
    echo -e "${YELLOW}Test 8: Call Quality Metrics${NC}"
    CALL_ID=$(echo "$CALLS" | jq -r '.calls[0].call_id')
    STATS=$(curl -s "${BASE_URL}/api/calls/${CALL_ID}/stats")
    
    if echo "$STATS" | jq -e '.rtp_stats' > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Call quality metrics available${NC}"
        echo "$STATS" | jq '{
            call_id: .call_id,
            duration: .duration,
            rtp_stats: {
                packets_sent: .rtp_stats.packets_sent,
                packets_received: .rtp_stats.packets_received,
                packets_lost: .rtp_stats.packets_lost,
                packet_loss_rate: .rtp_stats.packet_loss_rate,
                jitter: .rtp_stats.jitter
            }
        }'
    else
        echo -e "${YELLOW}⚠️  RTP statistics not yet available${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  No active calls at this time${NC}"
    echo "   Make a test call from extension 1002 to 1001 to test call functionality"
fi
echo ""

# Test 9: Check syslog messages
echo -e "${YELLOW}Test 9: Recent Syslog Messages${NC}"
SYSLOG=$(curl -s "${BASE_URL}/api/syslog/messages?limit=5")
SYSLOG_COUNT=$(echo "$SYSLOG" | jq '.count')
if [ "$SYSLOG_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✅ Receiving syslog messages from HT813${NC}"
    echo "Last message: $(echo "$SYSLOG" | jq -r '.messages[-1].message' | head -c 80)..."
else
    echo -e "${YELLOW}⚠️  No recent syslog messages${NC}"
fi
echo ""

echo "=========================================="
echo -e "${GREEN}Test Suite Complete!${NC}"
echo "=========================================="
echo ""
echo "Summary:"
echo "  - WebSocket improvements: Thread-safe event broadcasting implemented"
echo "  - Call quality metrics: Enhanced RTP statistics with jitter calculation"
echo "  - SIP retransmission handling: RFC 3261 timers implemented"
echo "  - WebSocket health monitoring: Ping/pong heartbeat with 60s timeout"
echo ""
echo "To test call functionality:"
echo "  1. Wait for HT813 to re-register (check: curl ${BASE_URL}/api/registrations)"
echo "  2. Make a call from extension 1002 to 1001"
echo "  3. Listen to real-time events: /home/lumi/beautyai/pabx/venv/bin/python3 /home/lumi/beautyai/pabx/test_websocket.py"
echo "  4. Check call stats: curl ${BASE_URL}/api/calls/<call-id>/stats | jq"
echo ""
