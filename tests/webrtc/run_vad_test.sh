#!/bin/bash
# run_vad_test.sh - Test runner for WebRTC + VAD testing
# 
# This script:
# 1. Starts the VAD test server with mock models
# 2. Runs the WebRTC audio test with laser_hair.wav
# 3. Collects VAD metrics and logs
# 4. Generates test report
#
# Usage:
#   ./tests/webrtc/run_vad_test.sh
#   ./tests/webrtc/run_vad_test.sh --scenario silero_only
#
# Author: BeautyAI Framework
# Date: October 29, 2025

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BACKEND_DIR="$REPO_ROOT/backend"
CONFIG_FILE="$REPO_ROOT/config/config.vad_test.yaml"
LOG_DIR="$REPO_ROOT/logs"
VAD_DEBUG_DIR="$LOG_DIR/webrtc/vad_debug"
METRICS_FILE="$LOG_DIR/vad_test_metrics.json"
SERVER_LOG="$LOG_DIR/vad_test_server.log"
TEST_LOG="$LOG_DIR/vad_test_run.log"
SERVER_PID_FILE="/tmp/vad_test_server.pid"

# Test scenarios
SCENARIO="${1:-dual_vad}"  # dual_vad, silero_only, webrtc_only

echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}   BeautyAI WebRTC + VAD Test Runner${NC}"
echo -e "${BLUE}=================================================${NC}"
echo -e "Scenario: ${GREEN}$SCENARIO${NC}"
echo -e "Config: $CONFIG_FILE"
echo -e "Log directory: $LOG_DIR"
echo ""

# Create log directories
mkdir -p "$LOG_DIR"
mkdir -p "$VAD_DEBUG_DIR"
mkdir -p "$REPO_ROOT/reports/webRTC-VAD"

# Clean up previous logs
rm -f "$SERVER_LOG" "$TEST_LOG" "$METRICS_FILE"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Cleaning up...${NC}"
    
    # Kill server if running
    if [ -f "$SERVER_PID_FILE" ]; then
        SERVER_PID=$(cat "$SERVER_PID_FILE")
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Stopping VAD test server (PID: $SERVER_PID)"
            kill "$SERVER_PID" 2>/dev/null || true
            sleep 2
            # Force kill if still running
            kill -9 "$SERVER_PID" 2>/dev/null || true
        fi
        rm -f "$SERVER_PID_FILE"
    fi
    
    echo -e "${GREEN}Cleanup complete${NC}"
}

trap cleanup EXIT INT TERM

# Check prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"

if [ ! -f "$REPO_ROOT/tests/webrtc/laser_hair.wav" ]; then
    echo -e "${RED}ERROR: Test audio file not found: tests/webrtc/laser_hair.wav${NC}"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}ERROR: Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Check Python environment
cd "$BACKEND_DIR"
if [ ! -d "../.venv" ]; then
    echo -e "${RED}ERROR: Virtual environment not found at .venv${NC}"
    echo "Please run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source "$REPO_ROOT/.venv/bin/activate"

echo -e "${GREEN}✓ Prerequisites OK${NC}"
echo ""

# Set environment variables for VAD test mode
export VAD_TEST_MODE=1
export BEAUTYAI_VAD_DEBUG=1
export PYTHONPATH="$BACKEND_DIR/src:$PYTHONPATH"

# Set scenario-specific config
case "$SCENARIO" in
    "dual_vad")
        export VAD_DUAL_MODE=true
        export VAD_REQUIRE_CONFIRMATION=true
        echo -e "Mode: ${GREEN}Dual VAD${NC} (WebRTC + Silero confirmation)"
        ;;
    "silero_only")
        export VAD_DUAL_MODE=false
        export VAD_REQUIRE_CONFIRMATION=false
        echo -e "Mode: ${GREEN}Silero Only${NC} (ML-based VAD)"
        ;;
    "webrtc_only")
        export VAD_DUAL_MODE=true
        export VAD_REQUIRE_CONFIRMATION=false
        echo -e "Mode: ${GREEN}WebRTC Only${NC} (Rule-based VAD)"
        ;;
    *)
        echo -e "${RED}ERROR: Unknown scenario: $SCENARIO${NC}"
        echo "Valid scenarios: dual_vad, silero_only, webrtc_only"
        exit 1
        ;;
esac

echo ""

# Start VAD test server
echo -e "${BLUE}Starting VAD test server...${NC}"
python "$BACKEND_DIR/run_vad_test_server.py" \
    --config "$CONFIG_FILE" \
    --host 0.0.0.0 \
    --port 8000 \
    > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
echo $SERVER_PID > "$SERVER_PID_FILE"

echo -e "Server PID: ${GREEN}$SERVER_PID${NC}"
echo "Server log: $SERVER_LOG"

# Wait for server to be ready
echo -n "Waiting for server to start..."
MAX_WAIT=30
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e " ${GREEN}OK${NC}"
        break
    fi
    echo -n "."
    sleep 1
    WAIT_COUNT=$((WAIT_COUNT + 1))
done

if [ $WAIT_COUNT -eq $MAX_WAIT ]; then
    echo -e " ${RED}TIMEOUT${NC}"
    echo -e "${RED}ERROR: Server failed to start within ${MAX_WAIT}s${NC}"
    echo "Check server log: $SERVER_LOG"
    tail -n 50 "$SERVER_LOG"
    exit 1
fi

# Check server health
echo -n "Checking server health..."
HEALTH_RESPONSE=$(curl -s http://localhost:8000/health)
if echo "$HEALTH_RESPONSE" | grep -q '"status":"healthy"'; then
    echo -e " ${GREEN}OK${NC}"
    echo "VAD test mode: $(echo "$HEALTH_RESPONSE" | grep -o '"vad_test_mode":"[^"]*"' | cut -d'"' -f4)"
else
    echo -e " ${RED}FAILED${NC}"
    echo "Response: $HEALTH_RESPONSE"
    exit 1
fi

echo ""

# Run WebRTC audio test
echo -e "${BLUE}Running WebRTC audio test with VAD...${NC}"
echo "Test audio: tests/webrtc/laser_hair.wav"
echo "Test scenario: $SCENARIO"
echo ""

cd "$REPO_ROOT"

# Run pytest with verbose output
export WEBRTC_TEST_BASE_URL="http://localhost:8000/api/v1/webrtc/voice"

pytest tests/webrtc/webrtc_audio_test.py \
    -v \
    -s \
    --tb=short \
    --log-cli-level=INFO \
    2>&1 | tee "$TEST_LOG"

TEST_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}   Test Results${NC}"
echo -e "${BLUE}=================================================${NC}"

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "Test status: ${GREEN}PASSED ✓${NC}"
else
    echo -e "Test status: ${RED}FAILED ✗${NC}"
fi

echo ""
echo -e "${BLUE}Generated artifacts:${NC}"

# Show VAD debug audio files
if [ -d "$VAD_DEBUG_DIR" ]; then
    VAD_FILES=$(find "$VAD_DEBUG_DIR" -name "*.wav" -type f 2>/dev/null | wc -l)
    if [ $VAD_FILES -gt 0 ]; then
        echo -e "  ${GREEN}✓${NC} VAD debug audio: $VAD_FILES file(s) in $VAD_DEBUG_DIR"
        ls -lh "$VAD_DEBUG_DIR"/*.wav 2>/dev/null | tail -5
    else
        echo -e "  ${YELLOW}⚠${NC} No VAD debug audio files generated"
    fi
fi

# Show captured audio
CAPTURED_DIR="$REPO_ROOT/reports/webRTC-VAD"
if [ -d "$CAPTURED_DIR" ]; then
    CAPTURED_FILES=$(find "$CAPTURED_DIR" -name "*.wav" -type f 2>/dev/null | wc -l)
    if [ $CAPTURED_FILES -gt 0 ]; then
        echo -e "  ${GREEN}✓${NC} Captured audio: $CAPTURED_FILES file(s) in $CAPTURED_DIR"
        ls -lh "$CAPTURED_DIR"/*.wav 2>/dev/null | tail -3
    fi
fi

# Show metrics file
if [ -f "$METRICS_FILE" ]; then
    echo -e "  ${GREEN}✓${NC} Metrics: $METRICS_FILE"
else
    echo -e "  ${YELLOW}⚠${NC} No metrics file generated"
fi

echo ""
echo -e "${BLUE}Logs:${NC}"
echo -e "  Server log: $SERVER_LOG (tail -f to follow)"
echo -e "  Test log: $TEST_LOG"

# Extract VAD statistics from server log
echo ""
echo -e "${BLUE}VAD Statistics (from server log):${NC}"
if [ -f "$SERVER_LOG" ]; then
    echo -n "  WebRTC detections: "
    grep -c "WEBRTC-VAD.*detected=True" "$SERVER_LOG" 2>/dev/null || echo "0"
    
    echo -n "  Silero detections: "
    grep -c "SILERO-VAD.*detected=True" "$SERVER_LOG" 2>/dev/null || echo "0"
    
    echo -n "  VAD state transitions: "
    grep -c "VAD.*state.*->" "$SERVER_LOG" 2>/dev/null || echo "0"
    
    echo -n "  Speech segments: "
    grep -c "VOICE_ACTIVE" "$SERVER_LOG" 2>/dev/null || echo "0"
fi

echo ""
echo -e "${BLUE}=================================================${NC}"

# Generate simple test report
REPORT_FILE="$REPO_ROOT/reports/webRTC-VAD/vad_test_report_$(date +%Y%m%d_%H%M%S).md"
cat > "$REPORT_FILE" << EOF
# WebRTC + VAD Test Report

**Date**: $(date '+%Y-%m-%d %H:%M:%S')
**Scenario**: $SCENARIO
**Status**: $([ $TEST_EXIT_CODE -eq 0 ] && echo "✓ PASSED" || echo "✗ FAILED")

## Configuration

- Test Mode: VAD Test Mode (Mock Models)
- Audio File: tests/webrtc/laser_hair.wav
- VAD Strategy: $SCENARIO
- Server: http://localhost:8000

## Results

$(if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✓ Test completed successfully"
else
    echo "✗ Test failed - check logs for details"
fi)

### Artifacts

- Server Log: \`$SERVER_LOG\`
- Test Log: \`$TEST_LOG\`
- VAD Debug Audio: \`$VAD_DEBUG_DIR\`
- Captured Audio: \`$CAPTURED_DIR\`

### VAD Statistics

$(if [ -f "$SERVER_LOG" ]; then
    echo "- WebRTC Detections: $(grep -c "WEBRTC-VAD.*detected=True" "$SERVER_LOG" 2>/dev/null || echo "0")"
    echo "- Silero Detections: $(grep -c "SILERO-VAD.*detected=True" "$SERVER_LOG" 2>/dev/null || echo "0")"
    echo "- State Transitions: $(grep -c "VAD.*state.*->" "$SERVER_LOG" 2>/dev/null || echo "0")"
    echo "- Speech Segments: $(grep -c "VOICE_ACTIVE" "$SERVER_LOG" 2>/dev/null || echo "0")"
fi)

## Logs

### Server Log (last 50 lines)

\`\`\`
$(tail -50 "$SERVER_LOG" 2>/dev/null)
\`\`\`

### Test Output

\`\`\`
$(tail -100 "$TEST_LOG" 2>/dev/null)
\`\`\`

---
Generated by: tests/webrtc/run_vad_test.sh
EOF

echo -e "Test report: ${GREEN}$REPORT_FILE${NC}"
echo ""

# Exit with test status
exit $TEST_EXIT_CODE
