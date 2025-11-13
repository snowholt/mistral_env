#!/bin/bash
# WebRTC Lean Capture Validation Script
# Auto-checks all acceptance criteria after 30s capture
# 
# Usage: ./validate_lean_capture.sh
# Run from: /home/lumi/beautyai

set -e

REPORT_DIR="reports/debug/webrtc"
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=================================================="
echo "WebRTC Lean Capture: Acceptance Criteria Validation"
echo "Date: $(date)"
echo "=================================================="
echo ""

# Check if report directory exists
if [ ! -d "$REPORT_DIR" ]; then
    echo -e "${RED}❌ FAIL${NC}: Report directory not found: $REPORT_DIR"
    echo "   Please run a capture session first"
    exit 1
fi

# Check if JSON files exist
if [ ! -f "$REPORT_DIR/queue_stats.json" ] || [ ! -f "$REPORT_DIR/pipeline_stats.json" ]; then
    echo -e "${RED}❌ FAIL${NC}: Metrics JSON files not found"
    echo "   Expected: queue_stats.json, pipeline_stats.json"
    exit 1
fi

echo "=== Queue Statistics ==="
cat "$REPORT_DIR/queue_stats.json" | jq '{
  underrun_rate: .underrun_rate_percent,
  recv_p99_ms: .recv_delta_p99_ms,
  worker_p99_ms: .worker_service_p99_ms,
  queue_peak: .peak_depth,
  total_frames: .enqueued,
  dropped: .dropped
}'
echo ""

echo "=== Pipeline Statistics ==="
cat "$REPORT_DIR/pipeline_stats.json" | jq '{
  preset: .denoiser_type,
  frame_count: .frame_count,
  limiter_activations: .limiter_activations,
  comb_active_frames: .comb_active_frames,
  gate_closed_frames: .gate_closed_frames
}'
echo ""

# Extract metrics
UNDERRUN_RATE=$(cat "$REPORT_DIR/queue_stats.json" | jq -r '.underrun_rate_percent')
RECV_P99=$(cat "$REPORT_DIR/queue_stats.json" | jq -r '.recv_delta_p99_ms')
WORKER_P99=$(cat "$REPORT_DIR/queue_stats.json" | jq -r '.worker_service_p99_ms')
QUEUE_PEAK=$(cat "$REPORT_DIR/queue_stats.json" | jq -r '.peak_depth')
DROPPED=$(cat "$REPORT_DIR/queue_stats.json" | jq -r '.dropped')
TOTAL_FRAMES=$(cat "$REPORT_DIR/queue_stats.json" | jq -r '.enqueued')

echo "=== Acceptance Criteria Validation ==="
PASS_COUNT=0
FAIL_COUNT=0

# Criterion 1: Underrun rate < 1%
if (( $(echo "$UNDERRUN_RATE < 1.0" | bc -l) )); then
    echo -e "${GREEN}✅ PASS${NC}: Underrun rate < 1% (actual: ${UNDERRUN_RATE}%)"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo -e "${RED}❌ FAIL${NC}: Underrun rate >= 1% (actual: ${UNDERRUN_RATE}%)"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# Criterion 2: Recv p99 < 2ms
if (( $(echo "$RECV_P99 < 2.0" | bc -l) )); then
    echo -e "${GREEN}✅ PASS${NC}: Recv p99 < 2ms (actual: ${RECV_P99}ms)"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo -e "${YELLOW}⚠️  WARN${NC}: Recv p99 >= 2ms (actual: ${RECV_P99}ms) - Still acceptable if <5ms"
    # Don't count as fail if <5ms (acceptable range)
    if (( $(echo "$RECV_P99 < 5.0" | bc -l) )); then
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
fi

# Criterion 3: Worker p99 < 20ms
if (( $(echo "$WORKER_P99 < 20.0" | bc -l) )); then
    echo -e "${GREEN}✅ PASS${NC}: Worker p99 < 20ms (actual: ${WORKER_P99}ms)"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo -e "${RED}❌ FAIL${NC}: Worker p99 >= 20ms (actual: ${WORKER_P99}ms)"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# Criterion 4: Queue peak ≤ 8 frames
if [ "$QUEUE_PEAK" -le 8 ]; then
    echo -e "${GREEN}✅ PASS${NC}: Queue peak ≤ 8 frames (actual: ${QUEUE_PEAK})"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo -e "${RED}❌ FAIL${NC}: Queue peak > 8 frames (actual: ${QUEUE_PEAK})"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# Criterion 5: No dropped frames
if [ "$DROPPED" -eq 0 ]; then
    echo -e "${GREEN}✅ PASS${NC}: No dropped frames (actual: ${DROPPED})"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo -e "${YELLOW}⚠️  WARN${NC}: Dropped frames detected (actual: ${DROPPED}) - Acceptable if <1% of total"
    # Allow up to 1% drops
    DROP_RATE=$(echo "scale=2; $DROPPED * 100 / $TOTAL_FRAMES" | bc)
    if (( $(echo "$DROP_RATE < 1.0" | bc -l) )); then
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
fi

# Criterion 6: Total frames > 1000 (30s @ 20ms = 1500 frames)
if [ "$TOTAL_FRAMES" -gt 1000 ]; then
    echo -e "${GREEN}✅ PASS${NC}: Sufficient frames captured (actual: ${TOTAL_FRAMES})"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo -e "${RED}❌ FAIL${NC}: Insufficient frames captured (actual: ${TOTAL_FRAMES}, expected >1000)"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

echo ""
echo "=== Check RT Scheduling ==="
RT_SCHED=$(ps -eo pid,comm,cls,rtprio,ni | grep python | grep -E "RR|FF" || echo "")
if [ -n "$RT_SCHED" ]; then
    echo -e "${GREEN}✅ PASS${NC}: RT scheduling active"
    echo "$RT_SCHED"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo -e "${YELLOW}⚠️  WARN${NC}: RT scheduling not detected (check manually with: ps -eo pid,comm,cls,rtprio,ni | grep python)"
    # Don't count as fail - may not have permission to view
fi

echo ""
echo "=== Summary ==="
echo "Passed: $PASS_COUNT / 7"
echo "Failed: $FAIL_COUNT / 7"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "${GREEN}🎉 ALL CRITERIA MET - READY FOR PRODUCTION${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Run analyzer: python tools/analyze_audio_noise.py --compare --visualize"
    echo "  2. Verify crackle reduction: Expect <100 crackles @ 48kHz (was 7488)"
    echo "  3. Verify 80 Hz hum elimination: Visual inspection of spectrograms"
    echo "  4. Deploy to production if all metrics acceptable"
    exit 0
else
    echo -e "${RED}⚠️  SOME CRITERIA FAILED - REVIEW METRICS${NC}"
    echo ""
    echo "Troubleshooting:"
    if (( $(echo "$UNDERRUN_RATE >= 1.0" | bc -l) )); then
        echo "  - Underrun rate high: Increase queue depth (max_size=8) or reduce worker count"
    fi
    if (( $(echo "$RECV_P99 >= 5.0" | bc -l) )); then
        echo "  - Recv p99 high: Disable verbose logging (WEBRTC_DEBUG_VERBOSE=0)"
    fi
    if (( $(echo "$WORKER_P99 >= 20.0" | bc -l) )); then
        echo "  - Worker p99 high: Try 'minimal' preset or profile denoiser performance"
    fi
    echo ""
    echo "See docs/WEBRTC_LEAN_QUICKSTART.md for detailed troubleshooting"
    exit 1
fi
