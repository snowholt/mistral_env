#!/bin/bash
#
# Monitor incoming SIP calls from router
# Watch for INVITE messages in real-time
#

echo "=================================================="
echo "  PABX Call Monitor"
echo "=================================================="
echo ""
echo "Monitoring for incoming calls from router..."
echo "Router IP: 192.168.100.1"
echo "PABX IP: 192.168.100.39:5060"
echo ""
echo "Watching for:"
echo "  - INVITE messages (incoming calls)"
echo "  - 200 OK responses (call answered)"
echo "  - RTP session info"
echo "  - Recording status"
echo ""
echo "Press Ctrl+C to stop"
echo "=================================================="
echo ""

# Monitor journal logs in real-time
sudo journalctl -u pabx-backend.service -f --no-pager | \
    grep --line-buffered -E "(INVITE|200 OK|RTP|Recording|greeting|Auto-answer|from 192.168.100.1)" | \
    while IFS= read -r line; do
        # Highlight important events
        if echo "$line" | grep -q "INVITE"; then
            echo -e "\n🔔 \033[1;32mINCOMING CALL\033[0m"
            echo "$line"
        elif echo "$line" | grep -q "200 OK"; then
            echo -e "✅ \033[1;34mCALL ANSWERED\033[0m"
            echo "$line"
        elif echo "$line" | grep -q "RTP"; then
            echo -e "🔊 \033[1;33mAUDIO STREAM\033[0m"
            echo "$line"
        elif echo "$line" | grep -q "Recording"; then
            echo -e "⏺️  \033[1;35mRECORDING\033[0m"
            echo "$line"
        elif echo "$line" | grep -q "greeting"; then
            echo -e "🎵 \033[1;36mGREETING PLAYING\033[0m"
            echo "$line"
        else
            echo "$line"
        fi
    done
