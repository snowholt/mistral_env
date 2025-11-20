#!/bin/bash
#
# RTP Packet Capture Script
# Captures SIP and RTP traffic between Router and PABX server
#

# Configuration
ROUTER_IP="192.168.100.1"
SERVER_IP="192.168.100.39"
OUTPUT_DIR="/home/lumi/beautyai/pabx/logs/captures"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PCAP_FILE="${OUTPUT_DIR}/rtp_capture_${TIMESTAMP}.pcap"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=================================================="
echo "  RTP Traffic Capture - BeautyAI PABX"
echo "=================================================="
echo ""
echo "Capturing traffic between:"
echo "  Router: $ROUTER_IP"
echo "  PABX:   $SERVER_IP"
echo ""
echo "Output file: $PCAP_FILE"
echo ""
echo "Capturing the following:"
echo "  - SIP signaling (port 5060)"
echo "  - RTP audio (ports 10000-20000)"
echo "  - RTCP control (ports 10000-20000)"
echo ""
echo "Press Ctrl+C to stop capture"
echo "=================================================="
echo ""

# Capture filter:
# - SIP traffic on port 5060
# - RTP/RTCP traffic on port range 10000-20000
# - Both directions between Router and Server
sudo tcpdump -i any -w "$PCAP_FILE" \
    "(host $ROUTER_IP and host $SERVER_IP) and \
     ((port 5060) or \
      (portrange 10000-20000))" \
    -v

echo ""
echo "=================================================="
echo "Capture complete!"
echo "File saved: $PCAP_FILE"
echo ""
echo "To analyze with Wireshark on your laptop:"
echo "  1. Copy file to your laptop:"
echo "     scp lumi@192.168.100.39:$PCAP_FILE ."
echo ""
echo "  2. Open in Wireshark"
echo ""
echo "To analyze on server with tshark:"
echo "  tshark -r $PCAP_FILE -Y 'sip || rtp || rtcp'"
echo "=================================================="
