#!/bin/bash
#
# RTP Analysis Script
# Quick analysis of captured RTP traffic
#

if [ -z "$1" ]; then
    echo "Usage: $0 <pcap_file>"
    echo ""
    echo "Available captures:"
    ls -lh /home/lumi/beautyai/pabx/logs/captures/*.pcap 2>/dev/null | tail -5
    exit 1
fi

PCAP_FILE="$1"

if [ ! -f "$PCAP_FILE" ]; then
    echo "Error: File not found: $PCAP_FILE"
    exit 1
fi

echo "=================================================="
echo "  RTP Traffic Analysis"
echo "=================================================="
echo "File: $PCAP_FILE"
echo ""

echo "--- SIP Messages ---"
tshark -r "$PCAP_FILE" -Y 'sip' -T fields \
    -e frame.time \
    -e sip.Method \
    -e sip.Status-Code \
    -e sip.Status-Line \
    -e sip.From \
    -e sip.To \
    2>/dev/null | head -20

echo ""
echo "--- SDP Offers (c= connection lines) ---"
tshark -r "$PCAP_FILE" -Y 'sdp' -V 2>/dev/null | grep -E "(c=IN IP4|m=audio|a=rtpmap)" | head -30

echo ""
echo "--- RTP Stream Statistics ---"
tshark -r "$PCAP_FILE" -z rtp,streams 2>/dev/null

echo ""
echo "--- RTP Packet Count ---"
echo "Total RTP packets from HT813 -> Server:"
tshark -r "$PCAP_FILE" -Y 'rtp and ip.src==192.168.100.96' 2>/dev/null | wc -l

echo "Total RTP packets from Server -> HT813:"
tshark -r "$PCAP_FILE" -Y 'rtp and ip.src==192.168.100.39' 2>/dev/null | wc -l

echo ""
echo "--- Sample RTP Packets ---"
echo "From HT813:"
tshark -r "$PCAP_FILE" -Y 'rtp and ip.src==192.168.100.96' -T fields \
    -e frame.time \
    -e ip.src \
    -e udp.srcport \
    -e ip.dst \
    -e udp.dstport \
    -e rtp.ssrc \
    2>/dev/null | head -5

echo ""
echo "From Server:"
tshark -r "$PCAP_FILE" -Y 'rtp and ip.src==192.168.100.39' -T fields \
    -e frame.time \
    -e ip.src \
    -e udp.srcport \
    -e ip.dst \
    -e udp.dstport \
    -e rtp.ssrc \
    2>/dev/null | head -5

echo ""
echo "=================================================="
echo "Analysis complete!"
echo ""
echo "To open in Wireshark GUI on your laptop:"
echo "  scp lumi@192.168.100.39:$PCAP_FILE ."
echo "=================================================="
