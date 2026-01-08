#!/usr/bin/env python3
"""
Diagnose HT813 network activity and RTP traffic
"""

import sys
import time
from scapy.all import sniff, IP, UDP

def packet_callback(packet):
    """Print information about each packet"""
    if packet.haslayer(IP) and packet.haslayer(UDP):
        ip = packet[IP]
        udp = packet[UDP]
        
        # Check if it involves our HT813
        if ip.src == "192.168.100.96" or ip.dst == "192.168.100.96":
            print(f"📦 {ip.src}:{udp.sport} → {ip.dst}:{udp.dport} | Length: {len(packet)} bytes")
            
            # Check if it might be RTP
            if 10000 <= udp.sport <= 20000 or 10000 <= udp.dport <= 20000:
                print(f"   🎤 Possible RTP packet!")

print("🔍 HT813 Network Activity Monitor")
print("=" * 60)
print("Monitoring ALL traffic to/from 192.168.100.96")
print("This will show if the device is sending ANY packets")
print("Press Ctrl+C to stop")
print()

try:
    sniff(
        iface="enp12s0",
        filter="host 192.168.100.96",
        prn=packet_callback,
        store=False,
        timeout=60
    )
except KeyboardInterrupt:
    print("\n\n⏹️ Stopped")
except Exception as e:
    print(f"\n❌ Error: {e}")
