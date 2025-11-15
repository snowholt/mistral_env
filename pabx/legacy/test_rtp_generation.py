#!/usr/bin/env python3
"""
Test RTP packet generation to verify capture is working
Sends fake RTP packets that the capture script should detect
"""

import socket
import struct
import time
import random

def generate_rtp_packet(sequence, timestamp, ssrc, payload_type=0):
    """Generate a fake RTP packet with PCMU audio"""
    # RTP Header
    version = 2
    padding = 0
    extension = 0
    csrc_count = 0
    marker = 0
    
    byte0 = (version << 6) | (padding << 5) | (extension << 4) | csrc_count
    byte1 = (marker << 7) | payload_type
    
    header = struct.pack('!BBHII',
        byte0,           # V, P, X, CC
        byte1,           # M, PT
        sequence,        # Sequence number
        timestamp,       # Timestamp
        ssrc             # SSRC
    )
    
    # Generate fake audio payload (160 bytes of silence for PCMU)
    payload = bytes([0xFF] * 160)
    
    return header + payload

def send_test_rtp(duration=10):
    """Send test RTP packets"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # RTP parameters
    dest_ip = "192.168.100.39"
    dest_port = 15000  # Within our capture range
    ssrc = random.randint(100000, 999999)
    sequence = 0
    timestamp = 0
    
    print(f"🎵 Sending test RTP packets...")
    print(f"   Destination: {dest_ip}:{dest_port}")
    print(f"   SSRC: {ssrc}")
    print(f"   Duration: {duration} seconds")
    print(f"   Start your capture script NOW!")
    print()
    
    time.sleep(2)  # Give time to start capture
    
    start_time = time.time()
    packet_count = 0
    
    while time.time() - start_time < duration:
        packet = generate_rtp_packet(sequence, timestamp, ssrc)
        sock.sendto(packet, (dest_ip, dest_port))
        
        sequence = (sequence + 1) % 65536
        timestamp += 160  # 20ms at 8kHz
        packet_count += 1
        
        if packet_count % 50 == 0:
            print(f"📊 Sent {packet_count} packets...")
        
        time.sleep(0.02)  # 20ms between packets (50 packets/sec)
    
    sock.close()
    print(f"\n✅ Test complete! Sent {packet_count} RTP packets")
    print(f"   Check your capture script for results!")

if __name__ == '__main__':
    print("🧪 RTP Test Packet Generator")
    print("=" * 50)
    print()
    send_test_rtp(duration=10)
