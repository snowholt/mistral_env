#!/usr/bin/env python3
"""
HT813 Audio Capture Script
Captures RTP audio streams from the Grandstream HT813 ATA device
"""

import os
import sys
import time
import argparse
import struct
from datetime import datetime
from scapy.all import sniff, IP, UDP, Raw
from scapy.layers.rtp import RTP
import wave
import json

class RTPAudioCapture:
    """Captures and processes RTP audio streams from HT813"""
    
    def __init__(self, device_ip="192.168.100.96", server_ip="192.168.100.39", 
                 rtp_port_range=(10000, 20000), output_dir="captures"):
        self.device_ip = device_ip
        self.server_ip = server_ip
        self.rtp_port_range = rtp_port_range
        self.output_dir = output_dir
        
        # RTP session tracking
        self.rtp_sessions = {}
        self.packet_count = 0
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🎤 HT813 Audio Capture initialized")
        print(f"   Device IP: {device_ip}")
        print(f"   Server IP: {server_ip}")
        print(f"   RTP Port Range: {rtp_port_range[0]}-{rtp_port_range[1]}")
        print(f"   Output Directory: {output_dir}/")
        print()
    
    def _get_session_key(self, src_ip, dst_ip, src_port, dst_port):
        """Generate a unique key for each RTP session"""
        return f"{src_ip}:{src_port}->{dst_ip}:{dst_port}"
    
    def _is_rtp_packet(self, packet):
        """Check if packet is likely an RTP packet"""
        if not packet.haslayer(UDP):
            return False
        
        udp_layer = packet[UDP]
        
        # Check if port is in RTP range
        if not (self.rtp_port_range[0] <= udp_layer.sport <= self.rtp_port_range[1] or
                self.rtp_port_range[0] <= udp_layer.dport <= self.rtp_port_range[1]):
            return False
        
        # Check if packet involves our device
        if packet.haslayer(IP):
            ip_layer = packet[IP]
            if not (ip_layer.src == self.device_ip or ip_layer.dst == self.device_ip or
                    ip_layer.src == self.server_ip or ip_layer.dst == self.server_ip):
                return False
        
        # Check if it has raw payload (RTP data)
        if not packet.haslayer(Raw):
            return False
        
        payload = packet[Raw].load
        
        # RTP header check: Version should be 2 (binary 10)
        if len(payload) < 12:  # Minimum RTP header size
            return False
        
        version = (payload[0] >> 6) & 0x03
        if version != 2:
            return False
        
        return True
    
    def _parse_rtp_header(self, payload):
        """Parse RTP header and return header info and payload data"""
        if len(payload) < 12:
            return None, None
        
        # RTP Header (RFC 3550)
        # 0                   1                   2                   3
        # 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
        # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        # |V=2|P|X|  CC   |M|     PT      |       sequence number         |
        # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        # |                           timestamp                           |
        # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        # |           synchronization source (SSRC) identifier            |
        # +=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
        
        byte0 = payload[0]
        byte1 = payload[1]
        
        version = (byte0 >> 6) & 0x03
        padding = (byte0 >> 5) & 0x01
        extension = (byte0 >> 4) & 0x01
        csrc_count = byte0 & 0x0F
        
        marker = (byte1 >> 7) & 0x01
        payload_type = byte1 & 0x7F
        
        sequence_number = struct.unpack('!H', payload[2:4])[0]
        timestamp = struct.unpack('!I', payload[4:8])[0]
        ssrc = struct.unpack('!I', payload[8:12])[0]
        
        header_length = 12 + (csrc_count * 4)
        
        # Handle extension header if present
        if extension:
            if len(payload) < header_length + 4:
                return None, None
            ext_header_length = struct.unpack('!H', payload[header_length+2:header_length+4])[0]
            header_length += 4 + (ext_header_length * 4)
        
        rtp_header = {
            'version': version,
            'padding': padding,
            'extension': extension,
            'csrc_count': csrc_count,
            'marker': marker,
            'payload_type': payload_type,
            'sequence_number': sequence_number,
            'timestamp': timestamp,
            'ssrc': ssrc
        }
        
        # Extract audio payload
        audio_payload = payload[header_length:]
        
        # Remove padding if present
        if padding and len(audio_payload) > 0:
            padding_length = audio_payload[-1]
            audio_payload = audio_payload[:-padding_length]
        
        return rtp_header, audio_payload
    
    def _process_rtp_packet(self, packet):
        """Process an RTP packet and store audio data"""
        self.packet_count += 1
        
        ip_layer = packet[IP]
        udp_layer = packet[UDP]
        payload = packet[Raw].load
        
        # Parse RTP header
        rtp_header, audio_data = self._parse_rtp_header(payload)
        
        if not rtp_header or not audio_data:
            return
        
        # Get session key
        session_key = self._get_session_key(
            ip_layer.src, ip_layer.dst,
            udp_layer.sport, udp_layer.dport
        )
        
        # Initialize session if new
        if session_key not in self.rtp_sessions:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = os.path.join(self.output_dir, f"session_{timestamp_str}_{rtp_header['ssrc']}")
            os.makedirs(session_dir, exist_ok=True)
            
            self.rtp_sessions[session_key] = {
                'ssrc': rtp_header['ssrc'],
                'payload_type': rtp_header['payload_type'],
                'packets': [],
                'session_dir': session_dir,
                'start_time': time.time(),
                'last_seq': rtp_header['sequence_number'] - 1,
                'lost_packets': 0
            }
            
            print(f"📞 New RTP session detected:")
            print(f"   {session_key}")
            print(f"   SSRC: {rtp_header['ssrc']}")
            print(f"   Payload Type: {rtp_header['payload_type']}")
            print(f"   Output: {session_dir}/")
            print()
        
        session = self.rtp_sessions[session_key]
        
        # Check for lost packets
        expected_seq = (session['last_seq'] + 1) % 65536
        if rtp_header['sequence_number'] != expected_seq:
            lost = (rtp_header['sequence_number'] - expected_seq) % 65536
            session['lost_packets'] += lost
        
        session['last_seq'] = rtp_header['sequence_number']
        
        # Store packet data
        session['packets'].append({
            'sequence': rtp_header['sequence_number'],
            'timestamp': rtp_header['timestamp'],
            'data': audio_data
        })
        
        # Print progress every 100 packets
        if len(session['packets']) % 100 == 0:
            duration = time.time() - session['start_time']
            print(f"📊 Session {rtp_header['ssrc']}: {len(session['packets'])} packets, "
                  f"{session['lost_packets']} lost, {duration:.1f}s")
    
    def packet_handler(self, packet):
        """Handle each captured packet"""
        try:
            if self._is_rtp_packet(packet):
                self._process_rtp_packet(packet)
        except Exception as e:
            print(f"❌ Error processing packet: {e}")
    
    def save_sessions(self):
        """Save all RTP sessions to disk"""
        print("\n💾 Saving captured audio sessions...")
        
        for session_key, session in self.rtp_sessions.items():
            if not session['packets']:
                continue
            
            # Sort packets by sequence number
            session['packets'].sort(key=lambda x: x['sequence'])
            
            # Save raw audio data
            raw_file = os.path.join(session['session_dir'], 'audio_raw.bin')
            with open(raw_file, 'wb') as f:
                for packet in session['packets']:
                    f.write(packet['data'])
            
            # Save session metadata
            metadata = {
                'session_key': session_key,
                'ssrc': session['ssrc'],
                'payload_type': session['payload_type'],
                'total_packets': len(session['packets']),
                'lost_packets': session['lost_packets'],
                'duration_seconds': time.time() - session['start_time'],
                'codec': self._get_codec_name(session['payload_type'])
            }
            
            metadata_file = os.path.join(session['session_dir'], 'metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Saved session {session['ssrc']}:")
            print(f"   Raw audio: {raw_file}")
            print(f"   Metadata: {metadata_file}")
            print(f"   Total packets: {len(session['packets'])}")
            print(f"   Lost packets: {session['lost_packets']}")
            print(f"   Codec: {metadata['codec']}")
            print()
    
    def _get_codec_name(self, payload_type):
        """Get codec name from RTP payload type"""
        codecs = {
            0: 'PCMU (G.711 μ-law)',
            3: 'GSM',
            4: 'G.723',
            8: 'PCMA (G.711 A-law)',
            9: 'G.722',
            18: 'G.729'
        }
        return codecs.get(payload_type, f'Unknown ({payload_type})')
    
    def start_capture(self, interface=None, duration=None):
        """Start capturing RTP packets"""
        print("🎯 Starting RTP packet capture...")
        print("   Press Ctrl+C to stop\n")
        
        # Build capture filter
        filter_str = f"udp and (host {self.device_ip} or host {self.server_ip})"
        
        try:
            if duration:
                print(f"⏱️  Capturing for {duration} seconds...")
                sniff(iface=interface, filter=filter_str, prn=self.packet_handler, 
                      timeout=duration, store=False)
            else:
                sniff(iface=interface, filter=filter_str, prn=self.packet_handler, 
                      store=False)
        except KeyboardInterrupt:
            print("\n\n⏹️  Capture stopped by user")
        except Exception as e:
            print(f"\n❌ Error during capture: {e}")
        finally:
            self.save_sessions()
            print(f"\n📈 Total packets processed: {self.packet_count}")
            print(f"📁 Output directory: {self.output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description='Capture RTP audio streams from Grandstream HT813',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Capture audio with default settings
  sudo python3 ht813_audio_capture.py
  
  # Capture from specific interface for 60 seconds
  sudo python3 ht813_audio_capture.py -i enp12s0 -d 60
  
  # Capture with custom IP addresses
  sudo python3 ht813_audio_capture.py --device 192.168.100.96 --server 192.168.100.39
  
  # Custom RTP port range
  sudo python3 ht813_audio_capture.py --rtp-start 16384 --rtp-end 32767

Note: This script requires root/sudo privileges to capture network packets.
        """
    )
    
    parser.add_argument('--device', '-D', default='192.168.100.96',
                        help='HT813 device IP address (default: 192.168.100.96)')
    parser.add_argument('--server', '-S', default='192.168.100.39',
                        help='Server IP address (default: 192.168.100.39)')
    parser.add_argument('--interface', '-i', default=None,
                        help='Network interface to capture on (default: auto)')
    parser.add_argument('--duration', '-d', type=int, default=None,
                        help='Capture duration in seconds (default: unlimited)')
    parser.add_argument('--rtp-start', type=int, default=10000,
                        help='RTP port range start (default: 10000)')
    parser.add_argument('--rtp-end', type=int, default=20000,
                        help='RTP port range end (default: 20000)')
    parser.add_argument('--output', '-o', default='captures',
                        help='Output directory for captured audio (default: captures/)')
    
    args = parser.parse_args()
    
    # Check if running as root
    if os.geteuid() != 0:
        print("❌ Error: This script requires root privileges to capture packets")
        print("   Please run with sudo:")
        print(f"   sudo python3 {sys.argv[0]}")
        sys.exit(1)
    
    # Create capture instance
    capture = RTPAudioCapture(
        device_ip=args.device,
        server_ip=args.server,
        rtp_port_range=(args.rtp_start, args.rtp_end),
        output_dir=args.output
    )
    
    # Start capture
    capture.start_capture(interface=args.interface, duration=args.duration)


if __name__ == '__main__':
    main()
