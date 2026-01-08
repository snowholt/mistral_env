"""
Packet capture engine using scapy
Captures SIP and RTP packets from network
"""

import threading
import time
from dataclasses import dataclass
from typing import Optional, Callable, Set
from datetime import datetime

try:
    from scapy.all import sniff, UDP, IP, Raw, wrpcap
    from scapy.packet import Packet as ScapyPacket
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    ScapyPacket = None

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CaptureFilter:
    """Packet capture filter configuration"""
    target_ip: Optional[str] = None
    capture_sip: bool = True
    capture_rtp: bool = True
    capture_rtcp: bool = True
    sip_ports: Set[int] = None
    rtp_port_range: tuple = (10000, 20000)
    interface: Optional[str] = None
    
    def __post_init__(self):
        if self.sip_ports is None:
            self.sip_ports = {5060, 5061, 5062, 5012}
    
    def to_bpf_filter(self) -> str:
        """
        Convert filter to BPF (Berkeley Packet Filter) syntax
        
        Returns:
            BPF filter string
        """
        filters = []
        
        # UDP only
        filters.append("udp")
        
        # Target IP filter
        if self.target_ip:
            filters.append(f"host {self.target_ip}")
        
        # Port filters
        port_filters = []
        
        if self.capture_sip:
            sip_port_list = ' or '.join(f'port {p}' for p in self.sip_ports)
            port_filters.append(f"({sip_port_list})")
        
        if self.capture_rtp or self.capture_rtcp:
            rtp_start, rtp_end = self.rtp_port_range
            port_filters.append(f"(portrange {rtp_start}-{rtp_end})")
        
        if port_filters:
            filters.append(f"({' or '.join(port_filters)})")
        
        return ' and '.join(filters)


class PacketCapture:
    """
    Packet capture engine
    Captures and processes network packets in real-time
    """
    
    def __init__(
        self,
        capture_filter: Optional[CaptureFilter] = None,
        on_packet: Optional[Callable] = None
    ):
        """
        Initialize packet capture
        
        Args:
            capture_filter: Capture filter configuration
            on_packet: Callback function for each packet
        """
        if not SCAPY_AVAILABLE:
            raise ImportError("Scapy not available. Install with: pip install scapy")
        
        self.filter = capture_filter or CaptureFilter()
        self.on_packet = on_packet
        
        self.running = False
        self.capture_thread = None
        self.packets_captured = 0
        self.packets_dropped = 0
        self.start_time = None
        
        # Storage
        self.captured_packets = []
        self.max_stored_packets = 10000
        
        logger.info(f"PacketCapture initialized with filter: {self.filter.to_bpf_filter()}")
    
    def start(self):
        """Start packet capture"""
        if self.running:
            logger.warning("Packet capture already running")
            return
        
        self.running = True
        self.start_time = datetime.now()
        self.packets_captured = 0
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        logger.info("Packet capture started")
    
    def stop(self):
        """Stop packet capture"""
        if not self.running:
            return
        
        self.running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        duration = (datetime.now() - self.start_time).total_seconds()
        logger.info(
            f"Packet capture stopped. "
            f"Captured {self.packets_captured} packets in {duration:.2f}s"
        )
    
    def _capture_loop(self):
        """Main capture loop"""
        try:
            # Build BPF filter
            bpf_filter = self.filter.to_bpf_filter()
            
            logger.info(f"Starting scapy sniff with filter: {bpf_filter}")
            
            # Start sniffing
            sniff(
                filter=bpf_filter,
                iface=self.filter.interface,
                prn=self._process_packet,
                store=False,
                stop_filter=lambda x: not self.running
            )
            
        except Exception as e:
            logger.error(f"Error in capture loop: {e}", exc_info=True)
            self.running = False
    
    def _process_packet(self, packet: ScapyPacket):
        """
        Process captured packet
        
        Args:
            packet: Scapy packet object
        """
        try:
            # Check if it's a UDP packet
            if not packet.haslayer(UDP):
                return
            
            udp = packet[UDP]
            ip = packet[IP] if packet.haslayer(IP) else None
            
            if not ip:
                return
            
            # Extract packet info
            packet_info = {
                'timestamp': datetime.now(),
                'src_ip': ip.src,
                'dst_ip': ip.dst,
                'src_port': udp.sport,
                'dst_port': udp.dport,
                'length': len(packet),
                'raw_packet': bytes(packet),
            }
            
            # Determine packet type
            packet_type = self._identify_packet_type(udp.sport, udp.dport, packet)
            packet_info['type'] = packet_type
            
            # Extract payload
            if packet.haslayer(Raw):
                packet_info['payload'] = bytes(packet[Raw].load)
            else:
                packet_info['payload'] = b''
            
            # Store packet
            if len(self.captured_packets) < self.max_stored_packets:
                self.captured_packets.append(packet_info)
            else:
                self.packets_dropped += 1
            
            self.packets_captured += 1
            
            # Call callback
            if self.on_packet:
                try:
                    self.on_packet(packet_info)
                except Exception as e:
                    logger.error(f"Error in packet callback: {e}")
            
            # Log periodically
            if self.packets_captured % 100 == 0:
                logger.debug(f"Captured {self.packets_captured} packets")
                
        except Exception as e:
            logger.error(f"Error processing packet: {e}", exc_info=True)
    
    def _identify_packet_type(
        self,
        src_port: int,
        dst_port: int,
        packet: ScapyPacket
    ) -> str:
        """
        Identify packet type (SIP, RTP, RTCP)
        
        Args:
            src_port: Source port
            dst_port: Destination port
            packet: Scapy packet
            
        Returns:
            Packet type string
        """
        # Check SIP ports
        if src_port in self.filter.sip_ports or dst_port in self.filter.sip_ports:
            if packet.haslayer(Raw):
                payload = bytes(packet[Raw].load)
                # Check if it looks like SIP (starts with SIP method or SIP/2.0)
                if payload.startswith(b'SIP/') or payload.startswith(b'INVITE') or \
                   payload.startswith(b'REGISTER') or payload.startswith(b'ACK') or \
                   payload.startswith(b'BYE') or payload.startswith(b'OPTIONS'):
                    return 'SIP'
        
        # Check RTP/RTCP range
        rtp_start, rtp_end = self.filter.rtp_port_range
        if (rtp_start <= src_port <= rtp_end) or (rtp_start <= dst_port <= rtp_end):
            if packet.haslayer(Raw):
                payload = bytes(packet[Raw].load)
                if len(payload) >= 12:
                    # Check RTP header
                    first_byte = payload[0]
                    version = (first_byte >> 6) & 0x03
                    payload_type = payload[1] & 0x7F
                    
                    # RTP version should be 2
                    if version == 2:
                        # Check if RTCP (payload type 200-204)
                        if 200 <= payload_type <= 204:
                            return 'RTCP'
                        else:
                            return 'RTP'
        
        return 'UNKNOWN'
    
    def get_packets(
        self,
        packet_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> list:
        """
        Get captured packets
        
        Args:
            packet_type: Filter by packet type ('SIP', 'RTP', 'RTCP')
            limit: Maximum number of packets to return
            
        Returns:
            List of packet info dictionaries
        """
        packets = self.captured_packets
        
        if packet_type:
            packets = [p for p in packets if p['type'] == packet_type]
        
        if limit:
            packets = packets[-limit:]
        
        return packets
    
    def clear_packets(self):
        """Clear stored packets"""
        self.captured_packets.clear()
        logger.info("Cleared captured packets")
    
    def save_pcap(self, filename: str):
        """
        Save captured packets to PCAP file
        
        Args:
            filename: Output PCAP filename
        """
        if not self.captured_packets:
            logger.warning("No packets to save")
            return
        
        try:
            # Convert stored packets back to scapy packets
            from scapy.all import Ether
            packets = []
            
            for pkt_info in self.captured_packets:
                # Reconstruct packet from raw bytes
                pkt = Ether(pkt_info['raw_packet'])
                packets.append(pkt)
            
            # Write to PCAP file
            wrpcap(filename, packets)
            logger.info(f"Saved {len(packets)} packets to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving PCAP: {e}", exc_info=True)
    
    def get_statistics(self) -> dict:
        """
        Get capture statistics
        
        Returns:
            Statistics dictionary
        """
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            rate = self.packets_captured / duration if duration > 0 else 0
        else:
            duration = 0
            rate = 0
        
        # Count by type
        type_counts = {}
        for pkt in self.captured_packets:
            pkt_type = pkt['type']
            type_counts[pkt_type] = type_counts.get(pkt_type, 0) + 1
        
        return {
            'running': self.running,
            'duration_seconds': duration,
            'packets_captured': self.packets_captured,
            'packets_stored': len(self.captured_packets),
            'packets_dropped': self.packets_dropped,
            'capture_rate': rate,
            'packet_types': type_counts,
        }
