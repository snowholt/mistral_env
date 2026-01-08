"""
Packet capture exporters
Export captures to various formats (PCAP, JSON, text)
"""

import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PcapExporter:
    """
    Export captured packets to PCAP format
    """
    
    @staticmethod
    def export(
        packets: List[dict],
        output_file: str
    ):
        """
        Export packets to PCAP file
        
        Args:
            packets: List of packet info dictionaries
            output_file: Output PCAP file path
        """
        try:
            from scapy.all import wrpcap, Ether
            
            # Convert packets
            scapy_packets = []
            for pkt_info in packets:
                if 'raw_packet' in pkt_info:
                    pkt = Ether(pkt_info['raw_packet'])
                    scapy_packets.append(pkt)
            
            # Ensure directory exists
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write PCAP
            wrpcap(str(output_path), scapy_packets)
            
            logger.info(f"Exported {len(scapy_packets)} packets to {output_file}")
            
        except Exception as e:
            logger.error(f"Error exporting PCAP: {e}", exc_info=True)
            raise


class JsonExporter:
    """
    Export capture data to JSON format
    """
    
    @staticmethod
    def export_packets(
        packets: List[dict],
        output_file: str,
        include_payload: bool = False
    ):
        """
        Export packets to JSON file
        
        Args:
            packets: List of packet info dictionaries
            output_file: Output JSON file path
            include_payload: Include packet payload in export
        """
        try:
            # Prepare data for JSON serialization
            export_data = []
            
            for pkt in packets:
                pkt_data = {
                    'timestamp': pkt['timestamp'].isoformat() if isinstance(pkt['timestamp'], datetime) else str(pkt['timestamp']),
                    'type': pkt.get('type'),
                    'src_ip': pkt.get('src_ip'),
                    'dst_ip': pkt.get('dst_ip'),
                    'src_port': pkt.get('src_port'),
                    'dst_port': pkt.get('dst_port'),
                    'length': pkt.get('length'),
                }
                
                if include_payload and 'payload' in pkt:
                    # Convert payload to hex string
                    pkt_data['payload_hex'] = pkt['payload'].hex()
                
                export_data.append(pkt_data)
            
            # Write JSON
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported {len(export_data)} packets to {output_file}")
            
        except Exception as e:
            logger.error(f"Error exporting JSON: {e}", exc_info=True)
            raise
    
    @staticmethod
    def export_sessions(
        sessions: List,
        output_file: str
    ):
        """
        Export session data to JSON
        
        Args:
            sessions: List of session objects
            output_file: Output JSON file path
        """
        try:
            # Convert sessions to dict
            session_data = []
            
            for session in sessions:
                if hasattr(session, '__dict__'):
                    # Convert dataclass to dict
                    data = {}
                    for key, value in session.__dict__.items():
                        if isinstance(value, datetime):
                            data[key] = value.isoformat()
                        elif isinstance(value, list):
                            # Skip large lists
                            if key in ['packets', 'sequence_numbers']:
                                data[key] = f"<{len(value)} items>"
                            else:
                                data[key] = value
                        else:
                            data[key] = value
                    session_data.append(data)
            
            # Write JSON
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            logger.info(f"Exported {len(session_data)} sessions to {output_file}")
            
        except Exception as e:
            logger.error(f"Error exporting session JSON: {e}", exc_info=True)
            raise


class TextExporter:
    """
    Export capture data to human-readable text format
    """
    
    @staticmethod
    def export_summary(
        packets: List[dict],
        sessions: Optional[List] = None,
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate text summary of capture
        
        Args:
            packets: List of packet info dictionaries
            sessions: Optional list of session objects
            output_file: Optional output file path
            
        Returns:
            Summary text
        """
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append("PACKET CAPTURE SUMMARY")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Total Packets: {len(packets)}")
        lines.append("")
        
        # Packet type breakdown
        type_counts = {}
        for pkt in packets:
            pkt_type = pkt.get('type', 'UNKNOWN')
            type_counts[pkt_type] = type_counts.get(pkt_type, 0) + 1
        
        lines.append("Packet Types:")
        for pkt_type, count in sorted(type_counts.items()):
            lines.append(f"  {pkt_type}: {count}")
        lines.append("")
        
        # Session summary
        if sessions:
            lines.append(f"Total Sessions: {len(sessions)}")
            lines.append("")
            
            lines.append("Session Details:")
            for i, session in enumerate(sessions, 1):
                lines.append(f"\nSession {i}:")
                if hasattr(session, 'call_id'):
                    lines.append(f"  Call ID: {session.call_id}")
                    lines.append(f"  From: {session.from_user}")
                    lines.append(f"  To: {session.to_user}")
                    lines.append(f"  State: {session.state}")
                    lines.append(f"  Duration: {session.duration:.2f}s")
                elif hasattr(session, 'ssrc'):
                    lines.append(f"  SSRC: {session.ssrc}")
                    lines.append(f"  Route: {session.session_key}")
                    lines.append(f"  Payload Type: {session.payload_type}")
                    lines.append(f"  Packets: {session.packet_count}")
                    lines.append(f"  Loss: {session.packet_loss}")
                    lines.append(f"  Duration: {session.duration:.2f}s")
        
        lines.append("")
        lines.append("=" * 80)
        
        # Join lines
        summary = "\n".join(lines)
        
        # Write to file if specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(summary)
            
            logger.info(f"Exported summary to {output_file}")
        
        return summary
