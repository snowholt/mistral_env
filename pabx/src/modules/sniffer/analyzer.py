"""
Packet analyzer and session tracker
Analyzes captured packets and correlates them into sessions
"""

from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field

from src.core.sip.parser import SIPParser
from src.core.rtp.packet import parse_rtp_packet
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SIPTransaction:
    """SIP transaction (request + responses)"""
    call_id: str
    method: str
    from_user: Optional[str] = None
    to_user: Optional[str] = None
    request_timestamp: Optional[datetime] = None
    responses: List[dict] = field(default_factory=list)
    request_packet: Optional[dict] = None
    
    def add_response(self, status_code: int, timestamp: datetime, packet: dict):
        """Add response to transaction"""
        self.responses.append({
            'status_code': status_code,
            'timestamp': timestamp,
            'packet': packet
        })


@dataclass
class RTPSession:
    """RTP media session"""
    ssrc: int
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    payload_type: int
    start_time: datetime
    end_time: Optional[datetime] = None
    packets: List[dict] = field(default_factory=list)
    sequence_numbers: List[int] = field(default_factory=list)
    packet_count: int = 0
    packet_loss: int = 0
    bytes_received: int = 0
    
    @property
    def duration(self) -> float:
        """Get session duration in seconds"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (datetime.now() - self.start_time).total_seconds()
    
    @property
    def session_key(self) -> str:
        """Get unique session key"""
        return f"{self.src_ip}:{self.src_port}->{self.dst_ip}:{self.dst_port}"


@dataclass
class CallSession:
    """Complete call session (SIP + RTP)"""
    call_id: str
    from_user: Optional[str] = None
    to_user: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    state: str = "INIT"  # INIT, RINGING, ACTIVE, ENDED
    sip_transactions: List[SIPTransaction] = field(default_factory=list)
    rtp_sessions: List[RTPSession] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        """Get call duration in seconds"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (datetime.now() - self.start_time).total_seconds()


class PacketAnalyzer:
    """
    Analyzes packets and extracts protocol information
    """
    
    @staticmethod
    def analyze_sip_packet(packet_info: dict) -> Optional[dict]:
        """
        Analyze SIP packet
        
        Args:
            packet_info: Packet information dictionary
            
        Returns:
            Analyzed SIP information or None if not SIP
        """
        if packet_info['type'] != 'SIP':
            return None
        
        payload = packet_info.get('payload', b'')
        if not payload:
            return None
        
        try:
            # Decode payload
            payload_str = payload.decode('utf-8', errors='ignore')
            
            # Parse SIP message
            sip_msg = SIPParser.parse(payload_str)
            if not sip_msg:
                return None
            
            # Extract key information
            result = {
                'timestamp': packet_info['timestamp'],
                'src_ip': packet_info['src_ip'],
                'dst_ip': packet_info['dst_ip'],
                'is_request': sip_msg.is_request,
                'call_id': sip_msg.call_id,
            }
            
            if sip_msg.is_request:
                result['method'] = sip_msg.method.value if sip_msg.method else None
                result['request_uri'] = sip_msg.request_uri
                
                # Extract users from headers
                from_hdr = sip_msg.from_header
                to_hdr = sip_msg.to_header
                
                if from_hdr:
                    result['from_user'] = PacketAnalyzer._extract_user(from_hdr)
                if to_hdr:
                    result['to_user'] = PacketAnalyzer._extract_user(to_hdr)
                
                # Extract RTP port from SDP if INVITE
                if sip_msg.method and sip_msg.method.value == 'INVITE':
                    sdp = sip_msg.extract_sdp()
                    if sdp and sdp.get('media'):
                        for media in sdp['media']:
                            if media['type'] == 'audio':
                                result['rtp_port'] = media['port']
                                result['codecs'] = media['formats']
                                break
            else:
                result['status_code'] = sip_msg.status_code
                result['reason_phrase'] = sip_msg.reason_phrase
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing SIP packet: {e}")
            return None
    
    @staticmethod
    def analyze_rtp_packet(packet_info: dict) -> Optional[dict]:
        """
        Analyze RTP packet
        
        Args:
            packet_info: Packet information dictionary
            
        Returns:
            Analyzed RTP information or None if not RTP
        """
        if packet_info['type'] != 'RTP':
            return None
        
        payload = packet_info.get('payload', b'')
        if not payload:
            return None
        
        try:
            # Parse RTP packet
            rtp_pkt = parse_rtp_packet(payload)
            if not rtp_pkt:
                return None
            
            # Extract information
            result = {
                'timestamp': packet_info['timestamp'],
                'src_ip': packet_info['src_ip'],
                'dst_ip': packet_info['dst_ip'],
                'src_port': packet_info['src_port'],
                'dst_port': packet_info['dst_port'],
                'ssrc': rtp_pkt.header.ssrc,
                'payload_type': rtp_pkt.header.payload_type,
                'sequence_number': rtp_pkt.header.sequence_number,
                'rtp_timestamp': rtp_pkt.header.timestamp,
                'marker': rtp_pkt.header.marker,
                'payload_size': len(rtp_pkt.payload),
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing RTP packet: {e}")
            return None
    
    @staticmethod
    def _extract_user(sip_header: str) -> Optional[str]:
        """Extract username from SIP header"""
        if 'sip:' in sip_header:
            parts = sip_header.split('sip:')[1].split('@')
            if parts:
                return parts[0].strip('<>')
        return None


class SessionTracker:
    """
    Tracks and correlates packets into sessions
    """
    
    def __init__(self):
        """Initialize session tracker"""
        self.call_sessions: Dict[str, CallSession] = {}
        self.rtp_sessions: Dict[int, RTPSession] = {}
        self.sip_transactions: Dict[str, SIPTransaction] = {}
    
    def process_packet(self, packet_info: dict):
        """
        Process packet and update sessions
        
        Args:
            packet_info: Packet information dictionary
        """
        packet_type = packet_info.get('type')
        
        if packet_type == 'SIP':
            self._process_sip_packet(packet_info)
        elif packet_type == 'RTP':
            self._process_rtp_packet(packet_info)
    
    def _process_sip_packet(self, packet_info: dict):
        """Process SIP packet and update call sessions"""
        sip_data = PacketAnalyzer.analyze_sip_packet(packet_info)
        if not sip_data:
            return
        
        call_id = sip_data.get('call_id')
        if not call_id:
            return
        
        # Get or create call session
        if call_id not in self.call_sessions:
            self.call_sessions[call_id] = CallSession(
                call_id=call_id,
                from_user=sip_data.get('from_user'),
                to_user=sip_data.get('to_user'),
                start_time=sip_data['timestamp']
            )
        
        session = self.call_sessions[call_id]
        
        # Update session based on message type
        if sip_data['is_request']:
            method = sip_data.get('method')
            
            # Create transaction
            txn = SIPTransaction(
                call_id=call_id,
                method=method,
                from_user=sip_data.get('from_user'),
                to_user=sip_data.get('to_user'),
                request_timestamp=sip_data['timestamp'],
                request_packet=packet_info
            )
            session.sip_transactions.append(txn)
            self.sip_transactions[f"{call_id}:{method}"] = txn
            
            # Update session state
            if method == 'INVITE':
                session.state = 'INVITING'
            elif method == 'ACK':
                session.state = 'ACTIVE'
            elif method == 'BYE':
                session.state = 'ENDING'
                session.end_time = sip_data['timestamp']
        else:
            # Response
            status_code = sip_data.get('status_code')
            
            # Find corresponding transaction
            for txn in session.sip_transactions:
                if txn.call_id == call_id:
                    txn.add_response(status_code, sip_data['timestamp'], packet_info)
            
            # Update session state based on status
            if status_code == 180:
                session.state = 'RINGING'
            elif status_code == 200:
                if session.state == 'INVITING':
                    session.state = 'ANSWERED'
                elif session.state == 'ENDING':
                    session.state = 'ENDED'
                    session.end_time = sip_data['timestamp']
    
    def _process_rtp_packet(self, packet_info: dict):
        """Process RTP packet and update media sessions"""
        rtp_data = PacketAnalyzer.analyze_rtp_packet(packet_info)
        if not rtp_data:
            return
        
        ssrc = rtp_data['ssrc']
        
        # Get or create RTP session
        if ssrc not in self.rtp_sessions:
            self.rtp_sessions[ssrc] = RTPSession(
                ssrc=ssrc,
                src_ip=rtp_data['src_ip'],
                src_port=rtp_data['src_port'],
                dst_ip=rtp_data['dst_ip'],
                dst_port=rtp_data['dst_port'],
                payload_type=rtp_data['payload_type'],
                start_time=rtp_data['timestamp']
            )
        
        session = self.rtp_sessions[ssrc]
        
        # Update session
        session.packet_count += 1
        session.bytes_received += rtp_data['payload_size']
        session.sequence_numbers.append(rtp_data['sequence_number'])
        session.end_time = rtp_data['timestamp']
        
        # Detect packet loss
        if len(session.sequence_numbers) > 1:
            expected = (session.sequence_numbers[-2] + 1) & 0xFFFF
            actual = rtp_data['sequence_number']
            if actual != expected:
                # Packet loss detected
                if actual > expected:
                    loss = actual - expected
                else:
                    loss = (0xFFFF - expected) + actual + 1
                session.packet_loss += loss
        
        # Store packet (limited)
        if len(session.packets) < 1000:
            session.packets.append(rtp_data)
    
    def get_call_session(self, call_id: str) -> Optional[CallSession]:
        """Get call session by call ID"""
        return self.call_sessions.get(call_id)
    
    def get_rtp_session(self, ssrc: int) -> Optional[RTPSession]:
        """Get RTP session by SSRC"""
        return self.rtp_sessions.get(ssrc)
    
    def get_all_calls(self) -> List[CallSession]:
        """Get all call sessions"""
        return list(self.call_sessions.values())
    
    def get_active_calls(self) -> List[CallSession]:
        """Get active call sessions"""
        return [s for s in self.call_sessions.values() if s.state in ['ACTIVE', 'RINGING']]
    
    def get_statistics(self) -> dict:
        """Get session statistics"""
        return {
            'total_calls': len(self.call_sessions),
            'active_calls': len(self.get_active_calls()),
            'total_rtp_sessions': len(self.rtp_sessions),
            'total_packets_processed': sum(s.packet_count for s in self.rtp_sessions.values()),
        }
