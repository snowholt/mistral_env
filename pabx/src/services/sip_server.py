"""
SIP server engine
Handle SIP protocol operations and call management
"""

import socket
import threading
from typing import Optional, Dict, Callable
from dataclasses import dataclass, field
from datetime import datetime

from ..core.sip.parser import SIPParser, SIPMessage
from ..core.sip.builder import SIPBuilder
from ..core.sip.types import SIPMethod, SIPResponse
from ..utils.config import Config
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Registration:
    """SIP registration information"""
    user: str
    contact: str
    expires: int
    registered_at: datetime
    ip_address: str
    port: int


@dataclass
class CallSession:
    """Active call session"""
    call_id: str
    from_user: str
    to_user: str
    from_tag: str
    to_tag: Optional[str] = None
    state: str = "INIT"
    
    # Media info
    remote_ip: Optional[str] = None
    remote_port: Optional[int] = None
    local_port: Optional[int] = None
    codec: Optional[str] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    answered_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None


class SIPServer:
    """
    SIP server for handling registrations and calls
    """
    
    def __init__(self):
        """Initialize SIP server"""
        self.config = Config()
        self.sip_config = self.config.get('sip')
        
        # Bind address
        self.host = self.sip_config.get('bind_address', '0.0.0.0')
        self.port = self.sip_config.get('port', 5060)
        
        # State
        self.running = False
        self.socket: Optional[socket.socket] = None
        self.thread: Optional[threading.Thread] = None
        
        # Registrations and sessions
        self.registrations: Dict[str, Registration] = {}
        self.call_sessions: Dict[str, CallSession] = {}
        
        # Callbacks
        self.on_register: Optional[Callable] = None
        self.on_invite: Optional[Callable] = None
        self.on_bye: Optional[Callable] = None
        
        logger.info(f"SIP server initialized on {self.host}:{self.port}")
    
    def start(self):
        """Start the SIP server"""
        if self.running:
            logger.warning("SIP server already running")
            return
        
        try:
            # Create UDP socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            
            # Start receive thread
            self.running = True
            self.thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.thread.start()
            
            logger.info(f"SIP server started on {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Error starting SIP server: {e}", exc_info=True)
            self.running = False
            raise
    
    def stop(self):
        """Stop the SIP server"""
        if not self.running:
            return
        
        self.running = False
        
        if self.socket:
            self.socket.close()
        
        if self.thread:
            self.thread.join(timeout=2)
        
        logger.info("SIP server stopped")
    
    def _receive_loop(self):
        """Main receive loop"""
        logger.info("SIP server receive loop started")
        
        while self.running:
            try:
                # Receive message
                data, addr = self.socket.recvfrom(65535)
                
                # Process in separate thread to not block receive
                threading.Thread(
                    target=self._handle_message,
                    args=(data, addr),
                    daemon=True
                ).start()
                
            except Exception as e:
                if self.running:
                    logger.error(f"Error in receive loop: {e}", exc_info=True)
        
        logger.info("SIP server receive loop ended")
    
    def _handle_message(self, data: bytes, addr: tuple):
        """
        Handle incoming SIP message
        
        Args:
            data: Raw message data
            addr: Source address (ip, port)
        """
        try:
            # Parse message
            message = SIPParser.parse(data.decode('utf-8'))
            
            logger.info(f"Received {message.method or message.status_code} from {addr[0]}:{addr[1]}")
            
            # Route by method
            if message.method:
                self._handle_request(message, addr)
            else:
                self._handle_response(message, addr)
                
        except Exception as e:
            logger.error(f"Error handling message from {addr}: {e}", exc_info=True)
    
    def _handle_request(self, message: SIPMessage, addr: tuple):
        """Handle SIP request"""
        method = message.method
        
        if method == SIPMethod.REGISTER:
            self._handle_register(message, addr)
        elif method == SIPMethod.INVITE:
            self._handle_invite(message, addr)
        elif method == SIPMethod.ACK:
            self._handle_ack(message, addr)
        elif method == SIPMethod.BYE:
            self._handle_bye(message, addr)
        elif method == SIPMethod.CANCEL:
            self._handle_cancel(message, addr)
        elif method == SIPMethod.OPTIONS:
            self._handle_options(message, addr)
        else:
            logger.warning(f"Unhandled method: {method}")
            # Send 501 Not Implemented
            self._send_response(
                message,
                SIPResponse.NOT_IMPLEMENTED,
                addr
            )
    
    def _handle_response(self, message: SIPMessage, addr: tuple):
        """Handle SIP response"""
        logger.info(f"Received response: {message.status_code} {message.reason_phrase}")
    
    def _handle_register(self, message: SIPMessage, addr: tuple):
        """Handle REGISTER request"""
        try:
            # Extract registration info
            from_header = message.get_header('From')
            contact = message.get_header('Contact')
            expires_header = message.get_header('Expires')
            expires = int(expires_header) if expires_header else 3600
            
            # Parse user from From header
            user = self._extract_user(from_header)
            
            if not user:
                self._send_response(message, SIPResponse.BAD_REQUEST, addr)
                return
            
            # Store registration
            registration = Registration(
                user=user,
                contact=contact,
                expires=expires,
                registered_at=datetime.now(),
                ip_address=addr[0],
                port=addr[1]
            )
            
            self.registrations[user] = registration
            
            logger.info(f"Registered user: {user} from {addr[0]}:{addr[1]}")
            
            # Callback
            if self.on_register:
                self.on_register(registration)
            
            # Send 200 OK
            self._send_response(message, SIPResponse.OK, addr)
            
        except Exception as e:
            logger.error(f"Error handling REGISTER: {e}", exc_info=True)
            self._send_response(message, SIPResponse.SERVER_INTERNAL_ERROR, addr)
    
    def _handle_invite(self, message: SIPMessage, addr: tuple):
        """Handle INVITE request"""
        try:
            # Extract call info
            call_id = message.get_header('Call-ID')
            from_header = message.get_header('From')
            to_header = message.get_header('To')
            
            from_user = self._extract_user(from_header)
            to_user = self._extract_user(to_header)
            from_tag = self._extract_tag(from_header)
            
            # Parse SDP for media info
            sdp = message.sdp
            remote_ip = sdp.get('connection', {}).get('address') if sdp else None
            remote_port = None
            codec = None
            
            if sdp and 'media' in sdp:
                for media in sdp['media']:
                    if media.get('type') == 'audio':
                        remote_port = media.get('port')
                        # Get first codec
                        if media.get('formats'):
                            codec = media['formats'][0]
                        break
            
            # Create call session
            session = CallSession(
                call_id=call_id,
                from_user=from_user,
                to_user=to_user,
                from_tag=from_tag,
                state="INVITING",
                remote_ip=remote_ip,
                remote_port=remote_port,
                codec=codec
            )
            
            self.call_sessions[call_id] = session
            
            logger.info(f"INVITE: {from_user} -> {to_user} (call-id: {call_id})")
            
            # Callback
            if self.on_invite:
                self.on_invite(session)
            
            # Send 180 Ringing
            self._send_response(message, SIPResponse.RINGING, addr)
            
            # Auto-answer if configured
            if self.sip_config.get('call_handling', {}).get('auto_answer', False):
                # Generate to-tag
                to_tag = generate_tag()
                session.to_tag = to_tag
                session.state = "ANSWERED"
                session.answered_at = datetime.now()
                
                # Send 200 OK with SDP
                self._send_answer(message, addr, to_tag)
            
        except Exception as e:
            logger.error(f"Error handling INVITE: {e}", exc_info=True)
            self._send_response(message, SIPResponse.SERVER_INTERNAL_ERROR, addr)
    
    def _handle_ack(self, message: SIPMessage, addr: tuple):
        """Handle ACK request"""
        call_id = message.get_header('Call-ID')
        
        if call_id in self.call_sessions:
            session = self.call_sessions[call_id]
            session.state = "ACTIVE"
            logger.info(f"Call {call_id} is now active")
    
    def _handle_bye(self, message: SIPMessage, addr: tuple):
        """Handle BYE request"""
        try:
            call_id = message.get_header('Call-ID')
            
            if call_id in self.call_sessions:
                session = self.call_sessions[call_id]
                session.state = "ENDED"
                session.ended_at = datetime.now()
                
                logger.info(f"BYE: Call {call_id} ended")
                
                # Callback
                if self.on_bye:
                    self.on_bye(session)
                
                # Remove session
                del self.call_sessions[call_id]
            
            # Send 200 OK
            self._send_response(message, SIPResponse.OK, addr)
            
        except Exception as e:
            logger.error(f"Error handling BYE: {e}", exc_info=True)
            self._send_response(message, SIPResponse.SERVER_INTERNAL_ERROR, addr)
    
    def _handle_cancel(self, message: SIPMessage, addr: tuple):
        """Handle CANCEL request"""
        call_id = message.get_header('Call-ID')
        
        if call_id in self.call_sessions:
            session = self.call_sessions[call_id]
            session.state = "CANCELLED"
            session.ended_at = datetime.now()
            
            logger.info(f"CANCEL: Call {call_id} cancelled")
            
            del self.call_sessions[call_id]
        
        # Send 200 OK
        self._send_response(message, SIPResponse.OK, addr)
    
    def _handle_options(self, message: SIPMessage, addr: tuple):
        """Handle OPTIONS request"""
        # Send 200 OK with capabilities
        self._send_response(message, SIPResponse.OK, addr)
    
    def _send_response(
        self,
        request: SIPMessage,
        status_code: SIPResponse,
        addr: tuple
    ):
        """Send SIP response"""
        try:
            response = SIPBuilder.build_response(
                request=request,
                status_code=status_code
            )
            
            self.socket.sendto(response.encode('utf-8'), addr)
            
            logger.debug(f"Sent {status_code} to {addr[0]}:{addr[1]}")
            
        except Exception as e:
            logger.error(f"Error sending response: {e}", exc_info=True)
    
    def _send_answer(self, request: SIPMessage, addr: tuple, to_tag: str):
        """Send 200 OK answer with SDP"""
        try:
            # Get local RTP port
            local_rtp_port = self._allocate_rtp_port()
            
            # Build SDP
            sdp = {
                'version': 0,
                'origin': {
                    'username': 'beautyai',
                    'session_id': '0',
                    'session_version': '0',
                    'network_type': 'IN',
                    'address_type': 'IP4',
                    'address': self.host
                },
                'session_name': 'BeautyAI PABX',
                'connection': {
                    'network_type': 'IN',
                    'address_type': 'IP4',
                    'address': self.host
                },
                'media': [
                    {
                        'type': 'audio',
                        'port': local_rtp_port,
                        'protocol': 'RTP/AVP',
                        'formats': ['0', '8'],  # PCMU, PCMA
                        'attributes': {
                            'rtpmap': ['0 PCMU/8000', '8 PCMA/8000']
                        }
                    }
                ]
            }
            
            # Build response with SDP
            response = SIPBuilder.build_response(
                status_code=SIPResponse.OK,
                request=request,
                local_ip=self.host,
                local_port=self.port,
                to_tag=to_tag,
                sdp=sdp
            )
            
            self.socket.sendto(response.encode('utf-8'), addr)
            
            logger.info(f"Sent 200 OK answer to {addr[0]}:{addr[1]}")
            
        except Exception as e:
            logger.error(f"Error sending answer: {e}", exc_info=True)
    
    def _allocate_rtp_port(self) -> int:
        """Allocate RTP port for new call"""
        # Simple port allocation
        rtp_config = self.config.get('rtp')
        return rtp_config.get('port_range', {}).get('start', 10000)
    
    def _extract_user(self, header: str) -> str:
        """Extract user from SIP header"""
        try:
            # Parse "User Name" <sip:user@domain> format
            if '<' in header and '>' in header:
                uri = header.split('<')[1].split('>')[0]
            else:
                uri = header
            
            # Extract user from sip:user@domain
            if 'sip:' in uri:
                user = uri.split('sip:')[1].split('@')[0]
                return user
        except:
            pass
        return ""
    
    def _extract_tag(self, header: str) -> str:
        """Extract tag parameter from header"""
        try:
            if 'tag=' in header:
                tag = header.split('tag=')[1].split(';')[0].strip()
                return tag
        except:
            pass
        return ""
