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
        
        # Message tracking for retransmission detection (RFC 3261)
        # Key: (Call-ID, CSeq), Value: (timestamp, response_sent)
        self.message_cache: Dict[tuple, tuple] = {}
        self.message_cache_timeout = 32  # seconds (T1 * 64)
        
        # Retransmission timers (RFC 3261)
        self.T1 = 0.5  # RTT estimate, 500ms
        self.T2 = 4.0  # Maximum retransmit interval, 4s
        
        # Registration configuration
        # Use shorter expiry (60s) to force frequent re-registration
        # This ensures quick recovery after server restarts
        self.registration_expiry = self.sip_config.get('registration_expiry', 60)
        
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
            
            # Check for retransmissions (RFC 3261)
            if message.method:
                call_id = message.get_header('Call-ID')
                cseq = message.get_header('CSeq')
                
                if call_id and cseq:
                    cache_key = (call_id, cseq)
                    current_time = datetime.now().timestamp()
                    
                    # Clean old cache entries
                    self._clean_message_cache(current_time)
                    
                    # Check if this is a retransmission
                    if cache_key in self.message_cache:
                        cached_time, cached_response = self.message_cache[cache_key]
                        
                        # If we've seen this message recently, it's likely a retransmission
                        if current_time - cached_time < self.message_cache_timeout:
                            logger.debug(f"Detected retransmission: {message.method} {call_id} {cseq}")
                            
                            # Resend the cached response
                            if cached_response:
                                self.socket.sendto(cached_response.encode('utf-8'), addr)
                                logger.debug(f"Resent cached response for retransmission")
                            return
                    
                    # Store message in cache (will be updated with response later)
                    self.message_cache[cache_key] = (current_time, None)
            
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
        
        logger.info(f"=== REQUEST HANDLER === Method: {method}, From: {addr}")
        
        if method == SIPMethod.REGISTER:
            self._handle_register(message, addr)
        elif method == SIPMethod.INVITE:
            self._handle_invite(message, addr)
        elif method == SIPMethod.ACK:
            self._handle_ack(message, addr)
        elif method == SIPMethod.BYE:
            logger.info(f"Routing to BYE handler for Call-ID: {message.get_header('Call-ID')}")
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
            
            # Parse user from From header
            user = self._extract_user(from_header)
            
            if not user:
                self._send_response(message, SIPResponse.BAD_REQUEST, addr)
                return
            
            # Block HT813 registrations (device no longer in use)
            if addr[0] == "192.168.100.96" or user in ["1001", "1002"]:
                logger.warning(f"⚠️ Blocked REGISTER from legacy HT813 device: {user} at {addr[0]}:{addr[1]}")
                # Send 403 Forbidden to prevent re-registration attempts
                self._send_response(message, SIPResponse.FORBIDDEN, addr)
                return
            
            # IMPORTANT: Override expiry to shorter time for quick recovery after restarts
            # Client requests expiry, but we set our own shorter value (60s default)
            requested_expires = int(expires_header) if expires_header else 3600
            actual_expires = self.registration_expiry
            
            logger.info(f"REGISTER from user: {user} at {addr[0]}:{addr[1]} (requested: {requested_expires}s, granting: {actual_expires}s)")
            
            # Store registration with our shorter expiry
            registration = Registration(
                user=user,
                contact=contact,
                expires=actual_expires,
                registered_at=datetime.now(),
                ip_address=addr[0],
                port=addr[1]
            )
            
            self.registrations[user] = registration
            
            logger.info(f"✓ User {user} registered successfully, expires in {actual_expires}s")
            
            # Callback
            if self.on_register:
                self.on_register(registration)
            
            # Send 200 OK with Contact header and OUR expiry time (RFC 3261 compliance)
            # The Expires value we send back tells the client when to re-register
            additional_headers = {
                'Contact': contact,  # Echo back the Contact header
                'Expires': str(actual_expires)  # Send OUR expiry value (not client's request)
            }
            self._send_response(message, SIPResponse.OK, addr, additional_headers=additional_headers)
            
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
            sdp = None
            remote_ip = None
            remote_port = None
            codec = None
            
            # Parse SDP from message body if present
            if message.body and message.body.strip():
                from ..core.sip.parser import parse_sdp
                try:
                    sdp = parse_sdp(message.body)
                    logger.info(f"Parsed SDP from INVITE: {sdp}")
                except Exception as e:
                    logger.error(f"Failed to parse SDP: {e}", exc_info=True)
            
            if sdp:
                remote_ip = sdp.get('connection', {}).get('address')
                if 'media' in sdp:
                    for media in sdp['media']:
                        if media.get('type') == 'audio':
                            remote_port = media.get('port')
                            # Get first codec
                            if media.get('formats'):
                                codec = media['formats'][0]
                            break
                logger.info(f"Extracted RTP info: remote_ip={remote_ip}, remote_port={remote_port}, codec={codec}")
            
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
                to_tag = SIPBuilder.generate_tag()
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
            
            logger.info(f"=== BYE HANDLER CALLED ===")
            logger.info(f"BYE received for call-id: {call_id}")
            logger.info(f"Active call sessions: {list(self.call_sessions.keys())}")
            
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
                logger.info(f"Sent 200 OK response for BYE")
            else:
                logger.warning(f"BYE received for unknown call: {call_id}")
                logger.warning(f"Call-ID {call_id} not found in call_sessions. This might indicate the call was already cleaned up or never established")
                # Send 200 OK anyway to gracefully handle the BYE
                # This is more robust than sending 481
                self._send_response(message, SIPResponse.OK, addr)
                logger.info(f"Sent 200 OK response for BYE (unknown call)")
            
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
        addr: tuple,
        additional_headers: Optional[Dict[str, str]] = None
    ):
        """Send SIP response"""
        try:
            response = SIPBuilder.build_response(
                request=request,
                status_code=int(status_code),  # Convert enum to int
                additional_headers=additional_headers
            )
            
            self.socket.sendto(response.encode('utf-8'), addr)
            
            logger.debug(f"Sent {status_code} to {addr[0]}:{addr[1]}")
            
            # Cache the response for retransmission detection
            call_id = request.get_header('Call-ID')
            cseq = request.get_header('CSeq')
            
            if call_id and cseq:
                cache_key = (call_id, cseq)
                current_time = datetime.now().timestamp()
                self.message_cache[cache_key] = (current_time, response)
            
        except Exception as e:
            logger.error(f"Error sending response: {e}", exc_info=True)
    
    def _send_answer(self, request: SIPMessage, addr: tuple, to_tag: str):
        """Send 200 OK answer with SDP"""
        try:
            # Get local RTP port
            local_rtp_port = self._allocate_rtp_port()
            
            # Get server's actual IP (not 0.0.0.0)
            server_ip = self._get_server_ip()
            logger.info(f"Using server IP {server_ip} in SDP (RTP port {local_rtp_port})")
            
            # Build SDP body string
            sdp_body = SIPBuilder.build_sdp(
                host=server_ip,  # Use actual IP, not bind address
                port=local_rtp_port,
                session_name="BeautyAI PABX",
                codecs=[0, 8]  # PCMU, PCMA
            )
            
            # Build Contact header
            to_user = self._extract_user(request.to_header)
            # Use the actual IP that the client connected from (not 0.0.0.0)
            contact_host = self.host if self.host != '0.0.0.0' else addr[0]
            contact = f"<sip:{to_user}@{contact_host}:{self.port}>"
            
            # Build response with SDP and Contact header
            response = SIPBuilder.build_response(
                status_code=int(SIPResponse.OK),  # Convert enum to int
                request=request,
                to_tag=to_tag,
                body=sdp_body,
                additional_headers={'Contact': contact}
            )
            
            self.socket.sendto(response.encode('utf-8'), addr)
            
            logger.info(f"Sent 200 OK answer to {addr[0]}:{addr[1]}")
            
        except Exception as e:
            logger.error(f"Error sending answer: {e}", exc_info=True)
    
    def initiate_call(self, from_user: str, to_number: str) -> bool:
        """
        Initiate outbound call
        
        Args:
            from_user: Calling user ID (e.g., "1001")
            to_number: Destination phone number
            
        Returns:
            True if INVITE was sent successfully
        """
        try:
            # Get registration for from_user
            if from_user not in self.registrations:
                logger.error(f"User {from_user} not registered")
                return False
            
            registration = self.registrations[from_user]
            dest_addr = (registration.ip_address, registration.port)
            
            # Get server's actual IP address (not 0.0.0.0)
            server_ip = self._get_server_ip()
            
            # Build INVITE request
            request_uri = f"sip:{to_number}@{registration.ip_address}:{registration.port}"
            from_uri = f"sip:{from_user}@{server_ip}"
            to_uri = f"sip:{to_number}@{server_ip}"
            
            # Generate simple SDP
            local_rtp_port = self._allocate_rtp_port()
            sdp_body = SIPBuilder.build_sdp(
                host=server_ip,
                port=local_rtp_port,
                session_name="BeautyAI Outbound Call",
                codecs=[0, 8]  # PCMU, PCMA
            )
            
            invite = SIPBuilder.build_request(
                method=SIPMethod.INVITE,
                request_uri=request_uri,
                from_uri=from_uri,
                to_uri=to_uri,
                via_host=server_ip,
                via_port=self.port,
                contact=f"sip:{from_user}@{server_ip}:{self.port}",
                body=sdp_body
            )
            
            # Send INVITE
            self.socket.sendto(invite.encode('utf-8'), dest_addr)
            
            logger.info(f"Initiated call from {from_user} to {to_number}")
            return True
            
        except Exception as e:
            logger.error(f"Error initiating call: {e}", exc_info=True)
            return False
    
    def _get_server_ip(self) -> str:
        """Get server's actual IP address"""
        # Get from config first
        server_config = self.config.get('server', {})
        host = server_config.get('host')
        
        if host and host != '0.0.0.0':
            return host
        
        # Fallback: try to determine from network
        try:
            # Create a dummy socket to get local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except:
            return '127.0.0.1'  # Final fallback
    
    def _allocate_rtp_port(self) -> int:
        """Allocate RTP port for new call"""
        # Simple port allocation
        rtp_config = self.config.get('rtp')
        return rtp_config.get('port_range', {}).get('start', 10000)
    
    def _clean_message_cache(self, current_time: float):
        """
        Clean expired entries from message cache
        
        Args:
            current_time: Current timestamp
        """
        expired_keys = [
            key for key, (timestamp, _) in self.message_cache.items()
            if current_time - timestamp > self.message_cache_timeout
        ]
        
        for key in expired_keys:
            del self.message_cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned {len(expired_keys)} expired message cache entries")
    
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
