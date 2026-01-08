"""
SIP Client for Outbound Registration
Handles registration with external SIP providers (STC)
"""

import socket
import threading
import time
import hashlib
from typing import Optional, Dict
from dataclasses import dataclass
from datetime import datetime

from ..core.sip.parser import SIPParser, SIPMessage
from ..core.sip.builder import SIPBuilder
from ..core.sip.types import SIPMethod, SIPResponse
from ..utils.config import Config
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrunkRegistration:
    """Trunk registration status"""
    server: str
    user_id: str
    registered: bool = False
    expires: int = 3600
    last_register: Optional[datetime] = None
    next_register: Optional[datetime] = None
    error: Optional[str] = None


class SIPClient:
    """
    SIP Client for outbound registration to SIP trunk providers.
    Handles REGISTER with digest authentication and maintains registration.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.parser = SIPParser()
        # SIPBuilder is a static class, no instance needed
        
        # Local addressing (set first, needed for call_id generation)
        self.local_ip = config.get("sip.server.ip", "192.168.100.39")
        self.local_port = config.get("sip.server.port", 5060)
        
        # Socket for outbound communication
        self.socket: Optional[socket.socket] = None
        self.running = False
        
        # Registration state
        self.trunk_registration: Optional[TrunkRegistration] = None
        self.registration_thread: Optional[threading.Thread] = None
        
        # Call sequence
        self.cseq = 1
        self.call_id = self._generate_call_id()
        
        logger.info("SIP Client initialized")
    
    def _generate_call_id(self) -> str:
        """Generate unique Call-ID for registration"""
        timestamp = str(int(time.time() * 1000))
        return f"{timestamp}@{self.local_ip}"
    
    def _generate_nonce_response(self, username: str, password: str, realm: str, 
                                 nonce: str, uri: str, method: str) -> str:
        """
        Generate digest authentication response.
        
        Args:
            username: Authentication username
            password: Authentication password
            realm: Authentication realm from 401 challenge
            nonce: Nonce value from 401 challenge
            uri: Request URI (e.g., sip:domain.com)
            method: SIP method (REGISTER)
        
        Returns:
            Digest response hash
        """
        # HA1 = MD5(username:realm:password)
        ha1_input = f"{username}:{realm}:{password}"
        ha1 = hashlib.md5(ha1_input.encode()).hexdigest()
        
        # HA2 = MD5(method:uri)
        ha2_input = f"{method}:{uri}"
        ha2 = hashlib.md5(ha2_input.encode()).hexdigest()
        
        # Response = MD5(HA1:nonce:HA2)
        response_input = f"{ha1}:{nonce}:{ha2}"
        response = hashlib.md5(response_input.encode()).hexdigest()
        
        return response
    
    def start(self):
        """Start SIP client and registration loop"""
        if self.running:
            logger.warning("SIP Client already running")
            return
        
        # Check if trunk is enabled
        if not self.config.get("sip.trunk.enabled", False):
            logger.info("SIP trunk disabled in configuration")
            return
        
        if not self.config.get("sip.trunk.register", False):
            logger.info("SIP trunk registration disabled")
            return
        
        try:
            # Create UDP socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.settimeout(5.0)  # 5 second timeout for responses
            
            # Bind to local port so we can receive responses
            # Note: We can't bind to the SIP server's port (5060) if it's already in use
            # Instead, we'll use a different port for the client socket
            client_port = self.config.get("sip.trunk.client_port", 0)  # 0 = auto-assign
            if client_port == 0:
                # Let the OS assign a random port, but we need to update our headers
                self.socket.bind(('', 0))
                # Get the actual port assigned
                self.local_port = self.socket.getsockname()[1]
                logger.info(f"🔌 SIP Client bound to port {self.local_port}")
            else:
                # Use configured port
                self.socket.bind((self.local_ip, client_port))
                self.local_port = client_port
                logger.info(f"🔌 SIP Client bound to {self.local_ip}:{self.local_port}")
            
            self.running = True
            
            # Initialize trunk registration state
            trunk_config = self.config.get("sip.trunk", {})
            self.trunk_registration = TrunkRegistration(
                server=trunk_config.get("sip_server", "10.200.42.121"),
                user_id=trunk_config.get("user_id", "+966114874423"),
                expires=trunk_config.get("expires", 3600)
            )
            
            # Start registration thread
            self.registration_thread = threading.Thread(
                target=self._registration_loop,
                name="SIPClient-Registration",
                daemon=True
            )
            self.registration_thread.start()
            
            logger.info(f"✅ SIP Client started - registering with {self.trunk_registration.server}")
            
        except Exception as e:
            logger.error(f"Failed to start SIP Client: {e}")
            self.running = False
    
    def stop(self):
        """Stop SIP client and unregister"""
        if not self.running:
            return
        
        logger.info("Stopping SIP Client...")
        self.running = False
        
        # Send unregister (expires=0)
        if self.trunk_registration and self.trunk_registration.registered:
            try:
                self._send_register(expires=0)
                logger.info("Sent unregister to trunk")
            except Exception as e:
                logger.warning(f"Failed to unregister: {e}")
        
        # Close socket
        if self.socket:
            self.socket.close()
            self.socket = None
        
        # Wait for thread
        if self.registration_thread:
            self.registration_thread.join(timeout=2.0)
        
        logger.info("SIP Client stopped")
    
    def _registration_loop(self):
        """Main registration loop - maintains registration with provider"""
        logger.info("🔄 Registration loop started")
        
        # Initial registration
        self._perform_registration()
        
        while self.running:
            try:
                # Check if we need to re-register
                if self.trunk_registration and self.trunk_registration.registered:
                    # Re-register 60 seconds before expiry
                    expires = self.trunk_registration.expires
                    reregister_interval = max(expires - 60, expires // 2)
                    
                    time.sleep(reregister_interval)
                    
                    if self.running:  # Check again after sleep
                        logger.info("⏰ Time to re-register with trunk")
                        self._perform_registration()
                else:
                    # Registration failed, retry after 30 seconds
                    time.sleep(30)
                    if self.running:
                        logger.info("🔄 Retrying registration...")
                        self._perform_registration()
                        
            except Exception as e:
                logger.error(f"Error in registration loop: {e}")
                time.sleep(30)
        
        logger.info("Registration loop stopped")
    
    def _perform_registration(self):
        """
        Perform registration with SIP provider.
        Handles 401 Unauthorized and retries with digest auth.
        """
        try:
            # Step 1: Send initial REGISTER (without auth)
            logger.info(f"📤 Sending REGISTER to {self.trunk_registration.server}")
            response = self._send_register()
            
            if not response:
                logger.error("❌ No response to REGISTER")
                self.trunk_registration.registered = False
                self.trunk_registration.error = "Timeout"
                return
            
            # Step 2: Check response
            if response.status_code == 200:
                # Success (unlikely on first try, but handle it)
                logger.info("✅ Registered successfully (no auth required)")
                self._handle_register_success(response)
                
            elif response.status_code == 401:
                # Unauthorized - need to authenticate
                logger.info("🔐 Received 401 Unauthorized - authenticating...")
                self._handle_401_auth(response)
                
            else:
                # Other error
                logger.error(f"❌ Registration failed: {response.status_code} {response.reason}")
                self.trunk_registration.registered = False
                self.trunk_registration.error = f"{response.status_code} {response.reason}"
                
        except Exception as e:
            logger.error(f"❌ Registration error: {e}")
            self.trunk_registration.registered = False
            self.trunk_registration.error = str(e)
    
    def _send_register(self, expires: Optional[int] = None, auth_header: Optional[str] = None) -> Optional[SIPMessage]:
        """
        Send REGISTER request to trunk provider.
        
        Args:
            expires: Registration expiry (default from config)
            auth_header: Authorization header for authenticated request
        
        Returns:
            Parsed SIP response or None on timeout
        """
        if expires is None:
            expires = self.trunk_registration.expires
        
        trunk_config = self.config.get("sip.trunk", {})
        
        sip_server = trunk_config.get("sip_server", "10.200.42.121")
        sip_port = trunk_config.get("sip_port", 5060)
        user_id = trunk_config.get("user_id", "+966114874423")
        domain = trunk_config.get("domain", "fmc.stc.com.sa")
        
        # Build REGISTER message
        request_uri = f"sip:{domain}"
        from_uri = f"sip:{user_id}@{domain}"
        to_uri = from_uri
        contact_uri = f"sip:{user_id}@{self.local_ip}:{self.local_port}"
        
        # Build message
        message = [
            f"REGISTER {request_uri} SIP/2.0",
            f"Via: SIP/2.0/UDP {self.local_ip}:{self.local_port};branch=z9hG4bK{int(time.time())}",
            f"From: <{from_uri}>;tag={int(time.time())}",
            f"To: <{to_uri}>",
            f"Call-ID: {self.call_id}",
            f"CSeq: {self.cseq} REGISTER",
            f"Contact: <{contact_uri}>",
            f"Expires: {expires}",
            "Max-Forwards: 70",
            f"User-Agent: BeautyAI-PABX/1.0",
            "Allow: INVITE, ACK, BYE, CANCEL, OPTIONS, INFO, UPDATE",
        ]
        
        # Add Authorization header if provided
        if auth_header:
            message.insert(-3, auth_header)  # Insert before User-Agent
        
        message.extend(["Content-Length: 0", "", ""])
        
        register_msg = "\r\n".join(message)
        
        # Log the exact packet being sent
        logger.info(f"📦 REGISTER Packet Details:")
        logger.info(f"   Destination: {sip_server}:{sip_port}")
        logger.info(f"   Source: {self.local_ip}:{self.local_port}")
        logger.info(f"   Packet Content:\n{register_msg}")
        
        # Send to server
        try:
            bytes_sent = self.socket.sendto(register_msg.encode(), (sip_server, sip_port))
            logger.info(f"📤 Sent REGISTER ({bytes_sent} bytes, CSeq: {self.cseq}, Expires: {expires})")
            self.cseq += 1
            
            # Wait for response
            data, addr = self.socket.recvfrom(4096)
            response = self.parser.parse(data.decode())
            
            logger.debug(f"📥 Received: {response.status_code} {response.reason}")
            return response
            
        except socket.timeout:
            logger.warning("⏱️ REGISTER timeout - no response from server")
            return None
        except Exception as e:
            logger.error(f"Error sending REGISTER: {e}")
            return None
    
    def _handle_401_auth(self, response: SIPMessage):
        """
        Handle 401 Unauthorized response and retry with authentication.
        
        Args:
            response: 401 response message with WWW-Authenticate header
        """
        # Parse WWW-Authenticate header
        www_auth = response.headers.get("www-authenticate", "")
        
        if not www_auth:
            logger.error("❌ 401 response missing WWW-Authenticate header")
            self.trunk_registration.registered = False
            return
        
        # Extract realm and nonce
        realm = self._extract_auth_param(www_auth, "realm")
        nonce = self._extract_auth_param(www_auth, "nonce")
        
        if not realm or not nonce:
            logger.error(f"❌ Failed to parse WWW-Authenticate: {www_auth}")
            self.trunk_registration.registered = False
            return
        
        logger.info(f"🔐 Authenticating with realm: {realm}")
        
        # Get credentials
        trunk_config = self.config.get("sip.trunk", {})
        auth_id = trunk_config.get("auth_id", "+966114874423@fmc.stc.com.sa")
        auth_password = trunk_config.get("auth_password", "114874423114874423")
        domain = trunk_config.get("domain", "fmc.stc.com.sa")
        
        # Generate digest response
        uri = f"sip:{domain}"
        digest_response = self._generate_nonce_response(
            username=auth_id,
            password=auth_password,
            realm=realm,
            nonce=nonce,
            uri=uri,
            method="REGISTER"
        )
        
        # Build Authorization header
        auth_header = (
            f'Authorization: Digest username="{auth_id}", '
            f'realm="{realm}", '
            f'nonce="{nonce}", '
            f'uri="{uri}", '
            f'response="{digest_response}", '
            f'algorithm=MD5'
        )
        
        # Retry REGISTER with authentication
        logger.info("📤 Sending authenticated REGISTER")
        auth_response = self._send_register(auth_header=auth_header)
        
        if not auth_response:
            logger.error("❌ No response to authenticated REGISTER")
            self.trunk_registration.registered = False
            return
        
        if auth_response.status_code == 200:
            logger.info("✅ Authenticated REGISTER successful!")
            self._handle_register_success(auth_response)
        else:
            logger.error(f"❌ Authenticated REGISTER failed: {auth_response.status_code} {auth_response.reason}")
            self.trunk_registration.registered = False
            self.trunk_registration.error = f"{auth_response.status_code} {auth_response.reason}"
    
    def _handle_register_success(self, response: SIPMessage):
        """
        Handle successful REGISTER response (200 OK).
        
        Args:
            response: 200 OK response message
        """
        # Parse expires from response
        contact_header = response.headers.get("contact", "")
        expires = self._extract_expires(contact_header, response)
        
        # Update registration state
        self.trunk_registration.registered = True
        self.trunk_registration.last_register = datetime.now()
        self.trunk_registration.expires = expires
        self.trunk_registration.error = None
        
        logger.info(f"✅ Successfully registered with {self.trunk_registration.server}")
        logger.info(f"📝 Registration expires in {expires} seconds")
    
    def _extract_auth_param(self, auth_header: str, param: str) -> Optional[str]:
        """Extract parameter from WWW-Authenticate header"""
        try:
            # Find param="value"
            start = auth_header.find(f'{param}="')
            if start == -1:
                return None
            start += len(param) + 2  # Skip param="
            end = auth_header.find('"', start)
            if end == -1:
                return None
            return auth_header[start:end]
        except Exception as e:
            logger.warning(f"Error extracting {param}: {e}")
            return None
    
    def _extract_expires(self, contact_header: str, response: SIPMessage) -> int:
        """Extract expires value from Contact header or Expires header"""
        # Try Contact header first
        if "expires=" in contact_header.lower():
            try:
                start = contact_header.lower().find("expires=") + 8
                end = contact_header.find(";", start)
                if end == -1:
                    end = contact_header.find(">", start)
                if end == -1:
                    end = len(contact_header)
                expires_str = contact_header[start:end].strip()
                return int(expires_str)
            except Exception:
                pass
        
        # Try Expires header
        expires_header = response.headers.get("expires", "")
        if expires_header:
            try:
                return int(expires_header)
            except Exception:
                pass
        
        # Default
        return self.trunk_registration.expires
    
    def get_registration_status(self) -> Dict:
        """
        Get current trunk registration status.
        
        Returns:
            Dictionary with registration details
        """
        if not self.trunk_registration:
            return {
                "enabled": False,
                "registered": False,
                "error": "Trunk not configured"
            }
        
        return {
            "enabled": True,
            "registered": self.trunk_registration.registered,
            "server": self.trunk_registration.server,
            "user_id": self.trunk_registration.user_id,
            "expires": self.trunk_registration.expires,
            "last_register": self.trunk_registration.last_register.isoformat() if self.trunk_registration.last_register else None,
            "error": self.trunk_registration.error
        }
