"""
Syslog receiver for HT813 device logs
Receives and parses syslog messages from HT813
"""

import socket
import threading
from typing import Optional, Callable
from datetime import datetime
from dataclasses import dataclass

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SyslogMessage:
    """Parsed syslog message"""
    timestamp: datetime
    facility: int
    severity: int
    hostname: str
    message: str
    raw_message: str


class SyslogReceiver:
    """
    UDP Syslog receiver for HT813 device logs
    """
    
    def __init__(
        self,
        host: str = '0.0.0.0',
        port: int = 514,
        buffer_size: int = 8192
    ):
        """
        Initialize syslog receiver
        
        Args:
            host: Bind address
            port: Syslog port (default 514)
            buffer_size: UDP buffer size
        """
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        
        self.socket: Optional[socket.socket] = None
        self.thread: Optional[threading.Thread] = None
        self.running = False
        
        # Callbacks
        self.on_message: Optional[Callable[[SyslogMessage], None]] = None
        
        # Statistics
        self.messages_received = 0
        self.start_time: Optional[datetime] = None
        
        logger.info(f"Syslog receiver initialized on {host}:{port}")
    
    def start(self):
        """Start the syslog receiver"""
        if self.running:
            logger.warning("Syslog receiver already running")
            return
        
        try:
            # Create UDP socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            
            # Start receive thread
            self.running = True
            self.start_time = datetime.now()
            self.thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.thread.start()
            
            logger.info(f"Syslog receiver started on {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Error starting syslog receiver: {e}", exc_info=True)
            self.running = False
            raise
    
    def stop(self):
        """Stop the syslog receiver"""
        if not self.running:
            return
        
        self.running = False
        
        if self.socket:
            self.socket.close()
        
        if self.thread:
            self.thread.join(timeout=2)
        
        logger.info("Syslog receiver stopped")
    
    def _receive_loop(self):
        """Main receive loop"""
        logger.info("Syslog receiver loop started")
        
        while self.running:
            try:
                # Receive message
                data, addr = self.socket.recvfrom(self.buffer_size)
                message = data.decode('utf-8', errors='ignore')
                
                # Parse and process
                parsed = self._parse_message(message, addr)
                if parsed:
                    self.messages_received += 1
                    
                    # Log interesting messages
                    if 'SIP' in parsed.message or 'REGISTER' in parsed.message:
                        logger.info(f"HT813 Log: {parsed.message}")
                    
                    # Call callback
                    if self.on_message:
                        try:
                            self.on_message(parsed)
                        except Exception as e:
                            logger.error(f"Error in syslog callback: {e}")
                
            except Exception as e:
                if self.running:
                    logger.error(f"Error receiving syslog message: {e}")
    
    def _parse_message(self, message: str, addr: tuple) -> Optional[SyslogMessage]:
        """
        Parse syslog message
        
        Args:
            message: Raw syslog message
            addr: Source address tuple
            
        Returns:
            Parsed SyslogMessage or None
        """
        try:
            # Simple RFC 3164 parsing
            # Format: <PRI>TIMESTAMP HOSTNAME MESSAGE
            
            # Extract priority
            if message.startswith('<'):
                pri_end = message.find('>')
                if pri_end > 0:
                    pri = int(message[1:pri_end])
                    facility = pri >> 3
                    severity = pri & 0x07
                    message = message[pri_end + 1:]
                else:
                    facility = 0
                    severity = 6  # INFO
            else:
                facility = 0
                severity = 6
            
            # Extract hostname (source IP)
            hostname = addr[0]
            
            return SyslogMessage(
                timestamp=datetime.now(),
                facility=facility,
                severity=severity,
                hostname=hostname,
                message=message.strip(),
                raw_message=message
            )
            
        except Exception as e:
            logger.error(f"Error parsing syslog message: {e}")
            return None
    
    def get_statistics(self) -> dict:
        """
        Get receiver statistics
        
        Returns:
            Statistics dictionary
        """
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            rate = self.messages_received / duration if duration > 0 else 0
        else:
            duration = 0
            rate = 0
        
        return {
            'running': self.running,
            'duration_seconds': duration,
            'messages_received': self.messages_received,
            'message_rate': rate,
        }
