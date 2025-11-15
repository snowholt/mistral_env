"""
Call manager service
Coordinate SIP and RTP for call handling
"""

from typing import Optional, Dict, Callable
from datetime import datetime
from dataclasses import dataclass

from .sip_server import SIPServer, CallSession as SIPCallSession
from .rtp_handler import RTPHandler
from ..utils.config import Config
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Call:
    """Complete call information"""
    call_id: str
    from_user: str
    to_user: str
    state: str
    
    # SIP info
    sip_session: SIPCallSession
    
    # RTP info
    local_rtp_port: Optional[int] = None
    remote_rtp_ip: Optional[str] = None
    remote_rtp_port: Optional[int] = None
    
    # Timing
    started_at: datetime = None
    answered_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    
    # Audio recording
    recording_file: Optional[str] = None


class CallManager:
    """
    Manage call lifecycle
    Coordinates SIP signaling and RTP media
    """
    
    def __init__(self):
        """Initialize call manager"""
        self.config = Config()
        
        # Services
        self.sip_server = SIPServer()
        self.rtp_handler = RTPHandler()
        
        # Active calls
        self.calls: Dict[str, Call] = {}
        
        # Port allocation
        rtp_config = self.config.get('rtp')
        self.rtp_port_start = rtp_config.get('port_range', {}).get('start', 10000)
        self.rtp_port_end = rtp_config.get('port_range', {}).get('end', 20000)
        self.next_rtp_port = self.rtp_port_start
        
        # Callbacks
        self.on_call_incoming: Optional[Callable] = None
        self.on_call_answered: Optional[Callable] = None
        self.on_call_ended: Optional[Callable] = None
        self.on_audio_received: Optional[Callable] = None
        
        # Set up SIP callbacks
        self.sip_server.on_invite = self._handle_sip_invite
        self.sip_server.on_bye = self._handle_sip_bye
        
        # Set up RTP callbacks
        self.rtp_handler.on_audio_received = self._handle_audio_received
        
        logger.info("Call manager initialized")
    
    def start(self):
        """Start call manager"""
        try:
            # Start SIP server
            self.sip_server.start()
            
            logger.info("Call manager started")
            
        except Exception as e:
            logger.error(f"Error starting call manager: {e}", exc_info=True)
            raise
    
    def stop(self):
        """Stop call manager"""
        logger.info("Stopping call manager")
        
        # End all active calls
        for call_id in list(self.calls.keys()):
            self.end_call(call_id)
        
        # Stop services
        self.sip_server.stop()
        self.rtp_handler.shutdown()
        
        logger.info("Call manager stopped")
    
    def _handle_sip_invite(self, sip_session: SIPCallSession):
        """
        Handle incoming INVITE
        
        Args:
            sip_session: SIP call session
        """
        try:
            call_id = sip_session.call_id
            
            # Create call object
            call = Call(
                call_id=call_id,
                from_user=sip_session.from_user,
                to_user=sip_session.to_user,
                state="RINGING",
                sip_session=sip_session,
                started_at=datetime.now()
            )
            
            self.calls[call_id] = call
            
            logger.info(f"Incoming call: {call.from_user} -> {call.to_user}")
            
            # Callback
            if self.on_call_incoming:
                self.on_call_incoming(call)
            
            # Auto-answer if configured
            if self.config.get('sip', {}).get('call_handling', {}).get('auto_answer', False):
                self.answer_call(call_id)
            
        except Exception as e:
            logger.error(f"Error handling INVITE: {e}", exc_info=True)
    
    def answer_call(self, call_id: str) -> bool:
        """
        Answer incoming call
        
        Args:
            call_id: Call identifier
            
        Returns:
            True if answered successfully
        """
        if call_id not in self.calls:
            logger.error(f"Call not found: {call_id}")
            return False
        
        try:
            call = self.calls[call_id]
            
            # Allocate RTP port
            local_rtp_port = self._allocate_rtp_port()
            call.local_rtp_port = local_rtp_port
            
            # Get remote RTP info from SIP session
            call.remote_rtp_ip = call.sip_session.remote_ip
            call.remote_rtp_port = call.sip_session.remote_port
            
            # Create RTP stream
            stream = self.rtp_handler.create_stream(
                call_id=call_id,
                local_port=local_rtp_port,
                remote_ip=call.remote_rtp_ip,
                remote_port=call.remote_rtp_port,
                payload_type=int(call.sip_session.codec or '0')
            )
            
            if not stream:
                logger.error("Failed to create RTP stream")
                return False
            
            # Start stream
            self.rtp_handler.start_stream(call_id)
            
            # Update call state
            call.state = "ACTIVE"
            call.answered_at = datetime.now()
            
            logger.info(f"Answered call {call_id}")
            
            # Callback
            if self.on_call_answered:
                self.on_call_answered(call)
            
            # Start recording if configured
            if self.config.get('sip', {}).get('call_handling', {}).get('auto_record', False):
                self.start_recording(call_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error answering call: {e}", exc_info=True)
            return False
    
    def end_call(self, call_id: str):
        """
        End active call
        
        Args:
            call_id: Call identifier
        """
        if call_id not in self.calls:
            return
        
        try:
            call = self.calls[call_id]
            
            # Stop RTP stream
            self.rtp_handler.stop_stream(call_id)
            
            # Update call state
            call.state = "ENDED"
            call.ended_at = datetime.now()
            
            logger.info(f"Ended call {call_id}")
            
            # Callback
            if self.on_call_ended:
                self.on_call_ended(call)
            
            # Remove call
            del self.calls[call_id]
            
        except Exception as e:
            logger.error(f"Error ending call: {e}", exc_info=True)
    
    def _handle_sip_bye(self, sip_session: SIPCallSession):
        """
        Handle BYE request
        
        Args:
            sip_session: SIP call session
        """
        self.end_call(sip_session.call_id)
    
    def play_audio(self, call_id: str, audio_file: str) -> bool:
        """
        Play audio file on call
        
        Args:
            call_id: Call identifier
            audio_file: Path to audio file
            
        Returns:
            True if playback started successfully
        """
        return self.rtp_handler.play_audio(call_id, audio_file)
    
    def start_recording(self, call_id: str) -> bool:
        """
        Start recording call audio
        
        Args:
            call_id: Call identifier
            
        Returns:
            True if recording started successfully
        """
        if call_id not in self.calls:
            logger.error(f"Call not found: {call_id}")
            return False
        
        try:
            call = self.calls[call_id]
            
            # Generate recording filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            recording_file = f"recordings/call_{call_id}_{timestamp}.wav"
            
            call.recording_file = recording_file
            
            # Start recording
            return self.rtp_handler.record_audio(call_id, recording_file)
            
        except Exception as e:
            logger.error(f"Error starting recording: {e}", exc_info=True)
            return False
    
    def _handle_audio_received(self, call_id: str, audio_data: bytes):
        """
        Handle received audio data
        
        Args:
            call_id: Call identifier
            audio_data: Audio data
        """
        if self.on_audio_received:
            self.on_audio_received(call_id, audio_data)
    
    def get_active_calls(self) -> list:
        """
        Get list of active calls
        
        Returns:
            List of Call objects
        """
        return list(self.calls.values())
    
    def get_call(self, call_id: str) -> Optional[Call]:
        """
        Get call by ID
        
        Args:
            call_id: Call identifier
            
        Returns:
            Call object or None
        """
        return self.calls.get(call_id)
    
    def get_call_stats(self, call_id: str) -> Optional[dict]:
        """
        Get call statistics
        
        Args:
            call_id: Call identifier
            
        Returns:
            Statistics dictionary or None
        """
        if call_id not in self.calls:
            return None
        
        call = self.calls[call_id]
        rtp_stats = self.rtp_handler.get_stream_stats(call_id)
        
        # Calculate duration
        duration = 0.0
        if call.answered_at:
            end_time = call.ended_at or datetime.now()
            duration = (end_time - call.answered_at).total_seconds()
        
        return {
            'call_id': call.call_id,
            'from_user': call.from_user,
            'to_user': call.to_user,
            'state': call.state,
            'duration': duration,
            'rtp_stats': rtp_stats
        }
    
    def _allocate_rtp_port(self) -> int:
        """
        Allocate next available RTP port
        
        Returns:
            Port number
        """
        port = self.next_rtp_port
        self.next_rtp_port += 2  # RTP uses even ports, RTCP uses odd
        
        # Wrap around if we reach the end
        if self.next_rtp_port >= self.rtp_port_end:
            self.next_rtp_port = self.rtp_port_start
        
        return port
