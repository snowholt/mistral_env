"""
RTP handler service
Manage RTP sessions for active calls
"""

import threading
from typing import Optional, Dict, Callable
from datetime import datetime

from ..core.rtp.stream import RTPStream, RTPStreamManager
from ..modules.audio.loader import AudioLoader
from ..modules.audio.codecs import get_codec
from ..utils.config import Config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class RTPHandler:
    """
    Manage RTP streams for active calls
    """
    
    def __init__(self):
        """Initialize RTP handler"""
        self.config = Config()
        self.audio_config = self.config.get('audio')
        
        # Stream manager
        self.stream_manager = RTPStreamManager()
        
        # Active streams
        self.streams: Dict[str, RTPStream] = {}
        
        # Callbacks
        self.on_audio_received: Optional[Callable] = None
        
        logger.info("RTP handler initialized")
    
    def create_stream(
        self,
        call_id: str,
        local_port: int,
        remote_ip: str,
        remote_port: int,
        payload_type: int = 0
    ) -> Optional[RTPStream]:
        """
        Create RTP stream for call
        
        Args:
            call_id: Call identifier
            local_port: Local RTP port
            remote_ip: Remote IP address
            remote_port: Remote RTP port
            payload_type: RTP payload type (default: 0 = PCMU)
            
        Returns:
            RTPStream object or None if failed
        """
        try:
            # Get codec
            codec = get_codec(payload_type)
            if not codec:
                logger.error(f"Unsupported payload type: {payload_type}")
                return None
            
            # Create stream
            stream = RTPStream(
                local_ip="0.0.0.0",  # Bind to all interfaces
                local_port=local_port,
                remote_ip=remote_ip,
                remote_port=remote_port,
                payload_type=payload_type
            )
            
            # Set callback
            if self.on_audio_received:
                stream.on_audio_received = self.on_audio_received
            
            # Store stream
            self.streams[call_id] = stream
            
            logger.info(f"Created RTP stream for call {call_id} on port {local_port}")
            
            return stream
            
        except Exception as e:
            logger.error(f"Error creating RTP stream: {e}", exc_info=True)
            return None
    
    def start_stream(self, call_id: str) -> bool:
        """
        Start RTP stream
        
        Args:
            call_id: Call identifier
            
        Returns:
            True if started successfully
        """
        if call_id not in self.streams:
            logger.error(f"Stream not found for call {call_id}")
            return False
        
        try:
            stream = self.streams[call_id]
            stream.start()
            
            logger.info(f"Started RTP stream for call {call_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting stream: {e}", exc_info=True)
            return False
    
    def stop_stream(self, call_id: str):
        """
        Stop RTP stream
        
        Args:
            call_id: Call identifier
        """
        if call_id not in self.streams:
            return
        
        try:
            stream = self.streams[call_id]
            stream.stop()
            
            # Remove stream
            del self.streams[call_id]
            
            logger.info(f"Stopped RTP stream for call {call_id}")
            
        except Exception as e:
            logger.error(f"Error stopping stream: {e}", exc_info=True)
    
    def play_audio(
        self,
        call_id: str,
        audio_file: str
    ) -> bool:
        """
        Play audio file on call
        
        Args:
            call_id: Call identifier
            audio_file: Path to audio file
            
        Returns:
            True if playback started successfully
        """
        if call_id not in self.streams:
            logger.error(f"Stream not found for call {call_id}")
            return False
        
        try:
            stream = self.streams[call_id]
            
            # Load audio file
            audio_data, sample_rate = AudioLoader.load(audio_file)
            
            # Send audio to stream
            stream.send_audio(audio_data)
            
            logger.info(f"Playing audio on call {call_id}: {audio_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error playing audio: {e}", exc_info=True)
            return False
    
    def record_audio(
        self,
        call_id: str,
        output_file: str,
        duration: Optional[float] = None
    ) -> bool:
        """
        Record audio from call
        
        Args:
            call_id: Call identifier
            output_file: Output file path
            duration: Recording duration in seconds (None = until stopped)
            
        Returns:
            True if recording started successfully
        """
        if call_id not in self.streams:
            logger.error(f"Stream not found for call {call_id}")
            return False
        
        try:
            stream = self.streams[call_id]
            
            # Start recording in thread
            def record_thread():
                try:
                    # Collect audio data
                    audio_data = []
                    start_time = datetime.now()
                    
                    while stream.running:
                        # Check duration
                        if duration:
                            elapsed = (datetime.now() - start_time).total_seconds()
                            if elapsed >= duration:
                                break
                        
                        # Get audio from stream
                        # Note: This is a simplified implementation
                        # Real implementation would use stream's receive buffer
                        threading.Event().wait(0.1)
                    
                    # Save recording
                    if audio_data:
                        AudioLoader.save_wav(
                            audio_data,
                            output_file,
                            sample_rate=stream.sample_rate
                        )
                        logger.info(f"Saved recording: {output_file}")
                    
                except Exception as e:
                    logger.error(f"Error in recording thread: {e}", exc_info=True)
            
            # Start recording thread
            thread = threading.Thread(target=record_thread, daemon=True)
            thread.start()
            
            logger.info(f"Started recording call {call_id} to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting recording: {e}", exc_info=True)
            return False
    
    def get_stream_stats(self, call_id: str) -> Optional[dict]:
        """
        Get stream statistics
        
        Args:
            call_id: Call identifier
            
        Returns:
            Statistics dictionary or None
        """
        if call_id not in self.streams:
            return None
        
        stream = self.streams[call_id]
        
        return {
            'packets_sent': stream.packets_sent,
            'packets_received': stream.packets_received,
            'bytes_sent': stream.bytes_sent,
            'bytes_received': stream.bytes_received,
            'packet_loss': stream.packet_loss,
            'jitter': stream.jitter
        }
    
    def shutdown(self):
        """Shutdown RTP handler and stop all streams"""
        logger.info("Shutting down RTP handler")
        
        # Stop all streams
        for call_id in list(self.streams.keys()):
            self.stop_stream(call_id)
        
        logger.info("RTP handler shutdown complete")
