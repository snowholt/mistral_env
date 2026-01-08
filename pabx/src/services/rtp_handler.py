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
            
            # Load audio file (resampled to 8kHz for telephony)
            audio_data, sample_rate = AudioLoader.load(
                audio_file,
                target_sample_rate=8000,
                target_channels=1
            )
            
            logger.info(f"Playing audio on call {call_id}: {audio_file}")
            logger.info(f"Audio loaded: {len(audio_data)} samples at {sample_rate}Hz")
            
            # Get codec for encoding (payload type from stream)
            codec = get_codec(stream.payload_type)
            if not codec:
                logger.error(f"Unsupported codec payload type: {stream.payload_type}")
                return False
            
            logger.info(f"Encoding audio with codec: {codec.name}")
            
            # Encode PCM to codec format (G.711 µ-law/A-law)
            encoded_audio = codec.encode(audio_data)
            
            logger.info(f"Encoded audio: {len(encoded_audio)} bytes (from {len(audio_data)} samples)")
            
            # Start playback thread to chunk and send audio
            def _playback_thread():
                try:
                    # Chunk size: 160 samples = 160 bytes for G.711 = 20ms @ 8kHz
                    # Note: G.711 is 8-bit per sample, not 16-bit!
                    chunk_size = 160  # bytes (160 samples * 1 byte/sample for G.711)
                    total_chunks = (len(encoded_audio) + chunk_size - 1) // chunk_size
                    
                    logger.info(f"Sending {total_chunks} audio chunks (160 bytes each, 20ms)...")
                    
                    for i in range(0, len(encoded_audio), chunk_size):
                        if not stream.running:
                            break
                        
                        chunk = encoded_audio[i:i+chunk_size]
                        
                        # Pad last chunk if needed
                        if len(chunk) < chunk_size:
                            chunk = chunk + b'\x00' * (chunk_size - len(chunk))
                        
                        # Send chunk with marker on first packet
                        stream.send_audio(chunk, marker=(i == 0))
                        
                        # Wait 20ms between packets (ptime)
                        threading.Event().wait(0.020)
                    
                    logger.info(f"Audio playback complete for call {call_id}")
                    
                except Exception as e:
                    logger.error(f"Error in playback thread: {e}", exc_info=True)
            
            # Start playback in background thread
            playback_thread = threading.Thread(target=_playback_thread, daemon=True)
            playback_thread.start()
            
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
                    # Collect audio data from receive queue
                    audio_data = []
                    start_time = datetime.now()
                    
                    logger.info(f"Recording thread started for call {call_id}")
                    
                    while stream.running:
                        # Check duration
                        if duration:
                            elapsed = (datetime.now() - start_time).total_seconds()
                            if elapsed >= duration:
                                logger.info(f"Recording duration {duration}s reached")
                                break
                        
                        # Get audio from stream's receive queue
                        try:
                            audio_chunk = stream.receive_audio(timeout=0.1)
                            if audio_chunk:
                                audio_data.append(audio_chunk)
                        except Exception as e:
                            # Timeout or queue empty, continue
                            pass
                    
                    # Save recording if we got any audio
                    if audio_data:
                        # Concatenate all audio chunks
                        full_audio = b''.join(audio_data)
                        logger.info(f"Saving {len(full_audio)} bytes of audio to {output_file}")
                        
                        AudioLoader.save_wav(
                            full_audio,
                            output_file,
                            sample_rate=stream.sample_rate
                        )
                        logger.info(f"Saved recording: {output_file} ({len(audio_data)} chunks, {len(full_audio)} bytes)")
                    else:
                        logger.warning(f"No audio data received for call {call_id}, recording not saved")
                    
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
        stats = stream.get_statistics()
        
        return {
            'packets_sent': stats['packets_sent'],
            'packets_received': stats['packets_received'],
            'packets_lost': stats['packets_lost'],
            'bytes_sent': stream.bytes_sent,
            'bytes_received': stream.bytes_received,
            'packet_loss_rate': stats['loss_rate'],
            'jitter': stream.jitter,
            'sequence_number': stats['sequence_number'],
            'timestamp': stats['timestamp']
        }
    
    def shutdown(self):
        """Shutdown RTP handler and stop all streams"""
        logger.info("Shutting down RTP handler")
        
        # Stop all streams
        for call_id in list(self.streams.keys()):
            self.stop_stream(call_id)
        
        logger.info("RTP handler shutdown complete")
