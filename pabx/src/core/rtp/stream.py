"""
RTP stream management with PyAudio integration
Handles real-time audio streaming via RTP
"""

import socket
import threading
import time
from typing import Optional, Callable, Dict
from queue import Queue, Empty

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

from .packet import RTPPacket, RTPHeader, parse_rtp_packet, create_rtp_packet, detect_packet_loss
from .types import CODEC_MAP, DEFAULT_PTIME
from ..sip.parser import parse_sdp


class RTPStream:
    """
    RTP stream for sending/receiving audio
    """
    
    def __init__(
        self,
        local_ip: str,
        local_port: int,
        remote_ip: str,
        remote_port: int,
        payload_type: int = 0,
        ssrc: Optional[int] = None
    ):
        """
        Initialize RTP stream
        
        Args:
            local_ip: Local IP address
            local_port: Local port for RTP
            remote_ip: Remote IP address
            remote_port: Remote port for RTP
            payload_type: Codec payload type
            ssrc: Synchronization source ID (auto-generated if None)
        """
        self.local_ip = local_ip
        self.local_port = local_port
        self.remote_ip = remote_ip
        self.remote_port = remote_port
        self.payload_type = payload_type
        self.ssrc = ssrc if ssrc is not None else self._generate_ssrc()
        
        # Socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((local_ip, local_port))
        self.socket.settimeout(0.1)  # Non-blocking with timeout
        
        # Stream state
        self.running = False
        self.sequence_number = 0
        self.timestamp = 0
        self.last_received_seq = None
        self.packets_sent = 0
        self.packets_received = 0
        self.packets_lost = 0
        
        # Codec info
        self.codec_info = CODEC_MAP.get(payload_type, {})
        self.sample_rate = self.codec_info.get('sample_rate', 8000)
        self.frame_size = self.codec_info.get('frame_size', 160)
        self.bytes_per_frame = self.codec_info.get('bytes_per_frame', 160)
        
        # Queues for audio data
        self.send_queue = Queue(maxsize=100)
        self.receive_queue = Queue(maxsize=100)
        
        # Threads
        self.send_thread = None
        self.receive_thread = None
        
        # Callbacks
        self.on_packet_received: Optional[Callable[[RTPPacket], None]] = None
    
    def start(self):
        """Start RTP stream threads"""
        if self.running:
            return
        
        self.running = True
        
        # Start sender thread
        self.send_thread = threading.Thread(target=self._send_loop, daemon=True)
        self.send_thread.start()
        
        # Start receiver thread
        self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.receive_thread.start()
    
    def stop(self):
        """Stop RTP stream"""
        self.running = False
        
        if self.send_thread:
            self.send_thread.join(timeout=1.0)
        if self.receive_thread:
            self.receive_thread.join(timeout=1.0)
        
        self.socket.close()
    
    def send_audio(self, audio_data: bytes, marker: bool = False):
        """
        Queue audio data for sending
        
        Args:
            audio_data: Raw audio bytes
            marker: Marker bit (e.g., start of talk spurt)
        """
        try:
            self.send_queue.put((audio_data, marker), block=False)
        except:
            pass  # Queue full, drop packet
    
    def receive_audio(self, timeout: float = 0.1) -> Optional[bytes]:
        """
        Get received audio data
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Audio data or None if no data available
        """
        try:
            return self.receive_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def _send_loop(self):
        """Sender thread loop"""
        while self.running:
            try:
                # Get audio from queue
                audio_data, marker = self.send_queue.get(timeout=0.02)
                
                # Create RTP packet
                packet_bytes = create_rtp_packet(
                    payload_type=self.payload_type,
                    sequence_number=self.sequence_number,
                    timestamp=self.timestamp,
                    ssrc=self.ssrc,
                    payload=audio_data,
                    marker=marker
                )
                
                # Send packet
                self.socket.sendto(packet_bytes, (self.remote_ip, self.remote_port))
                
                # Update counters
                self.sequence_number = (self.sequence_number + 1) & 0xFFFF
                self.timestamp = (self.timestamp + self.frame_size) & 0xFFFFFFFF
                self.packets_sent += 1
                
            except Empty:
                continue
            except Exception as e:
                if self.running:
                    print(f"Error in RTP send loop: {e}")
    
    def _receive_loop(self):
        """Receiver thread loop"""
        while self.running:
            try:
                # Receive packet
                data, addr = self.socket.recvfrom(2048)
                
                # Parse RTP packet
                packet = parse_rtp_packet(data)
                if not packet:
                    continue
                
                # Detect packet loss
                if self.last_received_seq is not None:
                    lost = detect_packet_loss(
                        packet.header.sequence_number,
                        self.last_received_seq
                    )
                    self.packets_lost += lost
                
                self.last_received_seq = packet.header.sequence_number
                self.packets_received += 1
                
                # Queue payload for playback
                try:
                    self.receive_queue.put(packet.payload, block=False)
                except:
                    pass  # Queue full, drop packet
                
                # Call callback if registered
                if self.on_packet_received:
                    self.on_packet_received(packet)
                    
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"Error in RTP receive loop: {e}")
    
    def get_statistics(self) -> Dict:
        """Get stream statistics"""
        return {
            'packets_sent': self.packets_sent,
            'packets_received': self.packets_received,
            'packets_lost': self.packets_lost,
            'loss_rate': (self.packets_lost / max(1, self.packets_received)) * 100,
            'sequence_number': self.sequence_number,
            'timestamp': self.timestamp,
        }
    
    @staticmethod
    def _generate_ssrc() -> int:
        """Generate random SSRC"""
        import random
        return random.randint(0, 0xFFFFFFFF)


class RTPStreamManager:
    """
    Manages RTP streams with PyAudio integration
    Handles audio I/O for real-time communication
    """
    
    def __init__(self):
        """Initialize RTP stream manager"""
        if not PYAUDIO_AVAILABLE:
            raise ImportError("PyAudio not available - install with: pip install pyaudio")
        
        self.pyaudio = pyaudio.PyAudio()
        self.streams: Dict[str, RTPStream] = {}
        self.audio_streams: Dict[str, any] = {}  # PyAudio streams
        self.active = False
    
    def create_stream(
        self,
        stream_id: str,
        local_ip: str,
        local_port: int,
        remote_ip: str,
        remote_port: int,
        payload_type: int = 0,
        enable_audio_output: bool = True,
        enable_audio_input: bool = False
    ) -> RTPStream:
        """
        Create and register RTP stream with audio I/O
        
        Args:
            stream_id: Unique stream identifier
            local_ip: Local IP address
            local_port: Local RTP port
            remote_ip: Remote IP address
            remote_port: Remote RTP port
            payload_type: Codec payload type
            enable_audio_output: Enable speaker output
            enable_audio_input: Enable microphone input
            
        Returns:
            RTPStream object
        """
        # Create RTP stream
        rtp_stream = RTPStream(
            local_ip=local_ip,
            local_port=local_port,
            remote_ip=remote_ip,
            remote_port=remote_port,
            payload_type=payload_type
        )
        
        self.streams[stream_id] = rtp_stream
        
        # Create PyAudio streams if needed
        codec_info = CODEC_MAP.get(payload_type, {})
        sample_rate = codec_info.get('sample_rate', 8000)
        channels = codec_info.get('channels', 1)
        
        audio_config = {
            'output': None,
            'input': None
        }
        
        if enable_audio_output:
            # Output stream (speaker)
            try:
                output_stream = self.pyaudio.open(
                    format=pyaudio.paInt16,
                    channels=channels,
                    rate=sample_rate,
                    output=True,
                    frames_per_buffer=codec_info.get('frame_size', 160)
                )
                audio_config['output'] = output_stream
            except Exception as e:
                print(f"Warning: Could not open audio output: {e}")
        
        if enable_audio_input:
            # Input stream (microphone)
            try:
                input_stream = self.pyaudio.open(
                    format=pyaudio.paInt16,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=codec_info.get('frame_size', 160)
                )
                audio_config['input'] = input_stream
            except Exception as e:
                print(f"Warning: Could not open audio input: {e}")
        
        self.audio_streams[stream_id] = audio_config
        
        # Start RTP stream
        rtp_stream.start()
        
        return rtp_stream
    
    def get_stream(self, stream_id: str) -> Optional[RTPStream]:
        """Get RTP stream by ID"""
        return self.streams.get(stream_id)
    
    def close_stream(self, stream_id: str):
        """Close and remove RTP stream"""
        if stream_id in self.streams:
            self.streams[stream_id].stop()
            del self.streams[stream_id]
        
        if stream_id in self.audio_streams:
            audio_config = self.audio_streams[stream_id]
            if audio_config['output']:
                audio_config['output'].close()
            if audio_config['input']:
                audio_config['input'].close()
            del self.audio_streams[stream_id]
    
    def close_all(self):
        """Close all streams"""
        for stream_id in list(self.streams.keys()):
            self.close_stream(stream_id)
        
        self.pyaudio.terminate()
    
    def __del__(self):
        """Cleanup on deletion"""
        self.close_all()
