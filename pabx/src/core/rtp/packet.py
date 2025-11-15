"""
RTP packet handling
Parsing and creation of RTP packets (RFC 3550)
"""

import struct
from dataclasses import dataclass
from typing import Optional, Tuple

from .types import RTP_VERSION, RTP_HEADER_SIZE


@dataclass
class RTPHeader:
    """RTP header structure (RFC 3550)"""
    version: int = RTP_VERSION
    padding: bool = False
    extension: bool = False
    csrc_count: int = 0
    marker: bool = False
    payload_type: int = 0
    sequence_number: int = 0
    timestamp: int = 0
    ssrc: int = 0
    csrc_list: list = None
    
    def __post_init__(self):
        if self.csrc_list is None:
            self.csrc_list = []


@dataclass
class RTPPacket:
    """Complete RTP packet"""
    header: RTPHeader
    payload: bytes
    
    def to_bytes(self) -> bytes:
        """Serialize packet to bytes"""
        return create_rtp_packet(
            payload_type=self.header.payload_type,
            sequence_number=self.header.sequence_number,
            timestamp=self.header.timestamp,
            ssrc=self.header.ssrc,
            payload=self.payload,
            marker=self.header.marker,
            csrc_list=self.header.csrc_list
        )
    
    @property
    def size(self) -> int:
        """Total packet size in bytes"""
        return RTP_HEADER_SIZE + (len(self.header.csrc_list) * 4) + len(self.payload)


def parse_rtp_packet(data: bytes) -> Optional[RTPPacket]:
    """
    Parse RTP packet from bytes
    
    Args:
        data: Raw packet bytes
        
    Returns:
        RTPPacket object or None if parsing fails
    """
    if len(data) < RTP_HEADER_SIZE:
        return None
    
    # Parse fixed header (12 bytes)
    try:
        # First byte: V(2), P(1), X(1), CC(4)
        byte0 = data[0]
        version = (byte0 >> 6) & 0x03
        padding = bool((byte0 >> 5) & 0x01)
        extension = bool((byte0 >> 4) & 0x01)
        csrc_count = byte0 & 0x0F
        
        # Second byte: M(1), PT(7)
        byte1 = data[1]
        marker = bool((byte1 >> 7) & 0x01)
        payload_type = byte1 & 0x7F
        
        # Sequence number (2 bytes)
        sequence_number = struct.unpack('!H', data[2:4])[0]
        
        # Timestamp (4 bytes)
        timestamp = struct.unpack('!I', data[4:8])[0]
        
        # SSRC (4 bytes)
        ssrc = struct.unpack('!I', data[8:12])[0]
        
        # CSRC list (4 bytes each)
        csrc_list = []
        offset = 12
        for _ in range(csrc_count):
            if offset + 4 > len(data):
                return None
            csrc = struct.unpack('!I', data[offset:offset+4])[0]
            csrc_list.append(csrc)
            offset += 4
        
        # Header extension (if present)
        if extension:
            if offset + 4 > len(data):
                return None
            # Skip extension (profile + length)
            ext_length = struct.unpack('!H', data[offset+2:offset+4])[0]
            offset += 4 + (ext_length * 4)
        
        # Payload
        payload = data[offset:]
        
        # Remove padding if present
        if padding and len(payload) > 0:
            padding_length = payload[-1]
            if padding_length <= len(payload):
                payload = payload[:-padding_length]
        
        # Create header object
        header = RTPHeader(
            version=version,
            padding=padding,
            extension=extension,
            csrc_count=csrc_count,
            marker=marker,
            payload_type=payload_type,
            sequence_number=sequence_number,
            timestamp=timestamp,
            ssrc=ssrc,
            csrc_list=csrc_list
        )
        
        return RTPPacket(header=header, payload=payload)
        
    except (struct.error, IndexError):
        return None


def create_rtp_packet(
    payload_type: int,
    sequence_number: int,
    timestamp: int,
    ssrc: int,
    payload: bytes,
    marker: bool = False,
    csrc_list: Optional[list] = None
) -> bytes:
    """
    Create RTP packet
    
    Args:
        payload_type: Codec payload type
        sequence_number: Sequence number
        timestamp: RTP timestamp
        ssrc: Synchronization source identifier
        payload: Audio data payload
        marker: Marker bit
        csrc_list: Contributing source list (optional)
        
    Returns:
        Raw packet bytes
    """
    if csrc_list is None:
        csrc_list = []
    
    csrc_count = len(csrc_list)
    if csrc_count > 15:
        raise ValueError("CSRC count cannot exceed 15")
    
    # Build header
    header = bytearray()
    
    # Byte 0: V(2), P(0), X(0), CC(4)
    byte0 = (RTP_VERSION << 6) | csrc_count
    header.append(byte0)
    
    # Byte 1: M(1), PT(7)
    byte1 = (int(marker) << 7) | (payload_type & 0x7F)
    header.append(byte1)
    
    # Sequence number (2 bytes)
    header.extend(struct.pack('!H', sequence_number & 0xFFFF))
    
    # Timestamp (4 bytes)
    header.extend(struct.pack('!I', timestamp & 0xFFFFFFFF))
    
    # SSRC (4 bytes)
    header.extend(struct.pack('!I', ssrc & 0xFFFFFFFF))
    
    # CSRC list
    for csrc in csrc_list:
        header.extend(struct.pack('!I', csrc & 0xFFFFFFFF))
    
    # Combine header and payload
    return bytes(header) + payload


def calculate_timestamp_increment(sample_rate: int, ptime_ms: int = 20) -> int:
    """
    Calculate RTP timestamp increment for given packet time
    
    Args:
        sample_rate: Audio sample rate (Hz)
        ptime_ms: Packet time in milliseconds
        
    Returns:
        Timestamp increment value
    """
    return (sample_rate * ptime_ms) // 1000


def detect_packet_loss(seq_current: int, seq_previous: int) -> int:
    """
    Detect packet loss based on sequence numbers
    
    Args:
        seq_current: Current sequence number
        seq_previous: Previous sequence number
        
    Returns:
        Number of lost packets
    """
    # Handle sequence number wrap-around (65536)
    expected = (seq_previous + 1) & 0xFFFF
    
    if seq_current == expected:
        return 0
    
    # Calculate difference considering wrap-around
    if seq_current > expected:
        return seq_current - expected
    else:
        return (0xFFFF - expected) + seq_current + 1
