"""
RTP types and constants
"""

from enum import IntEnum
from typing import Dict


class RTPPayloadType(IntEnum):
    """RTP payload types (codecs)"""
    PCMU = 0        # G.711 μ-law
    GSM = 3         # GSM
    G723 = 4        # G.723
    DVI4_8000 = 5   # DVI4 at 8kHz
    DVI4_16000 = 6  # DVI4 at 16kHz
    LPC = 7         # LPC
    PCMA = 8        # G.711 A-law
    G722 = 9        # G.722
    L16_STEREO = 10 # L16 stereo
    L16_MONO = 11   # L16 mono
    QCELP = 12      # QCELP
    CN = 13         # Comfort noise
    MPA = 14        # MPEG audio
    G728 = 15       # G.728
    DVI4_11025 = 16 # DVI4 at 11.025kHz
    DVI4_22050 = 17 # DVI4 at 22.05kHz
    G729 = 18       # G.729
    DTMF = 101      # DTMF (telephone-event)


# Codec information mapping
CODEC_MAP: Dict[int, Dict[str, any]] = {
    0: {
        'name': 'PCMU',
        'full_name': 'G.711 μ-law',
        'sample_rate': 8000,
        'channels': 1,
        'bits_per_sample': 8,
        'frame_size': 160,  # samples (20ms at 8kHz)
        'bytes_per_frame': 160,
    },
    8: {
        'name': 'PCMA',
        'full_name': 'G.711 A-law',
        'sample_rate': 8000,
        'channels': 1,
        'bits_per_sample': 8,
        'frame_size': 160,
        'bytes_per_frame': 160,
    },
    9: {
        'name': 'G722',
        'full_name': 'G.722',
        'sample_rate': 16000,  # Wideband
        'channels': 1,
        'bits_per_sample': 8,
        'frame_size': 320,  # samples (20ms at 16kHz)
        'bytes_per_frame': 160,  # Compressed to 64kbps
    },
    18: {
        'name': 'G729',
        'full_name': 'G.729',
        'sample_rate': 8000,
        'channels': 1,
        'bits_per_sample': 8,
        'frame_size': 80,  # 10ms frames
        'bytes_per_frame': 10,  # Compressed to 8kbps
    },
    101: {
        'name': 'telephone-event',
        'full_name': 'DTMF',
        'sample_rate': 8000,
        'channels': 1,
        'bits_per_sample': 8,
        'frame_size': 0,
        'bytes_per_frame': 4,  # DTMF event payload
    },
}


# RTP version
RTP_VERSION = 2

# Default packet time (milliseconds)
DEFAULT_PTIME = 20

# RTP header size (bytes)
RTP_HEADER_SIZE = 12
