"""
BeautyAI Utilities Package

This package provides utility modules and helper functions for the BeautyAI inference framework.
"""

from .webm_decoder import (
    WebMDecoder,
    WebMDecodingMode,
    WebMDecodingError,
    create_realtime_decoder,
    create_batch_decoder
)

__all__ = [
    'WebMDecoder',
    'WebMDecodingMode', 
    'WebMDecodingError',
    'create_realtime_decoder',
    'create_batch_decoder'
]

from .webm_decoder import WebMDecoder, WebMDecodingMode, WebMDecodingError

__all__ = [
    "WebMDecoder",
    "WebMDecodingMode", 
    "WebMDecodingError"
]

__version__ = "1.0.0"
