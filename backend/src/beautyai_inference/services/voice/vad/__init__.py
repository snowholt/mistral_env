"""
Voice Activity Detection (VAD) Module for BeautyAI Voice Services

This module provides VAD services for different voice input modalities:
- WebRTC VAD service: Dual VAD with browser hints + Silero confirmation
- Existing VAD service: For WebSocket/PCM audio streams

Author: BeautyAI Framework
Date: 2025-10-15
"""

from .webrtc_vad_service import (
    WebRTCVADService,
    WebRTCVADConfig,
    VADState,
    create_webrtc_vad_service
)

__all__ = [
    'WebRTCVADService',
    'WebRTCVADConfig',
    'VADState',
    'create_webrtc_vad_service'
]
