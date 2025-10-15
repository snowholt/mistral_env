"""
WebRTC Voice Service Adapter for BeautyAI

This module provides the integration layer between WebRTC components and SimpleVoiceService:
- Orchestrates audio processor, VAD, and buffer manager
- Automatically injects '/no_think ' prefix for faster LLM responses
- Wires complete audio pipeline: RTP → PCM → VAD → Buffer → STT → LLM → TTS
- Provides session lifecycle management for WebRTC voice calls

Architecture:
  MediaStreamTrack → AudioProcessor → VAD → BufferManager → VoiceService
                                                                ↓
                                                          STT (Whisper)
                                                                ↓
                                                        LLM ("/no_think " auto)
                                                                ↓
                                                          TTS (Edge TTS)

Author: BeautyAI Framework
Date: 2025-10-15
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass

try:
    from aiortc import MediaStreamTrack
    AIORTC_AVAILABLE = True
except ImportError:
    AIORTC_AVAILABLE = False
    MediaStreamTrack = object

from ..webrtc_audio_processor import (
    WebRTCAudioProcessor,
    AudioProcessingConfig,
    create_audio_processor
)
from .vad.webrtc_vad_service import (
    WebRTCVADService,
    WebRTCVADConfig,
    VADState,
    create_webrtc_vad_service
)
from ...core.webrtc_buffer_manager import (
    WebRTCBufferManager,
    BufferConfig,
    create_buffer_manager
)

logger = logging.getLogger(__name__)


@dataclass
class WebRTCVoiceConfig:
    """Complete configuration for WebRTC voice pipeline."""
    
    # Audio processing
    audio_config: Optional[AudioProcessingConfig] = None
    
    # VAD configuration
    vad_config: Optional[WebRTCVADConfig] = None
    
    # Buffer configuration
    buffer_config: Optional[BufferConfig] = None
    
    # LLM optimization
    auto_inject_no_think: bool = True  # Automatically add "/no_think " prefix
    no_think_prefix: str = "/no_think "  # Prefix to inject
    
    # Voice settings
    default_language: str = "en"  # Default language if not specified
    default_gender: str = "female"  # Default voice gender


class WebRTCVoiceServiceAdapter:
    """
    Complete WebRTC voice pipeline adapter integrating all components.
    
    This adapter orchestrates the complete voice flow:
    1. Audio Processor: MediaStreamTrack → PCM conversion
    2. VAD Service: Dual VAD for speech detection
    3. Buffer Manager: Pre-roll/post-roll buffering
    4. Voice Service: STT → LLM (with /no_think) → TTS
    
    Features:
    - Automatic /no_think prefix injection for faster responses
    - Language-specific VAD thresholds
    - 10-second utterance limit enforcement
    - Complete session lifecycle management
    - Metrics and monitoring
    
    Usage:
        adapter = WebRTCVoiceServiceAdapter(
            peer_id="peer123",
            language="ar"
        )
        await adapter.initialize()
        await adapter.start_voice_session(audio_track)
    """
    
    def __init__(
        self,
        peer_id: str,
        session_id: str,
        language: str = "en",
        config: Optional[WebRTCVoiceConfig] = None,
        voice_service = None,  # SimpleVoiceService instance
        on_transcription: Optional[Callable[[str, str], None]] = None,
        on_llm_response: Optional[Callable[[str, str], None]] = None,
        on_tts_audio: Optional[Callable[[str, bytes], None]] = None
    ):
        """
        Initialize WebRTC voice service adapter.
        
        Args:
            peer_id: WebRTC peer identifier
            session_id: Voice session identifier
            language: Language code (ar, en)
            config: Complete pipeline configuration
            voice_service: SimpleVoiceService instance
            on_transcription: Callback when transcription ready
            on_llm_response: Callback when LLM response ready
            on_tts_audio: Callback when TTS audio ready
        """
        self.peer_id = peer_id
        self.session_id = session_id
        self.language = language.lower()
        self.config = config or WebRTCVoiceConfig()
        self.voice_service = voice_service
        self.logger = logging.getLogger(__name__)
        
        # Callbacks
        self._on_transcription = on_transcription
        self._on_llm_response = on_llm_response
        self._on_tts_audio = on_tts_audio
        
        # Pipeline components (initialized in initialize())
        self.audio_processor: Optional[WebRTCAudioProcessor] = None
        self.vad_service: Optional[WebRTCVADService] = None
        self.buffer_manager: Optional[WebRTCBufferManager] = None
        
        # State
        self.is_initialized = False
        self.is_active = False
        self.session_start_time: Optional[float] = None
        
        # Metrics
        self.utterances_processed = 0
        self.total_processing_time = 0.0
        
        self.logger.info(
            f"WebRTC voice adapter created for peer {peer_id}, "
            f"session {session_id}, language {language}"
        )
    
    async def initialize(self) -> bool:
        """
        Initialize all pipeline components.
        
        Returns:
            bool: True if initialization successful
        """
        if self.is_initialized:
            return True
        
        try:
            self.logger.info(f"Initializing WebRTC voice pipeline for {self.peer_id}")
            
            # Initialize audio processor
            self.audio_processor = create_audio_processor(
                peer_id=self.peer_id,
                config=self.config.audio_config,
                on_audio_chunk=self._on_audio_chunk_received,
                on_utterance_limit_exceeded=self._on_utterance_limit,
                on_processing_error=self._on_processing_error
            )
            
            # Initialize VAD service
            self.vad_service = create_webrtc_vad_service(
                peer_id=self.peer_id,
                language=self.language,
                config=self.config.vad_config,
                on_voice_start=self._on_voice_start,
                on_voice_end=self._on_voice_end,
                on_vad_state_change=self._on_vad_state_change
            )
            
            if not await self.vad_service.initialize():
                self.logger.error("Failed to initialize VAD service")
                return False
            
            # Initialize buffer manager
            self.buffer_manager = create_buffer_manager(
                peer_id=self.peer_id,
                config=self.config.buffer_config,
                on_segment_ready=self._on_segment_ready,
                on_buffer_overflow=self._on_buffer_overflow
            )
            
            self.is_initialized = True
            self.logger.info(f"WebRTC voice pipeline initialized for {self.peer_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize voice pipeline: {e}")
            return False
    
    async def start_voice_session(self, audio_track: MediaStreamTrack) -> bool:
        """
        Start processing voice session from audio track.
        
        Args:
            audio_track: WebRTC MediaStreamTrack for audio
            
        Returns:
            bool: True if session started successfully
        """
        if not self.is_initialized:
            self.logger.error("Cannot start session: pipeline not initialized")
            return False
        
        if self.is_active:
            self.logger.warning(f"Voice session already active for {self.peer_id}")
            return False
        
        try:
            self.logger.info(f"Starting voice session for {self.peer_id}")
            
            # Start audio processor
            if not await self.audio_processor.start_processing(audio_track):
                self.logger.error("Failed to start audio processor")
                return False
            
            self.is_active = True
            self.session_start_time = time.time()
            
            self.logger.info(f"Voice session active for {self.peer_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start voice session: {e}")
            return False
    
    async def stop_voice_session(self):
        """Stop voice session and cleanup."""
        if not self.is_active:
            return
        
        self.logger.info(f"Stopping voice session for {self.peer_id}")
        
        try:
            # Stop audio processor
            if self.audio_processor:
                await self.audio_processor.stop_processing()
            
            # Reset components
            if self.vad_service:
                self.vad_service.reset()
            
            if self.buffer_manager:
                self.buffer_manager.reset()
            
            self.is_active = False
            
            # Log session metrics
            if self.session_start_time:
                session_duration = time.time() - self.session_start_time
                self.logger.info(
                    f"Voice session stopped for {self.peer_id}: "
                    f"duration={session_duration:.2f}s, "
                    f"utterances={self.utterances_processed}"
                )
            
        except Exception as e:
            self.logger.error(f"Error stopping voice session: {e}")
    
    async def _on_audio_chunk_received(self, chunk: bytes, metadata: Dict[str, Any]):
        """
        Callback when audio chunk received from processor.
        
        Feeds chunk to VAD for speech detection, then to buffer manager.
        """
        try:
            # Process with VAD
            vad_result = await self.vad_service.process_audio_chunk(chunk, metadata)
            
            if not vad_result["success"]:
                return
            
            # Feed to buffer manager with VAD state
            await self.buffer_manager.feed_audio(
                chunk,
                vad_result["voice_state"].value,
                metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error processing audio chunk: {e}")
    
    async def _on_segment_ready(
        self,
        peer_id: str,
        audio_data: bytes,
        metadata: Dict[str, Any]
    ):
        """
        Callback when complete speech segment ready for transcription.
        
        This is where we inject the /no_think prefix and call SimpleVoiceService.
        """
        try:
            self.logger.info(
                f"Speech segment ready for {peer_id}: "
                f"{len(audio_data)} bytes, {metadata['duration_sec']:.2f}s"
            )
            
            start_time = time.time()
            
            # Convert PCM bytes to numpy array for voice service
            import numpy as np
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Process through voice service (STT → LLM → TTS)
            result = await self._process_voice_with_service(audio_array, metadata)
            
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            self.utterances_processed += 1
            
            self.logger.info(
                f"Voice processing complete for {peer_id} in {processing_time:.2f}s"
            )
            
        except Exception as e:
            self.logger.error(f"Error processing segment: {e}")
    
    async def _process_voice_with_service(
        self,
        audio_array: 'np.ndarray',
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process voice through SimpleVoiceService with /no_think prefix.
        
        Args:
            audio_array: Audio numpy array
            metadata: Segment metadata
            
        Returns:
            dict: Processing result
        """
        if not self.voice_service:
            self.logger.error("Voice service not available")
            return {"success": False, "error": "Voice service not available"}
        
        try:
            # 1. STT: Transcribe audio to text
            transcription_result = await self.voice_service.transcribe_audio(
                audio_data=audio_array.tobytes(),
                language=self.language if self.language != "en" else None  # Auto-detect for English
            )
            
            if not transcription_result.get("success"):
                self.logger.error("Transcription failed")
                return transcription_result
            
            transcript = transcription_result.get("transcription", "")
            
            if self._on_transcription:
                self._on_transcription(self.peer_id, transcript)
            
            self.logger.info(f"Transcription: {transcript}")
            
            # 2. LLM: Generate response with automatic /no_think prefix injection
            if self.config.auto_inject_no_think:
                # Inject /no_think prefix if not already present
                if not transcript.strip().startswith(self.config.no_think_prefix):
                    llm_input = self.config.no_think_prefix + transcript
                    self.logger.debug(f"Injected prefix: {llm_input}")
                else:
                    llm_input = transcript
            else:
                llm_input = transcript
            
            # Generate LLM response
            llm_result = await self.voice_service.generate_chat_response(
                user_message=llm_input,
                session_id=self.session_id,
                language=self.language
            )
            
            if not llm_result.get("success"):
                self.logger.error("LLM generation failed")
                return llm_result
            
            llm_response = llm_result.get("response", "")
            
            if self._on_llm_response:
                self._on_llm_response(self.peer_id, llm_response)
            
            self.logger.info(f"LLM response: {llm_response[:100]}...")
            
            # 3. TTS: Generate speech
            tts_result = await self.voice_service.synthesize_speech(
                text=llm_response,
                language=self.language,
                gender=self.config.default_gender
            )
            
            if not tts_result.get("success"):
                self.logger.error("TTS synthesis failed")
                return tts_result
            
            tts_audio = tts_result.get("audio_data")
            
            if self._on_tts_audio:
                self._on_tts_audio(self.peer_id, tts_audio)
            
            return {
                "success": True,
                "transcription": transcript,
                "llm_response": llm_response,
                "tts_audio_size": len(tts_audio) if tts_audio else 0,
                "llm_input_with_prefix": llm_input if self.config.auto_inject_no_think else None
            }
            
        except Exception as e:
            self.logger.error(f"Error in voice processing pipeline: {e}")
            return {"success": False, "error": str(e)}
    
    # Component callbacks
    
    def _on_voice_start(self, peer_id: str):
        """Callback when VAD detects voice start."""
        self.logger.debug(f"Voice started for {peer_id}")
    
    def _on_voice_end(self, peer_id: str, duration: float):
        """Callback when VAD detects voice end."""
        self.logger.debug(f"Voice ended for {peer_id}: {duration:.2f}s")
    
    def _on_vad_state_change(self, peer_id: str, new_state: VADState):
        """Callback when VAD state changes."""
        self.logger.debug(f"VAD state change for {peer_id}: {new_state.value}")
    
    def _on_utterance_limit(self, peer_id: str):
        """Callback when utterance exceeds 10s limit."""
        self.logger.warning(f"Utterance limit exceeded for {peer_id}")
        # Could trigger warning message or force finalization
    
    def _on_buffer_overflow(self, peer_id: str):
        """Callback when buffer overflows."""
        self.logger.warning(f"Buffer overflow for {peer_id}")
    
    def _on_processing_error(self, peer_id: str, error: Exception):
        """Callback for processing errors."""
        self.logger.error(f"Processing error for {peer_id}: {error}")
    
    # Metrics and monitoring
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get complete pipeline metrics.
        
        Returns:
            dict: Aggregated metrics from all components
        """
        metrics = {
            "peer_id": self.peer_id,
            "session_id": self.session_id,
            "language": self.language,
            "is_active": self.is_active,
            "utterances_processed": self.utterances_processed,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": (
                self.total_processing_time / self.utterances_processed
                if self.utterances_processed > 0 else 0
            )
        }
        
        if self.audio_processor:
            metrics["audio_processor"] = self.audio_processor.get_metrics()
        
        if self.vad_service:
            metrics["vad"] = self.vad_service.get_metrics()
        
        if self.buffer_manager:
            metrics["buffer"] = self.buffer_manager.get_metrics()
        
        return metrics
    
    async def cleanup(self):
        """Cleanup all pipeline resources."""
        self.logger.info(f"Cleaning up voice pipeline for {self.peer_id}")
        
        await self.stop_voice_session()
        
        if self.audio_processor:
            await self.audio_processor.stop_processing()
        
        if self.vad_service:
            await self.vad_service.cleanup()
        
        if self.buffer_manager:
            await self.buffer_manager.cleanup()
        
        self.is_initialized = False


# Factory function
def create_webrtc_voice_adapter(
    peer_id: str,
    session_id: str,
    language: str = "en",
    config: Optional[WebRTCVoiceConfig] = None,
    voice_service = None,
    **callbacks
) -> WebRTCVoiceServiceAdapter:
    """
    Factory function to create WebRTC voice service adapter.
    
    Args:
        peer_id: WebRTC peer identifier
        session_id: Voice session identifier
        language: Language code
        config: Pipeline configuration
        voice_service: SimpleVoiceService instance
        **callbacks: Optional callbacks
        
    Returns:
        WebRTCVoiceServiceAdapter instance
    """
    return WebRTCVoiceServiceAdapter(
        peer_id=peer_id,
        session_id=session_id,
        language=language,
        config=config,
        voice_service=voice_service,
        on_transcription=callbacks.get('on_transcription'),
        on_llm_response=callbacks.get('on_llm_response'),
        on_tts_audio=callbacks.get('on_tts_audio')
    )
