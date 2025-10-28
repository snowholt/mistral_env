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

from .webrtc_audio_processor import (
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
            
            # Link audio processor to VAD for lifecycle notifications
            if self.audio_processor:
                self.audio_processor.vad_service = self.vad_service

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

            if self.vad_service:
                self.vad_service.attach_buffer_manager(self.buffer_manager)
            
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
            print(f"[DEBUG-CHUNK] Audio chunk: {len(chunk)} bytes for {self.peer_id}")
            self.logger.debug(f"Audio chunk received: {len(chunk)} bytes")
            
            # Process with VAD
            vad_result = await self.vad_service.process_audio_chunk(chunk, metadata)
            
            if not vad_result["success"]:
                self.logger.warning(f"VAD processing failed: {vad_result.get('error', 'Unknown error')}")
                return
            
            voice_state = vad_result['voice_state'].value
            voice_detected = vad_result['voice_detected']
            silero_prob = vad_result.get('silero_probability', 0.0)
            webrtc_det = vad_result.get('webrtc_detected', False)
            print(f"[DEBUG-VAD] State={voice_state}, detected={voice_detected}, silero_prob={silero_prob:.4f}, webrtc={webrtc_det}")
            self.logger.info(f"[ADAPTER] VAD result for {self.peer_id}: state={voice_state}, detected={voice_detected}, chunk_size={len(chunk)}")
            
            # Feed to buffer manager with VAD state
            buffer_result = await self.buffer_manager.feed_audio(
                chunk,
                voice_state,
                metadata
            )
            
            buffer_size = self.buffer_manager.get_buffer_size_bytes()
            print(f"[DEBUG-BUFFER] Status={buffer_result.get('status')}, segment_ready={buffer_result.get('segment_ready')}, buffer_size={buffer_size}")
            self.logger.info(f"[ADAPTER] Buffer result for {self.peer_id}: status={buffer_result.get('status')}, segment_ready={buffer_result.get('segment_ready')}, buffer_size={buffer_size}")
            
        except Exception as e:
            print(f"[DEBUG-ERROR] Chunk processing failed: {e}")
            self.logger.error(f"Error processing audio chunk: {e}", exc_info=True)
    
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
                language=self.language  # Pass language through (Whisper auto-detects if None)
            )
            
            if not transcription_result.get("success"):
                self.logger.error(f"[ADAPTER] Transcription failed for peer {self.peer_id}")
                return transcription_result
            
            transcript = transcription_result.get("transcription", "")
            
            self.logger.info(f"[ADAPTER] ✓ Transcription complete for peer {self.peer_id}: '{transcript}'")
            
            if self._on_transcription:
                self.logger.info(f"[ADAPTER] Calling on_transcription callback for peer {self.peer_id}")
                self._on_transcription(self.peer_id, transcript)
            else:
                self.logger.warning(f"[ADAPTER] No on_transcription callback registered for peer {self.peer_id}")
            
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
            
            self.logger.info(f"[ADAPTER] ✓ LLM response generated for peer {self.peer_id}: '{llm_response[:100]}...'")
            
            if self._on_llm_response:
                self.logger.info(f"[ADAPTER] Calling on_llm_response callback for peer {self.peer_id}")
                self._on_llm_response(self.peer_id, llm_response)
            else:
                self.logger.warning(f"[ADAPTER] No on_llm_response callback registered for peer {self.peer_id}")
            
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
    
    async def _on_utterance_limit(self, peer_id: str):
        """Callback when utterance exceeds 10s limit - force transcription."""
        print(f"[DEBUG-LIMIT] Utterance limit for {peer_id}, buffer_size={self.buffer_manager.get_buffer_size_bytes() if self.buffer_manager else 'N/A'}")
        self.logger.warning(f"Utterance limit exceeded for {peer_id}, forcing transcription of buffered audio")
        # Force transcription of whatever audio we have buffered
        if self.buffer_manager:
            try:
                self.logger.info(f"[ADAPTER] Attempting to force-finalize buffered audio for {peer_id}")

                snapshot_bytes = self.buffer_manager.get_buffer_size_bytes()
                snapshot_frames = len(self.buffer_manager._active_buffer) if hasattr(self.buffer_manager, "_active_buffer") else "?"
                snapshot_state = getattr(self.buffer_manager, "is_recording", None)
                print(
                    f"[DEBUG-LIMIT] Force finalize snapshot bytes={snapshot_bytes}, "
                    f"frames={snapshot_frames}, is_recording={snapshot_state}"
                )
                self.logger.info(
                    f"[ADAPTER] Force finalize snapshot for {peer_id}: "
                    f"bytes={snapshot_bytes}, frames={snapshot_frames}, "
                    f"is_recording={snapshot_state}"
                )

                active_chunks = list(getattr(self.buffer_manager, "_active_buffer", []))
                if not active_chunks:
                    print(f"[DEBUG-LIMIT] Buffer is EMPTY (segment=None)")
                    self.logger.warning(f"[ADAPTER] No buffered audio available for {peer_id}")
                    return

                segment = b"".join(active_chunks)
                now = time.time()
                speech_start = getattr(self.buffer_manager, "speech_start_time", None)
                duration = (now - speech_start) if speech_start else len(segment) / (self.buffer_manager.config.sample_rate * 2)

                segment_metadata = {
                    "peer_id": peer_id,
                    "duration_sec": duration,
                    "num_frames": len(active_chunks),
                    "pre_roll_frames": getattr(self.buffer_manager.metrics, "pre_roll_chunks", 0),
                    "speech_frames": getattr(self.buffer_manager.metrics, "speech_chunks", len(active_chunks)),
                    "post_roll_frames": getattr(self.buffer_manager.metrics, "post_roll_chunks", 0),
                    "total_bytes": len(segment),
                    "sample_rate": self.buffer_manager.config.sample_rate,
                    "timestamp": now,
                    "forced": True,
                    "forced_by_limit": True
                }

                # Reset buffer for next utterance
                self.buffer_manager.reset()

                print(f"[DEBUG-LIMIT] Got {len(segment)} bytes ({duration:.2f}s), transcribing...")
                self.logger.info(
                    f"[ADAPTER] Force-finalized {len(segment)} bytes "
                    f"({duration:.2f}s) from buffer, triggering transcription"
                )

                await self._on_segment_ready(peer_id, segment, segment_metadata)
            except Exception as e:
                print(f"[DEBUG-LIMIT] ERROR: {e}")
                self.logger.error(f"[ADAPTER] Failed to process buffered audio on limit: {e}", exc_info=True)
    
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
