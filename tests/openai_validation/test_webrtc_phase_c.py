"""
Comprehensive Tests for WebRTC Phase C Components

Tests audio processor, VAD service, buffer manager, and voice adapter integration.

Test Coverage:
- Audio frame conversion and resampling
- Utterance limit enforcement  
- Dual VAD (WebRTC + Silero)
- Language-specific thresholds
- Pre-roll/post-roll buffering
- /no_think prefix injection
- Complete pipeline integration

Author: BeautyAI Framework  
Date: 2025-10-15
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from pathlib import Path

# Phase C imports
try:
    from beautyai_inference.services.voice.webrtc_audio_processor import (
        WebRTCAudioProcessor,
        AudioProcessingConfig,
        create_audio_processor
    )
    from beautyai_inference.services.voice.vad.webrtc_vad_service import (
        WebRTCVADService,
        WebRTCVADConfig,
        VADState,
        create_webrtc_vad_service
    )
    from beautyai_inference.core.webrtc_buffer_manager import (
        WebRTCBufferManager,
        BufferConfig,
        create_buffer_manager
    )
    from beautyai_inference.services.voice.webrtc_voice_service_adapter import (
        WebRTCVoiceServiceAdapter,
        WebRTCVoiceConfig,
        create_webrtc_voice_adapter
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    pytest.skip(f"Phase C imports not available: {e}", allow_module_level=True)


# ===========================
# Audio Processor Tests
# ===========================

class TestWebRTCAudioProcessor:
    """Tests for WebRTC audio processor component."""
    
    @pytest.fixture
    def audio_config(self):
        """Create test audio configuration."""
        return AudioProcessingConfig(
            target_sample_rate=16000,
            max_utterance_duration_sec=10,
            enable_level_monitoring=True
        )
    
    @pytest.fixture
    def mock_audio_frame(self):
        """Create mock AudioFrame object."""
        frame = MagicMock()
        frame.sample_rate = 48000
        frame.samples = 480
        frame.layout.channels = [0]  # Mono
        # Create float32 audio data in range [-1, 1]
        audio_data = np.random.uniform(-0.5, 0.5, 480).astype(np.float32)
        frame.to_ndarray.return_value = audio_data
        return frame
    
    def test_processor_initialization(self, audio_config):
        """Test audio processor initialization."""
        processor = WebRTCAudioProcessor(
            peer_id="test_peer",
            config=audio_config
        )
        
        assert processor.peer_id == "test_peer"
        assert processor.config.target_sample_rate == 16000
        assert processor.is_processing is False
    
    def test_factory_function(self, audio_config):
        """Test processor factory function."""
        processor = create_audio_processor(
            peer_id="test_peer",
            config=audio_config
        )
        
        assert isinstance(processor, WebRTCAudioProcessor)
        assert processor.peer_id == "test_peer"
    
    @pytest.mark.asyncio
    async def test_pcm_conversion(self, audio_config):
        """Test audio frame to PCM conversion."""
        processor = WebRTCAudioProcessor("test_peer", audio_config)
        
        # Create test audio array
        audio_float = np.array([0.5, -0.5, 0.25, -0.25], dtype=np.float32)
        
        # Convert to PCM
        pcm_bytes = processor._numpy_to_pcm(audio_float)
        
        # Verify conversion
        pcm_array = np.frombuffer(pcm_bytes, dtype=np.int16)
        assert len(pcm_array) == 4
        assert pcm_array[0] > 0  # Positive values
        assert pcm_array[1] < 0  # Negative values
    
    @pytest.mark.asyncio
    async def test_resampling(self, audio_config):
        """Test audio resampling from 48kHz to 16kHz."""
        processor = WebRTCAudioProcessor("test_peer", audio_config)
        
        # Create 48kHz audio (1 second = 48000 samples)
        source_audio = np.random.uniform(-0.5, 0.5, 48000).astype(np.float32)
        
        # Resample to 16kHz
        resampled = processor._resample_audio(source_audio, 48000, 16000)
        
        # Should have approximately 16000 samples
        assert 15900 < len(resampled) < 16100
    
    @pytest.mark.asyncio
    async def test_utterance_limit_enforcement(self, audio_config):
        """Test 10-second utterance limit enforcement."""
        limit_exceeded = False
        
        def on_limit_exceeded(peer_id):
            nonlocal limit_exceeded
            limit_exceeded = True
        
        processor = WebRTCAudioProcessor(
            "test_peer",
            audio_config,
            on_utterance_limit_exceeded=on_limit_exceeded
        )
        
        # Simulate exceeding limit
        processor.current_utterance_duration = 11.0  # Over 10 seconds
        
        # Create mock frame
        mock_frame = MagicMock()
        mock_frame.sample_rate = 16000
        mock_frame.layout.channels = [0]
        mock_frame.to_ndarray.return_value = np.zeros(480, dtype=np.float32)
        
        # Process frame (should trigger limit)
        await processor._process_audio_frame(mock_frame)
        
        assert limit_exceeded is True
    
    def test_audio_level_calculation(self, audio_config):
        """Test RMS audio level calculation."""
        processor = WebRTCAudioProcessor("test_peer", audio_config)
        
        # Test with known audio levels
        silence = np.zeros(1000, dtype=np.float32)
        loud_audio = np.ones(1000, dtype=np.float32) * 0.5
        
        silence_level = processor._calculate_audio_level(silence)
        loud_level = processor._calculate_audio_level(loud_audio)
        
        assert silence_level == 0.0
        assert loud_level > 0.4  # Should be close to 0.5


# ===========================
# VAD Service Tests
# ===========================

class TestWebRTCVADService:
    """Tests for WebRTC dual VAD service."""
    
    @pytest.fixture
    def vad_config(self):
        """Create test VAD configuration."""
        return WebRTCVADConfig(
            webrtc_sensitivity=3,
            silero_sensitivity=0.5,
            language_thresholds={"ar": 0.45, "en": 0.50},
            min_speech_duration_ms=300,
            post_speech_silence_ms=500
        )
    
    def test_vad_initialization(self, vad_config):
        """Test VAD service initialization."""
        vad = WebRTCVADService(
            peer_id="test_peer",
            language="ar",
            config=vad_config
        )
        
        assert vad.peer_id == "test_peer"
        assert vad.language == "ar"
        assert vad.silero_threshold == 0.45  # Arabic threshold
        assert vad.current_state == VADState.INACTIVE
    
    def test_language_specific_thresholds(self, vad_config):
        """Test language-specific VAD thresholds."""
        vad_ar = WebRTCVADService("peer1", "ar", vad_config)
        vad_en = WebRTCVADService("peer2", "en", vad_config)
        vad_default = WebRTCVADService("peer3", "fr", vad_config)
        
        assert vad_ar.silero_threshold == 0.45  # Arabic
        assert vad_en.silero_threshold == 0.50  # English
        assert vad_default.silero_threshold == 0.50  # Default
    
    @pytest.mark.asyncio
    async def test_vad_state_transitions(self, vad_config):
        """Test VAD state machine transitions."""
        state_changes = []
        
        def on_state_change(peer_id, new_state):
            state_changes.append(new_state)
        
        vad = WebRTCVADService(
            "test_peer",
            "en",
            vad_config,
            on_vad_state_change=on_state_change
        )
        
        # INACTIVE → VOICE_START (voice detected)
        await vad._update_state(True, {})
        assert vad.current_state == VADState.VOICE_START
        
        # VOICE_START → VOICE_ACTIVE (min duration passed)
        import time
        vad.speech_start_time = time.time() - 0.4  # 400ms ago
        await vad._update_state(True, {})
        assert vad.current_state == VADState.VOICE_ACTIVE
        
        # VOICE_ACTIVE → VOICE_END_PENDING (silence)
        await vad._update_state(False, {})
        assert vad.current_state == VADState.VOICE_END_PENDING
    
    def test_dual_vad_strategy(self, vad_config):
        """Test dual VAD decision logic."""
        vad = WebRTCVADService("test_peer", "en", vad_config)
        
        # Strict mode: Both must agree
        vad.config.require_silero_confirmation = True
        assert vad._determine_voice_detection(True, True, 0.6) is True
        assert vad._determine_voice_detection(True, False, 0.3) is False
        assert vad._determine_voice_detection(False, True, 0.6) is False
        
        # Permissive mode: Either can trigger
        vad.config.require_silero_confirmation = False
        assert vad._determine_voice_detection(True, False, 0.3) is True
        assert vad._determine_voice_detection(False, True, 0.6) is True
    
    def test_factory_function(self, vad_config):
        """Test VAD factory function."""
        vad = create_webrtc_vad_service(
            peer_id="test_peer",
            language="ar",
            config=vad_config
        )
        
        assert isinstance(vad, WebRTCVADService)
        assert vad.language == "ar"


# ===========================
# Buffer Manager Tests
# ===========================

class TestWebRTCBufferManager:
    """Tests for WebRTC buffer manager."""
    
    @pytest.fixture
    def buffer_config(self):
        """Create test buffer configuration."""
        return BufferConfig(
            pre_roll_duration_ms=300,
            post_roll_duration_ms=300,
            frame_duration_ms=30,
            sample_rate=16000
        )
    
    def test_buffer_initialization(self, buffer_config):
        """Test buffer manager initialization."""
        buffer_mgr = WebRTCBufferManager(
            peer_id="test_peer",
            config=buffer_config
        )
        
        assert buffer_mgr.peer_id == "test_peer"
        assert buffer_mgr.is_recording is False
        assert len(buffer_mgr._pre_roll_buffer) == 0
    
    @pytest.mark.asyncio
    async def test_pre_roll_buffering(self, buffer_config):
        """Test pre-roll buffer maintenance."""
        buffer_mgr = WebRTCBufferManager("test_peer", buffer_config)
        
        # Feed chunks to pre-roll buffer (INACTIVE state)
        test_chunk = b'\x00' * 960  # 30ms of 16kHz mono audio
        
        for _ in range(15):  # Feed multiple chunks
            result = await buffer_mgr.feed_audio(
                test_chunk,
                VADState.INACTIVE.value,
                {}
            )
            assert result["status"] == "buffering_pre_roll"
        
        # Pre-roll buffer should have frames
        assert len(buffer_mgr._pre_roll_buffer) > 0
    
    @pytest.mark.asyncio
    async def test_speech_recording_flow(self, buffer_config):
        """Test complete speech recording flow with pre/post roll."""
        segment_ready = False
        captured_audio = None
        
        def on_segment_ready(peer_id, audio, metadata):
            nonlocal segment_ready, captured_audio
            segment_ready = True
            captured_audio = audio
        
        buffer_mgr = WebRTCBufferManager(
            "test_peer",
            buffer_config,
            on_segment_ready=on_segment_ready
        )
        
        test_chunk = b'\x00' * 960
        
        # 1. Build pre-roll buffer
        for _ in range(10):
            await buffer_mgr.feed_audio(test_chunk, VADState.INACTIVE.value, {})
        
        # 2. Voice starts (pre-roll copied to active)
        result = await buffer_mgr.feed_audio(test_chunk, VADState.VOICE_START.value, {})
        assert result["status"] == "recording_speech"
        assert buffer_mgr.is_recording is True
        
        # 3. Continue recording
        for _ in range(20):
            await buffer_mgr.feed_audio(test_chunk, VADState.VOICE_ACTIVE.value, {})
        
        # 4. Voice ends, post-roll starts
        for _ in range(buffer_mgr.post_roll_frames):
            result = await buffer_mgr.feed_audio(test_chunk, VADState.VOICE_END.value, {})
        
        # Segment should be ready
        assert segment_ready is True
        assert captured_audio is not None
        assert len(captured_audio) > 0
    
    @pytest.mark.asyncio
    async def test_buffer_overflow_protection(self, buffer_config):
        """Test buffer overflow handling."""
        overflow_triggered = False
        
        def on_overflow(peer_id):
            nonlocal overflow_triggered
            overflow_triggered = True
        
        buffer_mgr = WebRTCBufferManager(
            "test_peer",
            buffer_config,
            on_buffer_overflow=on_overflow
        )
        
        test_chunk = b'\x00' * 960
        
        # Start recording
        await buffer_mgr.feed_audio(test_chunk, VADState.VOICE_START.value, {})
        
        # Fill buffer beyond max
        for _ in range(buffer_mgr.max_buffer_frames + 10):
            await buffer_mgr.feed_audio(test_chunk, VADState.VOICE_ACTIVE.value, {})
        
        # Should trigger overflow
        assert overflow_triggered is True


# ===========================
# Voice Adapter Integration Tests
# ===========================

class TestWebRTCVoiceServiceAdapter:
    """Tests for complete voice service adapter integration."""
    
    @pytest.fixture
    def voice_config(self):
        """Create test voice configuration."""
        return WebRTCVoiceConfig(
            auto_inject_no_think=True,
            default_language="ar"
        )
    
    @pytest.fixture
    def mock_voice_service(self):
        """Create mock SimpleVoiceService."""
        service = Mock()
        service.transcribe_audio = AsyncMock(return_value={
            "success": True,
            "transcription": "test transcription"
        })
        service.generate_chat_response = AsyncMock(return_value={
            "success": True,
            "response": "test response"
        })
        service.synthesize_speech = AsyncMock(return_value={
            "success": True,
            "audio_data": b"test audio"
        })
        return service
    
    def test_adapter_initialization(self, voice_config, mock_voice_service):
        """Test voice adapter initialization."""
        adapter = WebRTCVoiceServiceAdapter(
            peer_id="test_peer",
            session_id="session123",
            language="ar",
            config=voice_config,
            voice_service=mock_voice_service
        )
        
        assert adapter.peer_id == "test_peer"
        assert adapter.session_id == "session123"
        assert adapter.language == "ar"
        assert adapter.is_initialized is False
    
    @pytest.mark.asyncio
    async def test_adapter_initialization_pipeline(self, voice_config, mock_voice_service):
        """Test adapter pipeline component initialization."""
        adapter = WebRTCVoiceServiceAdapter(
            "test_peer",
            "session123",
            "en",
            voice_config,
            mock_voice_service
        )
        
        # Initialize (VAD may fail without torch, which is okay for testing structure)
        # Mock the VAD initialization to succeed
        with patch.object(WebRTCVADService, 'initialize', return_value=True):
            result = await adapter.initialize()
            
            assert adapter.audio_processor is not None
            assert adapter.vad_service is not None
            assert adapter.buffer_manager is not None
    
    @pytest.mark.asyncio
    async def test_no_think_prefix_injection(self, voice_config, mock_voice_service):
        """Test automatic /no_think prefix injection."""
        adapter = WebRTCVoiceServiceAdapter(
            "test_peer",
            "session123",
            "en",
            voice_config,
            mock_voice_service
        )
        
        # Create test audio array
        audio_array = np.zeros(16000, dtype=np.int16)
        
        # Process voice (should inject /no_think prefix)
        result = await adapter._process_voice_with_service(
            audio_array,
            {"duration_sec": 1.0}
        )
        
        # Verify LLM was called with prefixed input
        mock_voice_service.generate_chat_response.assert_called_once()
        call_args = mock_voice_service.generate_chat_response.call_args
        llm_input = call_args.kwargs.get('user_message')
        
        assert llm_input.startswith("/no_think ")
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_flow(self, voice_config, mock_voice_service):
        """Test complete voice processing pipeline."""
        transcriptions = []
        llm_responses = []
        tts_audio = []
        
        def on_transcription(peer_id, text):
            transcriptions.append(text)
        
        def on_llm_response(peer_id, text):
            llm_responses.append(text)
        
        def on_tts_audio(peer_id, audio):
            tts_audio.append(audio)
        
        adapter = WebRTCVoiceServiceAdapter(
            "test_peer",
            "session123",
            "ar",
            voice_config,
            mock_voice_service,
            on_transcription=on_transcription,
            on_llm_response=on_llm_response,
            on_tts_audio=on_tts_audio
        )
        
        # Process voice segment
        audio_array = np.zeros(16000, dtype=np.int16)
        result = await adapter._process_voice_with_service(
            audio_array,
            {"duration_sec": 1.0}
        )
        
        # Verify complete pipeline executed
        assert result["success"] is True
        assert len(transcriptions) == 1
        assert len(llm_responses) == 1
        assert len(tts_audio) == 1
        
        # Verify /no_think prefix was injected
        assert result.get("llm_input_with_prefix") is not None
        assert result["llm_input_with_prefix"].startswith("/no_think ")
    
    def test_factory_function(self, voice_config, mock_voice_service):
        """Test adapter factory function."""
        adapter = create_webrtc_voice_adapter(
            peer_id="test_peer",
            session_id="session123",
            language="ar",
            config=voice_config,
            voice_service=mock_voice_service
        )
        
        assert isinstance(adapter, WebRTCVoiceServiceAdapter)


# ===========================
# Integration Tests
# ===========================

class TestPhaseC Integration:
    """Integration tests for complete Phase C pipeline."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_audio_flow(self):
        """Test complete audio flow from PCM to buffered segment."""
        # Create components
        buffer_mgr = create_buffer_manager("test_peer")
        
        # Simulate audio flow
        test_pcm = b'\x00' * 960  # 30ms chunk
        
        # Build pre-roll
        for _ in range(10):
            await buffer_mgr.feed_audio(test_pcm, VADState.INACTIVE.value, {})
        
        # Start speech
        await buffer_mgr.feed_audio(test_pcm, VADState.VOICE_START.value, {})
        assert buffer_mgr.is_recording is True
        
        # Record speech
        for _ in range(30):
            await buffer_mgr.feed_audio(test_pcm, VADState.VOICE_ACTIVE.value, {})
        
        # Verify buffer accumulated correctly
        assert buffer_mgr.get_buffer_size_bytes() > 0
    
    def test_component_metrics_collection(self):
        """Test metrics collection from all components."""
        # Create components
        processor = create_audio_processor("test_peer")
        buffer_mgr = create_buffer_manager("test_peer")
        
        # Get metrics
        processor_metrics = processor.get_metrics()
        buffer_metrics = buffer_mgr.get_metrics()
        
        # Verify metrics structure
        assert "peer_id" in processor_metrics
        assert "frames_processed" in processor_metrics
        assert "peer_id" in buffer_metrics
        assert "chunks_received" in buffer_metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
