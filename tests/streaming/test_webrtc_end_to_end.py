"""
End-to-End WebRTC Voice Pipeline Integration Tests

Tests the complete WebRTC voice-to-voice flow using pre-recorded PCM audio
to simulate the audio track flow.

Author: BeautyAI Framework
Date: October 15, 2025
"""

import asyncio
import pytest
import time
import wave
import io
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Import WebRTC components
try:
    from beautyai_inference.services.voice.webrtc_audio_processor import WebRTCAudioProcessor
    from beautyai_inference.services.voice.vad.webrtc_vad_service import WebRTCVADService, VADState
    from beautyai_inference.core.webrtc_buffer_manager import WebRTCBufferManager
    from beautyai_inference.services.voice.webrtc_voice_service_adapter import WebRTCVoiceServiceAdapter
    from beautyai_inference.services.voice.simple_voice_service import SimpleVoiceService
    WEBRTC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: WebRTC components not available: {e}")
    WEBRTC_AVAILABLE = False


@dataclass
class EndToEndTestResult:
    """Results from end-to-end test execution."""
    success: bool
    total_duration_ms: float
    audio_processing_ms: float
    vad_detection_ms: float
    buffer_recording_ms: float
    transcription_ms: float
    llm_generation_ms: float
    tts_generation_ms: float
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = None


class MockAudioFrame:
    """Mock audio frame simulating aiortc AudioFrame."""
    
    def __init__(self, pcm_data: bytes, sample_rate: int = 48000, channels: int = 1):
        self.pcm_data = pcm_data
        self.sample_rate = sample_rate
        self.channels = channels
        self.samples = len(pcm_data) // 2  # 16-bit samples
        
    def to_ndarray(self):
        """Convert to numpy array format."""
        import numpy as np
        return np.frombuffer(self.pcm_data, dtype=np.int16)


class PCMAudioSimulator:
    """Simulates feeding PCM audio as WebRTC frames."""
    
    def __init__(self, pcm_file_path: Path, frame_duration_ms: int = 30):
        """
        Initialize simulator with PCM file.
        
        Args:
            pcm_file_path: Path to PCM audio file (16kHz, mono, 16-bit)
            frame_duration_ms: Frame duration in milliseconds (default 30ms)
        """
        self.pcm_file_path = pcm_file_path
        self.frame_duration_ms = frame_duration_ms
        self.sample_rate = 16000  # Expected sample rate for PCM files
        self.channels = 1
        self.sample_width = 2  # 16-bit = 2 bytes
        
        # Calculate frame size
        self.samples_per_frame = (self.sample_rate * frame_duration_ms) // 1000
        self.bytes_per_frame = self.samples_per_frame * self.sample_width
        
    def load_pcm_data(self) -> bytes:
        """Load PCM data from file."""
        with open(self.pcm_file_path, 'rb') as f:
            return f.read()
    
    def generate_frames(self) -> List[MockAudioFrame]:
        """Generate list of mock audio frames from PCM file."""
        pcm_data = self.load_pcm_data()
        frames = []
        
        # Split into frames
        for i in range(0, len(pcm_data), self.bytes_per_frame):
            frame_data = pcm_data[i:i + self.bytes_per_frame]
            
            # Pad last frame if needed
            if len(frame_data) < self.bytes_per_frame:
                frame_data += b'\x00' * (self.bytes_per_frame - len(frame_data))
            
            # Create mock frame (WebRTC uses 48kHz, but we'll use 16kHz for testing)
            frames.append(MockAudioFrame(
                pcm_data=frame_data,
                sample_rate=self.sample_rate,
                channels=self.channels
            ))
        
        return frames


@pytest.mark.skipif(not WEBRTC_AVAILABLE, reason="WebRTC components not available")
class TestWebRTCEndToEnd:
    """End-to-end integration tests for WebRTC voice pipeline."""
    
    @pytest.fixture
    def pcm_test_files(self) -> Dict[str, Path]:
        """Get paths to PCM test files."""
        base_path = Path(__file__).parent.parent.parent / "voice_tests" / "input_test_questions" / "pcm"
        
        test_files = {
            "q1_arabic": base_path / "q1.pcm",
            "q2_arabic": base_path / "q2.pcm",
        }
        
        # Verify files exist
        existing_files = {name: path for name, path in test_files.items() if path.exists()}
        
        if not existing_files:
            pytest.skip("No PCM test files available")
        
        return existing_files
    
    @pytest.fixture
    async def audio_processor(self):
        """Create WebRTC audio processor instance."""
        config = {
            "max_utterance_sec": 10,
            "sample_rate": 16000,
            "channels": 1,
            "enable_resampling": True
        }
        processor = WebRTCAudioProcessor(config)
        await processor.initialize()
        yield processor
        await processor.cleanup()
    
    @pytest.fixture
    async def vad_service(self):
        """Create VAD service instance."""
        config = {
            "language": "ar",
            "silero_threshold": 0.45,
            "webrtc_aggressiveness": 2,
            "enable_dual_vad": True
        }
        vad = WebRTCVADService(config)
        await vad.initialize()
        yield vad
        await vad.cleanup()
    
    @pytest.fixture
    async def buffer_manager(self):
        """Create buffer manager instance."""
        config = {
            "pre_roll_ms": 300,
            "post_roll_ms": 300,
            "max_buffer_size_mb": 50
        }
        manager = WebRTCBufferManager(config)
        await manager.initialize()
        yield manager
        await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_arabic_q1(self, pcm_test_files, audio_processor, 
                                               vad_service, buffer_manager):
        """
        Test complete WebRTC voice pipeline with Arabic Q1 audio.
        
        Flow:
        1. Load PCM file and generate frames
        2. Process frames through audio processor
        3. Run VAD detection
        4. Record to buffer when speech detected
        5. Mock transcription, LLM, and TTS
        6. Measure end-to-end latency
        """
        if "q1_arabic" not in pcm_test_files:
            pytest.skip("Q1 Arabic PCM file not available")
        
        test_file = pcm_test_files["q1_arabic"]
        start_time = time.time()
        
        # Step 1: Generate audio frames
        simulator = PCMAudioSimulator(test_file, frame_duration_ms=30)
        frames = simulator.generate_frames()
        
        assert len(frames) > 0, "No frames generated from PCM file"
        print(f"Generated {len(frames)} frames from {test_file.name}")
        
        # Step 2: Process frames through audio processor
        audio_processing_start = time.time()
        processed_frames = []
        
        for frame in frames:
            pcm_output = await audio_processor.process_frame(frame)
            if pcm_output:
                processed_frames.append(pcm_output)
        
        audio_processing_duration = (time.time() - audio_processing_start) * 1000
        assert len(processed_frames) > 0, "No frames processed"
        print(f"Audio processing: {audio_processing_duration:.2f}ms for {len(processed_frames)} frames")
        
        # Step 3: Run VAD detection
        vad_detection_start = time.time()
        vad_results = []
        
        for pcm_data in processed_frames:
            vad_state = await vad_service.process_audio(pcm_data)
            vad_results.append(vad_state)
        
        vad_detection_duration = (time.time() - vad_detection_start) * 1000
        speech_detected = any(state in [VADState.VOICE_ACTIVE, VADState.VOICE_START] 
                             for state in vad_results)
        
        assert speech_detected, "No speech detected by VAD"
        print(f"VAD detection: {vad_detection_duration:.2f}ms, speech detected: {speech_detected}")
        
        # Step 4: Record to buffer
        buffer_recording_start = time.time()
        
        for i, (pcm_data, vad_state) in enumerate(zip(processed_frames, vad_results)):
            await buffer_manager.feed_audio(pcm_data, vad_state)
        
        buffer_recording_duration = (time.time() - buffer_recording_start) * 1000
        
        # Get recorded audio
        recorded_audio = buffer_manager.get_recording()
        assert recorded_audio is not None and len(recorded_audio) > 0, "No audio recorded"
        print(f"Buffer recording: {buffer_recording_duration:.2f}ms, recorded {len(recorded_audio)} bytes")
        
        # Step 5: Mock downstream processing (STT, LLM, TTS)
        # In real scenario, this would call SimpleVoiceService
        transcription_start = time.time()
        mock_transcription = "ما هو الذكاء الاصطناعي؟"  # Simulated transcription
        transcription_duration = (time.time() - transcription_start) * 1000
        
        llm_generation_start = time.time()
        mock_llm_response = "الذكاء الاصطناعي هو..."  # Simulated LLM response
        llm_generation_duration = (time.time() - llm_generation_start) * 1000
        
        tts_generation_start = time.time()
        mock_tts_audio = b'\x00' * 10000  # Simulated TTS audio
        tts_generation_duration = (time.time() - tts_generation_start) * 1000
        
        # Calculate total duration
        total_duration = (time.time() - start_time) * 1000
        
        # Create result
        result = EndToEndTestResult(
            success=True,
            total_duration_ms=total_duration,
            audio_processing_ms=audio_processing_duration,
            vad_detection_ms=vad_detection_duration,
            buffer_recording_ms=buffer_recording_duration,
            transcription_ms=transcription_duration,
            llm_generation_ms=llm_generation_duration,
            tts_generation_ms=tts_generation_duration,
            metrics={
                "frames_generated": len(frames),
                "frames_processed": len(processed_frames),
                "vad_states": [state.name for state in vad_results],
                "recorded_audio_bytes": len(recorded_audio),
                "transcription": mock_transcription,
                "llm_response": mock_llm_response,
                "tts_audio_bytes": len(mock_tts_audio)
            }
        )
        
        # Print summary
        print("\n" + "="*60)
        print("END-TO-END TEST RESULTS")
        print("="*60)
        print(f"Total Duration: {result.total_duration_ms:.2f}ms")
        print(f"  Audio Processing: {result.audio_processing_ms:.2f}ms")
        print(f"  VAD Detection: {result.vad_detection_ms:.2f}ms")
        print(f"  Buffer Recording: {result.buffer_recording_ms:.2f}ms")
        print(f"  Transcription: {result.transcription_ms:.2f}ms (mocked)")
        print(f"  LLM Generation: {result.llm_generation_ms:.2f}ms (mocked)")
        print(f"  TTS Generation: {result.tts_generation_ms:.2f}ms (mocked)")
        print("="*60)
        
        # Assertions
        assert result.success, "End-to-end test failed"
        assert result.total_duration_ms < 10000, "Total duration exceeds 10 seconds (processing overhead)"
        assert result.audio_processing_ms < 1000, "Audio processing too slow"
        assert result.vad_detection_ms < 1000, "VAD detection too slow"
        
    @pytest.mark.asyncio
    async def test_pipeline_with_no_think_prefix(self, pcm_test_files):
        """
        Test that /no_think prefix is automatically injected.
        
        This test verifies Phase C requirement for automatic prefix injection.
        """
        if "q1_arabic" not in pcm_test_files:
            pytest.skip("Q1 Arabic PCM file not available")
        
        # Mock the voice service adapter
        mock_voice_service = AsyncMock(spec=SimpleVoiceService)
        mock_voice_service.process_transcription.return_value = "Response text"
        
        # Create adapter
        config = {
            "auto_no_think_prefix": True,
            "language": "ar"
        }
        
        with patch('beautyai_inference.services.voice.webrtc_voice_service_adapter.SimpleVoiceService', 
                   return_value=mock_voice_service):
            adapter = WebRTCVoiceServiceAdapter(config, mock_voice_service)
            await adapter.initialize()
            
            # Process a mock transcription
            user_input = "ما هو الذكاء الاصطناعي؟"
            await adapter.process_transcription(user_input, session_id="test_session")
            
            # Verify /no_think prefix was added
            mock_voice_service.process_transcription.assert_called_once()
            call_args = mock_voice_service.process_transcription.call_args
            
            # Check that the input was modified to include /no_think prefix
            processed_input = call_args[0][0] if call_args[0] else call_args.kwargs.get('text', '')
            assert processed_input.startswith("/no_think "), "Automatic /no_think prefix not injected"
            assert user_input in processed_input, "Original user input not preserved"
            
            print(f"✅ /no_think prefix automatically injected: {processed_input[:50]}...")
    
    @pytest.mark.asyncio
    async def test_pipeline_performance_slo_compliance(self, pcm_test_files, audio_processor, 
                                                       vad_service, buffer_manager):
        """
        Test that pipeline meets SLO requirements.
        
        Migration Plan SLOs (90th percentile):
        - Round-trip ≤ 6s
        - STT ≤ 2s
        - LLM ≤ 3s
        - TTS ≤ 1s
        
        Note: This test measures processing overhead only, not actual AI model latency.
        """
        if "q1_arabic" not in pcm_test_files:
            pytest.skip("Q1 Arabic PCM file not available")
        
        test_file = pcm_test_files["q1_arabic"]
        
        # Run multiple iterations to get p90 metrics
        iterations = 5
        results = []
        
        for i in range(iterations):
            start_time = time.time()
            
            # Simulate pipeline
            simulator = PCMAudioSimulator(test_file, frame_duration_ms=30)
            frames = simulator.generate_frames()
            
            # Process all frames
            for frame in frames:
                pcm_output = await audio_processor.process_frame(frame)
                if pcm_output:
                    vad_state = await vad_service.process_audio(pcm_output)
                    await buffer_manager.feed_audio(pcm_output, vad_state)
            
            duration_ms = (time.time() - start_time) * 1000
            results.append(duration_ms)
        
        # Calculate p90
        results.sort()
        p90_index = int(len(results) * 0.9)
        p90_latency = results[p90_index] if p90_index < len(results) else results[-1]
        
        print(f"\nPerformance Results ({iterations} iterations):")
        print(f"  Min: {min(results):.2f}ms")
        print(f"  Max: {max(results):.2f}ms")
        print(f"  Avg: {sum(results)/len(results):.2f}ms")
        print(f"  P90: {p90_latency:.2f}ms")
        
        # Assert processing overhead is minimal (< 2s for audio processing)
        assert p90_latency < 2000, f"P90 processing latency {p90_latency}ms exceeds 2s threshold"
        print("✅ Pipeline meets processing overhead SLO (<2s)")
    
    @pytest.mark.asyncio
    async def test_pipeline_memory_efficiency(self, pcm_test_files, audio_processor, 
                                              vad_service, buffer_manager):
        """
        Test memory efficiency of WebRTC pipeline.
        
        Expected memory footprint per session: ~40 MB (from Phase C report)
        """
        if "q1_arabic" not in pcm_test_files:
            pytest.skip("Q1 Arabic PCM file not available")
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Measure baseline memory
        baseline_memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Run pipeline
        test_file = pcm_test_files["q1_arabic"]
        simulator = PCMAudioSimulator(test_file, frame_duration_ms=30)
        frames = simulator.generate_frames()
        
        for frame in frames:
            pcm_output = await audio_processor.process_frame(frame)
            if pcm_output:
                vad_state = await vad_service.process_audio(pcm_output)
                await buffer_manager.feed_audio(pcm_output, vad_state)
        
        # Measure peak memory
        peak_memory_mb = process.memory_info().rss / 1024 / 1024
        memory_increase_mb = peak_memory_mb - baseline_memory_mb
        
        print(f"\nMemory Usage:")
        print(f"  Baseline: {baseline_memory_mb:.2f} MB")
        print(f"  Peak: {peak_memory_mb:.2f} MB")
        print(f"  Increase: {memory_increase_mb:.2f} MB")
        
        # Assert memory usage is reasonable (<100 MB increase)
        assert memory_increase_mb < 100, f"Memory increase {memory_increase_mb:.2f}MB exceeds 100MB threshold"
        print(f"✅ Memory usage is efficient (<100MB increase)")


@pytest.mark.skipif(not WEBRTC_AVAILABLE, reason="WebRTC components not available")
class TestWebRTCErrorHandling:
    """Test error handling in WebRTC pipeline."""
    
    @pytest.mark.asyncio
    async def test_invalid_audio_frame_handling(self):
        """Test handling of invalid audio frames."""
        config = {
            "max_utterance_sec": 10,
            "sample_rate": 16000,
            "channels": 1
        }
        processor = WebRTCAudioProcessor(config)
        await processor.initialize()
        
        # Test with invalid frame (corrupted data)
        invalid_frame = MockAudioFrame(b'invalid_data', sample_rate=48000)
        
        try:
            result = await processor.process_frame(invalid_frame)
            # Should handle gracefully, may return None or empty data
            print(f"Invalid frame handled gracefully: {result is None}")
        except Exception as e:
            pytest.fail(f"Invalid frame caused unhandled exception: {e}")
        finally:
            await processor.cleanup()
    
    @pytest.mark.asyncio
    async def test_utterance_limit_enforcement(self):
        """Test 10-second utterance limit enforcement."""
        config = {
            "max_utterance_sec": 1,  # 1 second for quick testing
            "sample_rate": 16000,
            "channels": 1
        }
        processor = WebRTCAudioProcessor(config)
        await processor.initialize()
        
        # Generate frames for 2 seconds (should exceed limit)
        frame_duration_ms = 30
        frames_per_second = 1000 // frame_duration_ms
        total_frames = frames_per_second * 2  # 2 seconds
        
        limit_exceeded = False
        
        for i in range(total_frames):
            # Create dummy frame
            samples_per_frame = (16000 * frame_duration_ms) // 1000
            pcm_data = b'\x00' * (samples_per_frame * 2)
            frame = MockAudioFrame(pcm_data, sample_rate=16000)
            
            try:
                result = await processor.process_frame(frame)
                if processor.utterance_exceeded:
                    limit_exceeded = True
                    break
            except Exception as e:
                if "utterance limit" in str(e).lower():
                    limit_exceeded = True
                    break
        
        assert limit_exceeded, "Utterance limit was not enforced"
        print("✅ 10-second utterance limit enforced")
        
        await processor.cleanup()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
