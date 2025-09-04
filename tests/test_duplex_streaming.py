"""
Integration tests for full duplex voice streaming.

Tests the complete duplex streaming pipeline including:
- Echo suppression with synthetic audio
- Barge-in scenarios and VAD detection  
- WebSocket message ordering and protocol
- Device selection and persistence
- Fallback modes and error handling
"""

import pytest
import asyncio
import numpy as np
import json
import time
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, AsyncMock
import websockets
from websockets.exceptions import WebSocketException

# Import components under test
from beautyai_inference.services.voice.utils.echo_detector import EchoDetector, EchoMetrics
from beautyai_inference.services.voice.echo_suppression import EchoSuppressor, EchoState
from beautyai_inference.services.voice.streaming.metrics import SessionMetrics, DuplexMetrics
from beautyai_inference.api.endpoints.streaming_voice import pack_binary_frame, unpack_binary_frame


class TestEchoDetection:
    """Test suite for echo detection functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.detector = EchoDetector(
            sample_rate=16000,
            frame_size_ms=20,
            correlation_threshold=0.3,
            spectral_threshold=0.4,
            max_delay_ms=500,
            adaptive_threshold=True
        )
    
    def test_no_echo_silent_audio(self):
        """Test that silent audio produces no echo detection."""
        mic_frame = np.zeros(320, dtype=np.float32)  # 20ms at 16kHz
        tts_frame = np.zeros(320, dtype=np.float32)
        
        metrics = self.detector.process_audio_frames(mic_frame, tts_frame)
        
        assert metrics.echo_probability < 0.1
        assert metrics.correlation_score < 0.1
        assert metrics.confidence > 0.8  # High confidence in "no echo"
    
    def test_synthetic_echo_detection(self):
        """Test echo detection with synthetic echo signal."""
        # Generate test signal
        sample_rate = 16000
        duration_ms = 100
        samples = int(sample_rate * duration_ms / 1000)
        
        # Create sine wave signal
        t = np.linspace(0, duration_ms/1000, samples)
        frequency = 440  # A4 note
        original_signal = np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        # Create echo with delay and attenuation
        delay_samples = 80  # ~5ms delay at 16kHz
        echo_signal = np.zeros_like(original_signal)
        echo_signal[delay_samples:] = original_signal[:-delay_samples] * 0.3  # 30% echo
        
        # Add some noise to make it realistic
        noise = np.random.normal(0, 0.05, samples).astype(np.float32)
        mic_input = original_signal + echo_signal + noise
        
        # Process in chunks
        chunk_size = 320  # 20ms chunks
        max_echo_prob = 0.0
        max_correlation = 0.0
        
        for i in range(0, len(original_signal) - chunk_size, chunk_size):
            mic_chunk = mic_input[i:i+chunk_size]
            tts_chunk = original_signal[i:i+chunk_size]
            
            metrics = self.detector.process_audio_frames(mic_chunk, tts_chunk)
            max_echo_prob = max(max_echo_prob, metrics.echo_probability)
            max_correlation = max(max_correlation, metrics.correlation_score)
        
        # Should detect echo
        assert max_echo_prob > 0.6, f"Expected echo probability > 0.6, got {max_echo_prob}"
        assert max_correlation > 0.4, f"Expected correlation > 0.4, got {max_correlation}"
    
    def test_adaptive_threshold_learning(self):
        """Test that adaptive thresholds learn from background noise."""
        # Feed background noise for several frames
        for _ in range(20):
            noise_frame = np.random.normal(0, 0.1, 320).astype(np.float32)
            self.detector.process_audio_frames(noise_frame, noise_frame)
        
        initial_thresholds = self.detector.get_adaptive_thresholds()
        
        # Feed more frames with higher correlation
        high_corr_signal = np.sin(np.linspace(0, 2*np.pi, 320)).astype(np.float32)
        for _ in range(10):
            self.detector.process_audio_frames(high_corr_signal, high_corr_signal)
        
        final_thresholds = self.detector.get_adaptive_thresholds()
        
        # Thresholds should have adapted
        assert final_thresholds[0] != initial_thresholds[0]  # Correlation threshold changed
        assert final_thresholds[1] != initial_thresholds[1]  # Spectral threshold changed
    
    def test_detector_statistics(self):
        """Test that detector statistics are properly tracked."""
        # Process some frames
        for i in range(10):
            frame = np.sin(np.linspace(0, 2*np.pi*i, 320)).astype(np.float32)
            self.detector.process_audio_frames(frame, frame)
        
        stats = self.detector.get_statistics()
        
        assert stats["frames_processed"] == 10
        assert "correlation_threshold" in stats
        assert "spectral_threshold" in stats
        assert "buffer_size_mic" in stats
        assert "buffer_size_tts" in stats
        assert stats["adaptive_threshold_enabled"] is True


class TestEchoSuppression:
    """Test suite for echo suppression service."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.suppressor = EchoSuppressor(
            sample_rate=16000,
            vad_threshold=0.02,
            echo_threshold=0.3,
            barge_in_delay_ms=300,
            resume_delay_ms=500
        )
    
    def test_state_transitions(self):
        """Test echo suppression state machine transitions."""
        # Start in IDLE state
        assert self.suppressor.get_state() == EchoState.IDLE
        
        # Start TTS playback -> should go to TTS_PLAYING
        self.suppressor.start_tts_playback()
        assert self.suppressor.get_state() == EchoState.TTS_PLAYING
        
        # Simulate user speech during TTS -> should trigger barge-in
        speech_frame = np.random.normal(0, 0.1, 320).astype(np.float32)  # Above VAD threshold
        result = self.suppressor.process_mic_audio(speech_frame)
        
        # Should eventually transition to BARGE_IN state
        # May need multiple frames to trigger
        for _ in range(10):
            result = self.suppressor.process_mic_audio(speech_frame)
            if self.suppressor.get_state() == EchoState.BARGE_IN:
                break
        
        assert self.suppressor.get_state() == EchoState.BARGE_IN
        assert result.should_gate is False  # Don't gate during barge-in
    
    def test_mic_gating_during_tts(self):
        """Test that mic input is gated during TTS playback without speech."""
        self.suppressor.start_tts_playbook()
        
        # Silent mic input should be gated
        silent_frame = np.zeros(320, dtype=np.float32)
        result = self.suppressor.process_mic_audio(silent_frame)
        
        assert result.should_gate is True
        assert result.vad_active is False
        
        # Low-level noise should still be gated
        noise_frame = np.random.normal(0, 0.005, 320).astype(np.float32)  # Below threshold
        result = self.suppressor.process_mic_audio(noise_frame)
        
        assert result.should_gate is True
        assert result.vad_active is False
    
    def test_tts_ducking_during_barge_in(self):
        """Test TTS audio ducking during barge-in."""
        self.suppressor.start_tts_playback()
        
        # Trigger barge-in with speech
        speech_frame = np.ones(320, dtype=np.float32) * 0.1  # Clear speech signal
        
        # Process multiple frames to trigger barge-in
        for _ in range(15):  # Exceed barge_in_delay_ms frames
            self.suppressor.process_mic_audio(speech_frame)
        
        # TTS should be ducked
        tts_frame = np.ones(320, dtype=np.float32) * 0.5  # Original TTS signal
        tts_result = self.suppressor.process_tts_audio(tts_frame)
        
        # Should be ducked (reduced volume)
        assert tts_result.should_duck is True
        assert np.mean(tts_result.processed_audio) < np.mean(tts_frame)
    
    def test_metrics_tracking(self):
        """Test that suppression metrics are properly tracked."""
        # Generate some activity
        self.suppressor.start_tts_playback()
        
        for i in range(20):
            # Alternate between speech and silence
            if i % 4 == 0:
                frame = np.random.normal(0, 0.1, 320).astype(np.float32)  # Speech
            else:
                frame = np.random.normal(0, 0.005, 320).astype(np.float32)  # Silence
            
            self.suppressor.process_mic_audio(frame)
        
        metrics = self.suppressor.get_metrics()
        
        assert metrics.frames_processed == 20
        assert metrics.vad_active_frames > 0
        assert metrics.gated_frames > 0
        assert len(metrics.recent_correlations) > 0


class TestDuplexMetrics:
    """Test suite for duplex streaming metrics."""
    
    def setup_method(self):
        """Setup test fixtures.""" 
        self.session_metrics = SessionMetrics(
            session_id="test_session_duplex",
            duplex_enabled=True
        )
    
    def test_duplex_metrics_initialization(self):
        """Test that duplex metrics are properly initialized."""
        assert self.session_metrics.duplex_enabled is True
        assert self.session_metrics.duplex_metrics is not None
        assert isinstance(self.session_metrics.duplex_metrics, DuplexMetrics)
    
    def test_tts_metrics_tracking(self):
        """Test TTS streaming metrics tracking."""
        # Simulate TTS streaming
        self.session_metrics.update_tts_first_byte(125.5)  # 125.5ms first byte latency
        self.session_metrics.update_tts_stream_duration(2340.0)  # 2.34s duration
        self.session_metrics.update_tts_chunk(1024)  # 1KB chunk
        self.session_metrics.update_tts_chunk(512)   # 512B chunk
        
        snapshot = self.session_metrics.snapshot()
        
        # Check duplex metrics are present
        assert "duplex" in snapshot
        duplex_data = snapshot["duplex"]
        
        # Check TTS metrics
        assert duplex_data["tts_first_byte_ms"]["count"] == 1
        assert duplex_data["tts_first_byte_ms"]["mean"] == 125.5
        assert duplex_data["tts_stream_duration_ms"]["count"] == 1
        assert duplex_data["counts"]["tts_chunks_sent"] == 2
        assert duplex_data["totals"]["tts_bytes_sent"] == 1536
    
    def test_echo_and_barge_in_metrics(self):
        """Test echo correlation and barge-in metrics."""
        # Simulate echo detection and barge-in events
        self.session_metrics.update_echo_correlation(0.15)
        self.session_metrics.update_echo_correlation(0.05)
        self.session_metrics.update_echo_correlation(0.25)
        
        self.session_metrics.inc_barge_in()
        self.session_metrics.inc_barge_in()
        
        self.session_metrics.inc_playback_stall()
        
        snapshot = self.session_metrics.snapshot()
        duplex_data = snapshot["duplex"]
        
        # Check echo metrics
        echo_data = duplex_data["echo_correlation_score"]
        assert echo_data["count"] == 3
        assert echo_data["mean"] == 0.15  # (0.15 + 0.05 + 0.25) / 3
        
        # Check event counts
        counts = duplex_data["counts"]
        assert counts["barge_ins"] == 2
        assert counts["playback_stalls"] == 1
    
    def test_jitter_buffer_metrics(self):
        """Test jitter buffer size tracking."""
        # Simulate varying buffer sizes
        buffer_sizes = [80.0, 120.0, 95.0, 110.0, 150.0]
        for size in buffer_sizes:
            self.session_metrics.update_jitter_buffer_size(size)
        
        snapshot = self.session_metrics.snapshot()
        jitter_data = snapshot["duplex"]["jitter_buffer_size_ms"]
        
        assert jitter_data["count"] == 5
        assert jitter_data["min"] == 80.0
        assert jitter_data["max"] == 150.0
        assert jitter_data["mean"] == 111.0  # Average of buffer sizes


class TestBinaryProtocol:
    """Test suite for binary WebSocket protocol."""
    
    def test_pack_unpack_mic_chunk(self):
        """Test binary frame packing/unpacking for mic chunks."""
        # Create test audio data
        audio_data = np.random.randint(-32768, 32767, 320, dtype=np.int16)
        timestamp = int(time.time() * 1000)
        sequence = 123
        
        # Pack frame
        packed_frame = pack_binary_frame(
            frame_type=0x01,  # mic_chunk
            sequence=sequence,
            flags=0x01,  # start flag
            timestamp=timestamp,
            payload=audio_data.tobytes()
        )
        
        # Unpack frame
        unpacked = unpack_binary_frame(packed_frame)
        
        assert unpacked["type"] == 0x01
        assert unpacked["sequence"] == sequence
        assert unpacked["flags"] == 0x01
        assert unpacked["timestamp"] == timestamp
        
        # Check audio data integrity
        unpacked_audio = np.frombuffer(unpacked["payload"], dtype=np.int16)
        np.testing.assert_array_equal(audio_data, unpacked_audio)
    
    def test_pack_unpack_tts_chunk(self):
        """Test binary frame packing/unpacking for TTS chunks."""
        # Create test TTS data (Opus encoded)
        tts_data = b'\x01\x02\x03\x04\x05' * 100  # Simulated Opus data
        timestamp = int(time.time() * 1000)
        sequence = 456
        
        # Pack frame  
        packed_frame = pack_binary_frame(
            frame_type=0x02,  # tts_chunk
            sequence=sequence,
            flags=0x02,  # end flag
            timestamp=timestamp,
            payload=tts_data
        )
        
        # Unpack frame
        unpacked = unpack_binary_frame(packed_frame)
        
        assert unpacked["type"] == 0x02
        assert unpacked["sequence"] == sequence
        assert unpacked["flags"] == 0x02
        assert unpacked["timestamp"] == timestamp
        assert unpacked["payload"] == tts_data
    
    def test_pack_unpack_control_message(self):
        """Test binary frame packing/unpacking for control messages."""
        # Create control message
        control_msg = {
            "action": "pause_tts",
            "reason": "barge_in_detected",
            "metadata": {"correlation": 0.75}
        }
        control_data = json.dumps(control_msg).encode('utf-8')
        
        # Pack frame
        packed_frame = pack_binary_frame(
            frame_type=0x03,  # control
            sequence=0,  # No sequence for control messages
            flags=0x04,  # urgent flag
            timestamp=int(time.time() * 1000),
            payload=control_data
        )
        
        # Unpack frame
        unpacked = unpack_binary_frame(packed_frame)
        
        assert unpacked["type"] == 0x03
        assert unpacked["flags"] == 0x04
        
        # Parse control message
        parsed_control = json.loads(unpacked["payload"].decode('utf-8'))
        assert parsed_control["action"] == "pause_tts"
        assert parsed_control["reason"] == "barge_in_detected"
        assert parsed_control["metadata"]["correlation"] == 0.75


@pytest.mark.asyncio
class TestDuplexStreamingIntegration:
    """Integration tests for the complete duplex streaming pipeline."""
    
    @pytest.fixture
    async def mock_websocket_server(self):
        """Mock WebSocket server for testing."""
        connected_clients = []
        
        async def server_handler(websocket, path):
            connected_clients.append(websocket)
            try:
                async for message in websocket:
                    # Echo binary messages back for testing
                    if isinstance(message, bytes):
                        unpacked = unpack_binary_frame(message)
                        
                        # Simulate TTS response for mic chunks
                        if unpacked["type"] == 0x01:  # mic_chunk
                            tts_response = pack_binary_frame(
                                frame_type=0x02,  # tts_chunk
                                sequence=unpacked["sequence"] + 1000,
                                flags=0x01,  # start
                                timestamp=int(time.time() * 1000),
                                payload=b"fake_tts_data" * 50
                            )
                            await websocket.send(tts_response)
                    
            except WebSocketException:
                pass
            finally:
                if websocket in connected_clients:
                    connected_clients.remove(websocket)
        
        server = await websockets.serve(server_handler, "localhost", 0)
        port = server.server.sockets[0].getsockname()[1]
        
        yield f"ws://localhost:{port}"
        
        server.close()
        await server.wait_closed()
    
    async def test_duplex_websocket_communication(self, mock_websocket_server):
        """Test bidirectional WebSocket communication with binary protocol."""
        uri = mock_websocket_server
        
        async with websockets.connect(uri) as websocket:
            # Send mic chunk
            mic_data = np.random.randint(-1000, 1000, 320, dtype=np.int16)
            mic_frame = pack_binary_frame(
                frame_type=0x01,
                sequence=1,
                flags=0x01,
                timestamp=int(time.time() * 1000),
                payload=mic_data.tobytes()
            )
            
            await websocket.send(mic_frame)
            
            # Receive TTS response
            response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
            unpacked_response = unpack_binary_frame(response)
            
            assert unpacked_response["type"] == 0x02  # tts_chunk
            assert unpacked_response["sequence"] == 1001  # sequence + 1000
            assert len(unpacked_response["payload"]) > 0
    
    async def test_message_ordering_and_sequencing(self, mock_websocket_server):
        """Test that messages maintain proper ordering and sequencing."""
        uri = mock_websocket_server
        
        async with websockets.connect(uri) as websocket:
            # Send multiple frames with different sequences
            sequences = [10, 11, 12, 13, 14]
            
            for seq in sequences:
                frame = pack_binary_frame(
                    frame_type=0x01,
                    sequence=seq,
                    flags=0x00,
                    timestamp=int(time.time() * 1000),
                    payload=b"test_data"
                )
                await websocket.send(frame)
            
            # Collect responses
            responses = []
            for _ in sequences:
                response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                unpacked = unpack_binary_frame(response)
                responses.append(unpacked["sequence"])
            
            # Check sequence ordering (should be original + 1000)
            expected_sequences = [seq + 1000 for seq in sequences]
            assert responses == expected_sequences


class TestDuplexFallbackModes:
    """Test suite for duplex streaming fallback modes."""
    
    def test_half_duplex_fallback(self):
        """Test fallback to half-duplex mode when full duplex fails."""
        # Simulate full duplex failure by creating suppressor with high echo detection
        suppressor = EchoSuppressor(
            sample_rate=16000,
            echo_threshold=0.1,  # Very low threshold - will trigger easily
            adaptive_mode=True
        )
        
        # Simulate high echo conditions
        high_echo_signal = np.ones(320, dtype=np.float32) * 0.5
        
        for _ in range(20):
            suppressor.process_mic_audio(high_echo_signal)
        
        metrics = suppressor.get_metrics()
        
        # Should recommend fallback mode
        assert metrics.echo_correlation_mean > 0.3
        
        # Check that suppressor recommends half-duplex
        recommendation = suppressor.get_duplex_recommendation()
        assert recommendation["recommended_mode"] in ["half", "off"]
        assert recommendation["reason"] == "high_echo_detected"
    
    def test_device_compatibility_fallback(self):
        """Test fallback when audio devices don't support echo cancellation."""
        # This would be tested with actual device enumeration in real scenario
        # For now, simulate device capability detection
        
        mock_device_caps = {
            "echo_cancellation": False,
            "noise_suppression": True,
            "auto_gain_control": False
        }
        
        # Should recommend constraints without echo cancellation
        recommended_constraints = self._get_recommended_constraints(mock_device_caps)
        
        assert recommended_constraints["echoCancellation"] is False
        assert recommended_constraints["noiseSuppression"] is True
        assert "fallback_mode" in recommended_constraints
        assert recommended_constraints["fallback_mode"] == "software_echo_suppression"
    
    def _get_recommended_constraints(self, device_caps: Dict[str, bool]) -> Dict[str, Any]:
        """Helper to get recommended audio constraints based on device capabilities."""
        constraints = {
            "sampleRate": 48000,
            "channelCount": 1,
            "echoCancellation": device_caps.get("echo_cancellation", True),
            "noiseSuppression": device_caps.get("noise_suppression", True),
            "autoGainControl": device_caps.get("auto_gain_control", True)
        }
        
        # Add fallback mode if hardware echo cancellation not available
        if not device_caps.get("echo_cancellation", True):
            constraints["fallback_mode"] = "software_echo_suppression"
        
        return constraints


if __name__ == "__main__":
    pytest.main([__file__, "-v"])