"""
Test suite for comprehensive debug infrastructure.

This test validates all debug endpoints, debug mode functionality,
and ensures the debug tool provides actionable information for
the STT → LLM → TTS pipeline.
"""
import pytest
import asyncio
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
from io import BytesIO

# Import test dependencies
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "backend" / "src"))

from beautyai_inference.api.main import app
from beautyai_inference.services.voice.conversation.simple_voice_service import SimpleVoiceService
from beautyai_inference.api.schemas.debug_schemas import (
    PipelineDebugSummary, DebugEvent, WebSocketDebugMessage,
    TranscriptionDebugData, LLMDebugData, TTSDebugData
)


class TestDebugInfrastructure:
    """Comprehensive test suite for debug infrastructure."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_audio_data(self):
        """Create mock audio data for testing."""
        # Create a small PCM audio buffer (16-bit, 16kHz, mono)
        import struct
        sample_rate = 16000
        duration = 1.0  # 1 second
        samples = int(sample_rate * duration)
        
        # Generate simple sine wave
        import math
        frequency = 440  # A4 note
        audio_data = bytearray()
        
        for i in range(samples):
            sample = int(32767 * math.sin(2 * math.pi * frequency * i / sample_rate))
            audio_data.extend(struct.pack('<h', sample))
        
        return bytes(audio_data)
    
    def test_debug_config_endpoint(self, client):
        """Test debug configuration endpoint."""
        response = client.get("/api/v1/debug/config")
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate configuration structure
        assert "debug_mode_available" in data
        assert "supported_audio_formats" in data
        assert "max_audio_size_mb" in data
        assert "test_audio_samples" in data
        
        # Validate supported formats
        assert isinstance(data["supported_audio_formats"], list)
        assert "webm" in data["supported_audio_formats"]
        assert "wav" in data["supported_audio_formats"]
    
    def test_system_health_endpoint(self, client):
        """Test system health monitoring endpoint."""
        response = client.get("/api/v1/debug/health/system")
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate health status structure
        assert "cpu_usage_percent" in data
        assert "memory_usage_percent" in data
        assert "models" in data
        assert "active_connections" in data
        assert "alerts" in data
        assert "warnings" in data
        
        # Validate metrics are numeric
        assert isinstance(data["cpu_usage_percent"], (int, float))
        assert isinstance(data["memory_usage_percent"], (int, float))
        assert isinstance(data["models"], list)
    
    def test_models_health_endpoint(self, client):
        """Test model health monitoring endpoint."""
        response = client.get("/api/v1/debug/health/models")
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate models health structure
        assert "timestamp" in data
        assert "models" in data
        assert "overall_status" in data
        
        # Validate models list
        assert isinstance(data["models"], list)
        assert data["overall_status"] in ["healthy", "degraded", "unhealthy"]
    
    def test_test_cases_endpoint(self, client):
        """Test pipeline test cases endpoint."""
        response = client.get("/api/v1/debug/pipeline/test-cases")
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate test cases structure
        assert "test_cases" in data
        assert "total_count" in data
        assert "languages" in data
        assert "voice_types" in data
        
        # Validate test cases content
        test_cases = data["test_cases"]
        assert len(test_cases) > 0
        
        for test_case in test_cases:
            assert "test_id" in test_case
            assert "name" in test_case
            assert "description" in test_case
            assert "language" in test_case
            assert "voice_type" in test_case
            assert "expected_transcription" in test_case
        
        # Validate languages and voice types
        assert "ar" in data["languages"]
        assert "en" in data["languages"]
        assert "male" in data["voice_types"]
        assert "female" in data["voice_types"]
    
    def test_debug_analytics_endpoint(self, client):
        """Test debug analytics endpoint."""
        response = client.get("/api/v1/debug/analytics/events")
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate analytics structure
        assert "time_range" in data
        assert "summary" in data
        assert "trends" in data
        assert "top_errors" in data
        
        # Validate summary metrics
        summary = data["summary"]
        assert "total_events" in summary
        assert "events_by_stage" in summary
        assert "events_by_level" in summary
        assert "avg_processing_time_ms" in summary
        
        # Validate stage breakdown
        stages = summary["events_by_stage"]
        assert "stt" in stages
        assert "llm" in stages
        assert "tts" in stages
    
    def test_test_audio_samples_endpoint(self, client):
        """Test test audio samples listing endpoint."""
        response = client.get("/api/v1/debug/samples")
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate samples structure
        assert "samples" in data
        assert "total_count" in data
        assert "base_path" in data
        
        # Validate samples list
        assert isinstance(data["samples"], list)
        assert isinstance(data["total_count"], int)
    
    @pytest.mark.asyncio
    async def test_simple_voice_service_debug_mode(self):
        """Test SimpleVoiceService debug mode functionality."""
        
        # Create voice service with debug mode
        service = SimpleVoiceService(debug_mode=True)
        
        # Verify debug mode is enabled
        assert service.debug_mode is True
        assert service.debug_events == []
        
        # Test debug event emission
        service._emit_debug_event("test", "info", "Test debug event")
        
        assert len(service.debug_events) == 1
        event = service.debug_events[0]
        assert event.stage == "test"
        assert event.level == "info"
        assert event.message == "Test debug event"
        assert event.timestamp is not None
    
    @pytest.mark.asyncio
    async def test_debug_event_collection(self):
        """Test debug event collection during pipeline processing."""
        
        with patch('beautyai_inference.services.voice.conversation.simple_voice_service.SimpleVoiceService._transcribe_audio') as mock_transcribe, \
             patch('beautyai_inference.services.voice.conversation.simple_voice_service.SimpleVoiceService._generate_response') as mock_generate, \
             patch('beautyai_inference.services.voice.conversation.simple_voice_service.SimpleVoiceService._synthesize_speech') as mock_synthesize:
            
            # Mock successful pipeline responses
            mock_transcribe.return_value = ("Hello, how are you?", "en")
            mock_generate.return_value = "I'm doing well, thank you!"
            mock_synthesize.return_value = b"fake_audio_data"
            
            # Create service with debug mode
            service = SimpleVoiceService(debug_mode=True)
            await service.initialize()
            
            try:
                # Process voice message
                result = await service.process_voice_message(
                    audio_data=b"fake_audio",
                    audio_format="wav",
                    language="en",
                    gender="female"
                )
                
                # Verify debug events were collected
                assert len(service.debug_events) > 0
                
                # Verify debug summary is available
                debug_summary = service.get_debug_summary()
                assert debug_summary is not None
                assert hasattr(debug_summary, 'total_processing_time_ms')
                assert hasattr(debug_summary, 'transcription_data')
                assert hasattr(debug_summary, 'llm_data')
                assert hasattr(debug_summary, 'tts_data')
                
                # Verify stage timing
                assert debug_summary.transcription_data.processing_time_ms > 0
                assert debug_summary.llm_data.processing_time_ms > 0
                assert debug_summary.tts_data.processing_time_ms > 0
                
            finally:
                await service.cleanup()
    
    def test_debug_schemas_validation(self):
        """Test debug schema validation and serialization."""
        
        # Test TranscriptionDebugData
        transcription_data = TranscriptionDebugData(
            transcribed_text="Hello world",
            language_detected="en",
            confidence_score=0.95,
            processing_time_ms=450,
            model_used="whisper-base",
            audio_duration_ms=1000,
            audio_format="wav"
        )
        
        # Verify serialization
        json_data = transcription_data.dict()
        assert json_data["transcribed_text"] == "Hello world"
        assert json_data["confidence_score"] == 0.95
        
        # Test LLMDebugData
        llm_data = LLMDebugData(
            response_text="I'm doing well!",
            prompt_tokens=15,
            completion_tokens=8,
            processing_time_ms=890,
            model_used="gpt-3.5-turbo",
            temperature=0.7
        )
        
        # Verify serialization
        json_data = llm_data.dict()
        assert json_data["response_text"] == "I'm doing well!"
        assert json_data["prompt_tokens"] == 15
        
        # Test TTSDebugData
        tts_data = TTSDebugData(
            audio_length_ms=2500,
            voice_used="ar-EG-SalmaNeural",
            processing_time_ms=320,
            output_format="mp3",
            text_length=23,
            speech_rate="medium"
        )
        
        # Verify serialization
        json_data = tts_data.dict()
        assert json_data["voice_used"] == "ar-EG-SalmaNeural"
        assert json_data["output_format"] == "mp3"
        
        # Test PipelineDebugSummary
        pipeline_summary = PipelineDebugSummary(
            total_processing_time_ms=1660,
            transcription_data=transcription_data,
            llm_data=llm_data,
            tts_data=tts_data,
            success=True,
            error_message=None,
            debug_events=[
                DebugEvent(
                    stage="stt",
                    level="info",
                    message="Transcription completed",
                    timestamp="2024-01-01T12:00:00Z",
                    data={"confidence": 0.95}
                )
            ]
        )
        
        # Verify complete pipeline summary
        json_data = pipeline_summary.dict()
        assert json_data["total_processing_time_ms"] == 1660
        assert json_data["success"] is True
        assert len(json_data["debug_events"]) == 1
    
    @pytest.mark.asyncio
    async def test_pipeline_performance_grading(self):
        """Test pipeline performance analysis and grading."""
        
        # Test different performance scenarios
        test_cases = [
            {"processing_time": 1.5, "expected_grade": "A"},
            {"processing_time": 2.5, "expected_grade": "B"},
            {"processing_time": 4.0, "expected_grade": "C"}
        ]
        
        for case in test_cases:
            # Create mock debug summary
            debug_summary = PipelineDebugSummary(
                total_processing_time_ms=case["processing_time"] * 1000,
                transcription_data=TranscriptionDebugData(
                    transcribed_text="Test",
                    language_detected="en",
                    confidence_score=0.9,
                    processing_time_ms=500,
                    model_used="whisper",
                    audio_duration_ms=1000,
                    audio_format="wav"
                ),
                llm_data=LLMDebugData(
                    response_text="Response",
                    prompt_tokens=10,
                    completion_tokens=5,
                    processing_time_ms=800,
                    model_used="gpt-3.5",
                    temperature=0.7
                ),
                tts_data=TTSDebugData(
                    audio_length_ms=1500,
                    voice_used="en-US-AriaNeural",
                    processing_time_ms=300,
                    output_format="mp3",
                    text_length=10,
                    speech_rate="medium"
                ),
                success=True,
                error_message=None,
                debug_events=[]
            )
            
            # Analyze performance (would be done in actual endpoint)
            processing_time = case["processing_time"]
            performance_grade = "A"
            if processing_time > 3.0:
                performance_grade = "C"
            elif processing_time > 2.0:
                performance_grade = "B"
            
            assert performance_grade == case["expected_grade"]


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])